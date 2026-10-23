"""Verify legacy dual-frame cache, raw provenance and fixed inference spot parity."""
import json
import sys
import time
from pathlib import Path

from dual_common import ROOT, BASE, CACHE, CAMERAS, CHECKPOINT, digest, ancestry
from review_paths import DATA_ROOT
sys.path[:0] = [str(BASE / 'src/activecam_bev'), str(BASE / 'src/legacy_v31')]
from mini_preflight import load_dependencies, ordered_split_frames


def main():
    started = time.monotonic()
    ROOT.mkdir(parents=True, exist_ok=True)
    old = json.loads((CACHE / 'manifest.json').read_text())
    assert old['expected_samples'] == 6019 and old['batch_size'] == 1
    assert digest(CHECKPOINT) == old['checkpoint_sha256']
    source = BASE / 'src/legacy_v31/train_official_nuscenes_nds_kd_v2.py'
    assert digest(source) == old['training_source_sha256'], 'Training source differs from frozen export'
    legacy, NuScenes, _, _, splits = load_dependencies(BASE / 'src/legacy_v31')
    nusc = NuScenes(version='v1.0-trainval', dataroot=str(DATA_ROOT), verbose=False)
    frames = ordered_split_frames(nusc, sorted(splits()['val']))
    tokens = [t for t, _ in frames]
    assert len(tokens) == 6019
    provenance = ancestry(nusc, tokens)
    provenance_path = ROOT / 'dual_image_provenance.json'
    provenance_path.write_text(json.dumps(provenance, separators=(',', ':')))
    manifest = dict(status='preflight', expected_samples=6019, need_temporal_inputs=True,
        precision='FP32', batch_size=1, checkpoint_sha256=old['checkpoint_sha256'],
        legacy_manifest_path=str(CACHE / 'manifest.json'), legacy_manifest_sha256=digest(CACHE / 'manifest.json'),
        training_source_sha256=old['training_source_sha256'], cameras={},
        provenance_sha256=digest(provenance_path), preparer_sha256=digest(Path(__file__)),
        temporal_source='previous same-camera keyframe, scene first frame prev_valid=0',
        detector_parameters=7191090)
    # Indices are fixed before any metric inspection: scene first/early/middle, and another scene.
    scene2 = next(i for i, (_, s) in enumerate(frames) if s != frames[0][1])
    chosen_tokens = [tokens[i] for i in (0, 1, 20, scene2 + 1)]
    selected_cache = {}
    for camera in CAMERAS:
        path = CACHE / camera / 'results_nusc.json'
        assert path.stat().st_size == old['cameras'][camera]['bytes']
        assert digest(path) == old['cameras'][camera]['sha256']
        protocol_path = CACHE / camera / 'evaluation_protocol.json'
        p = json.loads(protocol_path.read_text())
        assert p['evaluated_sample_count'] == p['timing']['camera_pass_count'] == 6019
        assert p['temporal_bev'] and p['temporal_pre_encoder'] and p['parameters'] == 7191090
        assert p['student_eval_cameras'] == [camera] and p['score_threshold'] == .05
        assert p['merge_nms_iou'] == .05 and p['per_view_nms_iou'] == .2 and p['max_boxes_per_sample'] == 500
        records = json.loads(path.read_text())['results']
        assert set(records) == set(tokens)
        selected_cache[camera] = {t: records[t] for t in chosen_tokens}
        manifest['cameras'][camera] = dict(path=str(path), samples=6019, result_sha256=digest(path),
                                          protocol_sha256=digest(protocol_path), protocol=p)
        del records
        print('verified cache', camera, flush=True)
    # Check the current checkpoint/decoder reproduces fixed native batch-one dual-frame outputs.
    import torch
    # The frozen native exporter enables cuDNN autotuning before evaluation.
    torch.backends.cudnn.benchmark = True
    import benchmark_v31_jetson as builder
    from benchmark_activecam_end_to_end import decode_current_camera, compare_records
    envelope_path = BASE / 'configs/fp32_repeatability_guard.json'
    envelope = json.loads(envelope_path.read_text())
    assert envelope['protocol'] == 'all-pairs repeatability across independent production FP32 batch=1 exports'
    manifest.update(numeric_guard_path=str(envelope_path), numeric_guard_sha256=digest(envelope_path),
                    numeric_guard_thresholds=envelope['thresholds'],
                    numeric_guard_scope='Previously frozen implementation smoke guard; not bitwise or task-quality equivalence')
    training, base, model, checkpoint = builder.build_model(CHECKPOINT)
    training.prepare_model_for_eval(model, checkpoint)
    assert sum(p.numel() for p in model.parameters()) == 7191090
    model.cuda().eval()
    torch.set_num_threads(4)
    dataset = training.NuScenesOfficialDetectionDataset(base, str(DATA_ROOT),
        'v1.0-trainval', 'val', *training.make_transforms(base), build_depth=False,
        need_teacher_inputs=False, need_student_inputs=True, need_teacher_depth=False,
        need_student_depth=False, need_temporal_inputs=True, need_history_targets=False,
        build_detection_targets=False, target_scope='student_fov', student_view_mode='front',
        student_camera='CAM_FRONT', student_cameras=list(base.CAM_CHANNELS))
    index_by_token = {t: i for i, t in enumerate(dataset.sample_tokens)}
    checks = []
    with torch.inference_mode():
        for camera_index, camera in enumerate(base.CAM_CHANNELS):
            for token in chosen_tokens:
                item = dataset[(index_by_token[token], camera_index)]
                inputs = [item[i].unsqueeze(0).cuda() for i in list(range(5, 9)) + list(range(18, 25))]
                assert bool(item[24] > .5) == bool(provenance[token][camera]['previous_sd'])
                heads = model.forward_outputs(*inputs)
                boxes = decode_current_camera(training, base, model, nusc, token, camera_index, item, heads)
                boxes = sorted(training.nms_classwise(boxes, .05), key=lambda b: b[9], reverse=True)[:500]
                actual = [training.ego_box_to_nusc_result(nusc, token, box, 'CAM_FRONT') for box in boxes]
                expected = selected_cache[camera][token]
                diagnostic = ROOT / 'inference_spot_diagnostic.json'
                diagnostic.write_text(json.dumps(dict(token=token, camera=camera, actual=actual, expected=expected)))
                row = compare_records(expected, actual)
                exceeded = {key: row[key + '_max_absolute_error'] for key, limit in envelope['thresholds'].items()
                            if row[key + '_max_absolute_error'] > limit}
                assert not exceeded, (token, camera, exceeded)
                checks.append(dict(token=token, camera=camera, errors=row))
    manifest.update(status='complete', inference_spot_checks=checks,
        valid_previous_camera_events=sum(bool(v['previous_sd']) for c in provenance.values() for v in c.values()),
        max_detector_previous_age_s=max((v['timestamp'] - v['previous_timestamp']) / 1e6
                                        for c in provenance.values() for v in c.values() if v['previous_sd']),
        elapsed_seconds=time.monotonic() - started)
    (ROOT / 'dual_cache_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(dict(status='complete', spot_checks=len(checks), elapsed_seconds=manifest['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
