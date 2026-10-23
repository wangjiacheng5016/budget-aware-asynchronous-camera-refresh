"""Dual-frame, same-checkpoint comparisons with explicit raw-image ancestry."""
import argparse
import copy
import gzip
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

from review_paths import ROOT, BASE, CHECKPOINT, DATA_ROOT, EXTERNAL
sys.path[:0] = [str(BASE / 'src/activecam_bev'), str(Path(__file__).resolve().parent)]
from dual_common import raw_context, official_metrics
from activecam_core import CAMERAS, oldest_first_selections, propagate_detection
from mini_preflight import load_dependencies, ordered_split_frames, merge_sources, evaluate_predictions, sha256
from official_history_protocol import camera_capture_timestamps_us
from mechanism_adapters import ab_predict, simple_predict, common_attribute, TRACKING_CLASSES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['none', 'static', 'motion', 'ab3dmot', 'simpletrack'], required=True)
    parser.add_argument('--k', type=int, choices=[3, 4, 6], required=True)
    parser.add_argument('--window', type=float, choices=[1.5], default=1.5)
    parser.add_argument('--smoke-frames', type=int, default=0)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--parallel-smoke', action='store_true')
    args = parser.parse_args()
    name = f'dual_{args.method}_k{args.k}_h{args.window:g}'
    if args.smoke_frames:
        name += '_smoke_v3'
        if args.parallel_smoke:
            name += '_parallel_check'
    output = ROOT / 'mechanism_results' / name
    output.mkdir(parents=True, exist_ok=False)
    status = dict(status='preflight', method=args.method, k=args.k, window_s=args.window, smoke=bool(args.smoke_frames))
    def save():
        (output / 'status.json').write_text(json.dumps(status, indent=2))
    save()
    if not args.smoke_frames:
        assert json.loads((ROOT / 'dual_parallel_parity.json').read_text())['passed']
    manifest = json.loads((ROOT / 'dual_cache_manifest.json').read_text())
    assert manifest['status'] == 'complete'
    provenance = json.loads((ROOT / 'dual_image_provenance.json').read_text())
    assert sha256(ROOT / 'dual_image_provenance.json') == manifest['provenance_sha256']
    assert manifest['checkpoint_sha256'] == sha256(CHECKPOINT)
    assert manifest['need_temporal_inputs'] is True and manifest['valid_previous_camera_events'] == (6019 - 150) * 6
    legacy, NuScenes, config_factory, NuScenesEval, splits = load_dependencies(BASE / 'src/legacy_v31')
    nusc = NuScenes(version='v1.0-trainval', dataroot=str(DATA_ROOT), verbose=False)
    frames = ordered_split_frames(nusc, sorted(splits()['val']))
    assert len(frames) == 6019
    tokens = {token for token, _ in frames}
    caches = {}
    # The old current-only exporter stores per-camera caches in the six camera dirs.
    for camera in CAMERAS:
        path = Path(manifest['cameras'][camera]['path'])
        assert sha256(path) == manifest['cameras'][camera]['result_sha256']
        caches[camera] = legacy.read_cache(path)
        assert set(caches[camera]) == tokens
    if args.smoke_frames:
        frames = frames[:args.smoke_frames]
    tokens = {token for token, _ in frames}
    selections = oldest_first_selections(frames, camera_count=args.k)
    # Native manuscript selections serialize each mask in canonical camera order.
    selections = {token: tuple(c for c in CAMERAS if c in selected) for token, selected in selections.items()}
    predictions, trace, times = {}, [], []
    expected_frame_count = len(frames)
    shard = False
    started = time.perf_counter()
    # Whole scenes are independent because all state resets at scene boundaries.
    # Fork after loading metadata/caches so workers share their read-only pages.
    if (not args.smoke_frames or args.parallel_smoke) and args.workers > 1:
        scenes = list(dict.fromkeys(scene for _, scene in frames))
        worker_count = min(args.workers, len(scenes))
        scene_rank = {scene: index % worker_count for index, scene in enumerate(scenes)}
        children = []
        for rank in range(worker_count):
            child = os.fork()
            if child == 0:
                shard = True
                frames = [(token, scene) for token, scene in frames if scene_rank[scene] == rank]
                expected_frame_count = len(frames)
                output = output / f'shard_{rank}'
                output.mkdir()
                break
            children.append(child)
        if not shard:
            status.update(status='replaying_scenes', worker_pids=children, workers=worker_count)
            save()
            for child in children:
                _, result = os.waitpid(child, 0)
                assert os.waitstatus_to_exitcode(result) == 0, f'Scene worker {child} failed'
            for rank in range(worker_count):
                directory = output / f'shard_{rank}'
                with gzip.open(directory / 'predictions.json.gz', 'rt') as source:
                    partial = json.load(source)
                assert not (set(partial) & set(predictions)), 'Repeated scene/frame in workers'
                predictions.update(partial)
                trace.extend(json.loads((directory / 'ancestry_trace.json').read_text()))
                times.extend(json.loads((directory / 'frame_times.json').read_text()))
            assert set(predictions) == tokens and len(times) == expected_frame_count
            frames = []
    history, active_scene = [], None
    status.update(status='replaying', expected_frames=expected_frame_count, checkpoint_sha256=manifest['checkpoint_sha256'])
    save()
    for index, (token, scene) in enumerate(frames):
        if scene != active_scene:
            history, active_scene = [], scene
        sample_timestamp = int(nusc.get('sample', token)['timestamp'])
        captures = camera_capture_timestamps_us(nusc, token, CAMERAS)
        selected = selections[token]
        # Only selected current-token detections enter a mechanism.
        current = {camera: caches[camera][token] for camera in selected}
        beginning = time.perf_counter()
        common_history = []
        past_images, max_age = 0, 0.0
        for past_timestamp, past_captures, observations, past_token in history:
            allowed = {}
            for camera, records in observations.items():
                age = (captures[camera] - past_captures[camera]) / 1e6
                if 0 < age <= args.window:
                    allowed[camera] = records
                    past_images += 1
                    max_age = max(max_age, age)
            if allowed:
                common_history.append((past_timestamp, allowed, past_token))
        assert max_age <= args.window and past_images <= args.k * (4 if args.window == 1.5 else 2)
        latest = {}
        for past_timestamp, observations, past_token in common_history:
            for camera, records in observations.items():
                latest[camera] = (past_timestamp, records, past_token)
        available_raw = raw_context(provenance, token, [(old, list(obs)) for _, obs, old in common_history] + [(token, selected)])
        used_observations = [(token, selected)]
        current_merged = merge_sources(legacy, token, list(current.items()), .05, 500)
        if args.method in ('none', 'static', 'motion'):
            sources = list(current.items())
            if args.method != 'none':
                for camera in CAMERAS:
                    if camera in current or camera not in latest:
                        continue
                    past_timestamp, records, past_token = latest[camera]
                    used_observations.append((past_token, [camera]))
                    age = (sample_timestamp - past_timestamp) / 1e6
                    propagated = []
                    for record in records:
                        # Three non-tracking classes use the same static fallback in all reuse methods.
                        use_velocity = args.method == 'motion' and record['detection_name'] in TRACKING_CLASSES
                        propagated.append(propagate_detection(record, age_s=age, tau_s=1e9, use_velocity=use_velocity))
                    sources.append((camera, propagated))
            result = merge_sources(legacy, token, sources, .05, 500)
        else:
            used_observations += [(old, list(obs)) for _, obs, old in common_history]
            observations = []
            for past_timestamp, per_camera, _ in common_history:
                merged = merge_sources(legacy, token, list(per_camera.items()), .05, 500)
                observations.append((past_timestamp / 1e6, merged))
            observations.append((sample_timestamp / 1e6, current_merged))
            function = ab_predict if args.method == 'ab3dmot' else simple_predict
            result = function(observations)
            fallback = [(camera, [r for r in records if r['detection_name'] not in TRACKING_CLASSES])
                        for camera, records in current.items()]
            for camera, (_, records, _) in latest.items():
                if camera not in current:
                    fallback.append((camera, [r for r in records if r['detection_name'] not in TRACKING_CLASSES]))
            result += merge_sources(legacy, token, fallback, .05, 500)
            result = merge_sources(legacy, token, [(CAMERAS[0], result)], .05, 500)
        for record in result:
            record['sample_token'] = token
            record['attribute_name'] = common_attribute(record)
            assert all(float(value) > 0 for value in record['size'])
            assert 0 <= record['detection_score'] <= 1
        times.append(time.perf_counter() - beginning)
        predictions[token] = result
        trace.append(dict(sample_token=token, selected=list(selected), current_images=args.k,
                          available_raw=available_raw, used_raw=raw_context(provenance, token, used_observations),
                          selected_previous_images=sum(bool(provenance[token][c]['previous_sd']) for c in selected),
                          available_past_images=past_images, max_available_past_image_age_s=max_age,
                          state_rebuilt_from_bounded_window=args.method in ('ab3dmot', 'simpletrack')))
        history.append((sample_timestamp, captures, current, token))
        # Four recent keyframes suffice for the frozen <=1.5s context; eligibility is checked by raw times.
        history = history[-4:]
        if (index + 1) % 100 == 0 or index == 0:
            status.update(completed_frames=index + 1, elapsed_seconds=time.perf_counter() - started,
                          mean_replay_ms=1000 * statistics.fmean(times))
            save()
            print(json.dumps(status), flush=True)
    status.update(status='scoring' if not args.smoke_frames else 'smoke_complete',
                  completed_frames=len(predictions), elapsed_replay_seconds=time.perf_counter() - started,
                  replay_ms_mean=1000*statistics.fmean(times), replay_ms_p95=1000*sorted(times)[int(.95*(len(times)-1))],
                  max_available_past_images=max(t['available_past_images'] for t in trace),
                  max_available_past_image_age_s=max(t['max_available_past_image_age_s'] for t in trace),
                  unselected_current_token_reads=0, history_window_violations=0,
                  inference_timing_scope='CPU cache replay and bounded-state rebuild; excludes detector and image I/O',
                  official_history_compliant=False,
                  max_available_raw_image_age_s=max(t['available_raw']['max_raw_image_age_s'] for t in trace),
                  max_used_raw_image_age_s=max(t['used_raw']['max_raw_image_age_s'] for t in trace),
                  max_available_past_raw_images=max(t['available_raw']['past_raw_images'] for t in trace),
                  max_used_past_raw_images=max(t['used_raw']['past_raw_images'] for t in trace),
                  scope='Adapted mechanism comparison; 7 tracker classes plus shared static fallback for 3 other classes',
                  detector_parameters=7191090, added_learned_parameters=0,
                  detector_previous_branch_disabled=False,
                  tracking_birth_survival_adaptation='immediate birth; survive within bounded available context',
                  common_attribute_rule=True, confidence='AB3DMOT and SimpleTrack retain their native track-score behavior')
    (output / 'ancestry_trace.json').write_text(json.dumps(trace))
    (output / 'source_manifest.json').write_text(json.dumps(dict(cache=manifest,
        replay_script_sha256=sha256(Path(__file__)), adapters_sha256=sha256(Path(__file__).with_name('mechanism_adapters.py')),
        dual_common_sha256=sha256(Path(__file__).with_name('dual_common.py')),
        provenance_sha256=sha256(ROOT / 'dual_image_provenance.json'),
        official_commits={repo:subprocess.check_output(['git','-C',str(EXTERNAL/repo),'rev-parse','HEAD'],text=True).strip()
                          for repo in ('AB3DMOT','SimpleTrack')}), indent=2))
    save()
    if shard:
        with gzip.open(output / 'predictions.json.gz', 'wt', compresslevel=1) as target:
            json.dump(predictions, target, separators=(',', ':'))
        (output / 'frame_times.json').write_text(json.dumps(times))
        status['status'] = 'shard_complete'
        save()
    elif args.smoke_frames:
        (output / 'predictions.json').write_text(json.dumps(predictions))
    else:
        assert set(predictions) == tokens
        status['metrics'] = official_metrics(method=name, predictions=predictions, nusc=nusc,
            config_factory=config_factory, NuScenesEval=NuScenesEval, output_dir=output)
        # Keep formal prediction evidence while reducing disk usage on this new directory only.
        path = output / 'evaluations' / name / 'results_nusc.json'
        original_hash = sha256(path)
        with path.open('rb') as source, gzip.open(str(path)+'.gz', 'wb', compresslevel=1) as target:
            import shutil
            shutil.copyfileobj(source, target)
        with gzip.open(str(path)+'.gz', 'rb') as check:
            compressed_hash = hashlib.sha256()
            for block in iter(lambda: check.read(1024*1024), b''):
                compressed_hash.update(block)
        assert compressed_hash.hexdigest() == original_hash
        path.unlink()
        # These files were created by this isolated run; verified final predictions are retained.
        for shard_prediction in output.glob('shard_*/predictions.json.gz'):
            shard_prediction.unlink()
        status.update(status='complete', elapsed_total_seconds=time.perf_counter()-started)
        save()
    print(json.dumps(status, indent=2), flush=True)


if __name__ == '__main__':
    main()
