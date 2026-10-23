"""Serial FP32/batch-one native dual-frame export; no metric-based selection."""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/paper_comparison'))
from review_paths import BASE, CACHE, CHECKPOINT, DATA_ROOT
from dual_common import CAMERAS, digest

EXPECTED_CHECKPOINT = 'c50f8a1e3c35d9e211e666e1bfa2edd434c9d86e683ad3d0f527f8dfdb1d9043'

def main():
    if digest(CHECKPOINT) != EXPECTED_CHECKPOINT:
        raise RuntimeError('The checkpoint does not match the frozen paper checkpoint')
    training = BASE / 'src/legacy_v31/train_official_nuscenes_nds_kd_v2.py'
    CACHE.mkdir(parents=True, exist_ok=True)
    manifest = dict(version='v1.0-trainval', split='val', expected_samples=6019,
                    batch_size=1, checkpoint_sha256=digest(CHECKPOINT),
                    training_source_sha256=digest(training), cameras={})
    args = ['eval', '--model', 'student', '--data-root', str(DATA_ROOT),
        '--version', 'v1.0-trainval', '--ckpt', str(CHECKPOINT), '--eval-set', 'val',
        '--batch-size', '1', '--num-workers', '4', '--teacher-img-w', '640',
        '--teacher-img-h', '360', '--student-img-w', '640', '--student-img-h', '360',
        '--target-scope', 'student_fov', '--student-eval-views', 'all', '--student-fov-only',
        '--temporal-delta-limit', '16', '--score-thr', '0.05', '--topk', '500',
        '--nms-iou', '0.2', '--merge-nms-iou', '0.05', '--max-boxes-per-sample', '500',
        '--depth-bins', '32', '--student-backbone', 'multiscale', '--bev-norm', 'group',
        '--attribute-head', '--temporal-bev', '--temporal-pre-encoder',
        '--velocity-displacement-target', '--class-aware-motion-head',
        '--class-aware-size-head', '--metric-scope', 'official_360', '--skip-eval']
    for camera in CAMERAS:
        output = CACHE / camera
        # Refuse partial or stale caches rather than silently mixing exports.
        output.mkdir(exist_ok=False)
        with (output / 'export.log').open('w') as log:
            subprocess.run([sys.executable, str(training), *args,
                '--student-eval-cameras', camera, '--out-dir', str(output)],
                stdout=log, stderr=subprocess.STDOUT, check=True)
        result = output / 'results_nusc.json'
        protocol = json.loads((output / 'evaluation_protocol.json').read_text())
        count = len(json.loads(result.read_text())['results'])
        assert count == protocol['evaluated_sample_count'] == 6019
        assert protocol['timing']['batch_size'] == 1 and protocol['parameters'] == 7191090
        manifest['cameras'][camera] = dict(samples=count, bytes=result.stat().st_size,
            sha256=digest(result), export_seconds=protocol['timing']['export_total_seconds'])
        print('Verified', camera, count, flush=True)
    (CACHE / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

if __name__ == '__main__':
    main()
