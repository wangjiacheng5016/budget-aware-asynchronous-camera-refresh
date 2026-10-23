"""Check serial versus scene-parallel replay before full official evaluation."""
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/paper_comparison'))
from review_paths import BASE, ROOT

def main():
    if os.name != 'posix':
        raise RuntimeError('Scene-parallel replay uses POSIX fork; run this check on Linux')
    script = BASE / 'src/paper_comparison/dual_mechanism_replay.py'
    common = [sys.executable, str(script), '--method', 'motion', '--k', '3', '--smoke-frames', '80']
    subprocess.run([*common, '--workers', '1'], check=True)
    subprocess.run([*common, '--workers', '2', '--parallel-smoke'], check=True)
    parent = ROOT / 'mechanism_results'
    serial = parent / 'dual_motion_k3_h1.5_smoke_v3'
    parallel = parent / 'dual_motion_k3_h1.5_smoke_v3_parallel_check'
    a = json.loads((serial / 'predictions.json').read_text())
    b = json.loads((parallel / 'predictions.json').read_text())
    def trace(directory):
        return {r['sample_token']: r for r in json.loads((directory / 'ancestry_trace.json').read_text())}
    assert len(a) == 80 and a == b, 'Serial and parallel predictions differ'
    assert trace(serial) == trace(parallel), 'Raw-image ancestry differs'
    result = dict(passed=True, frames=80, method='motion', camera_count=3,
                  check='Exact serial/scene-parallel predictions and ancestry; no timing comparison')
    (ROOT / 'dual_parallel_parity.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))

if __name__ == '__main__':
    main()
