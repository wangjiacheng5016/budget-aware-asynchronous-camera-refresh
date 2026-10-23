"""Run the eleven predeclared dual-frame comparison conditions on Linux."""
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]

def main():
    script = BASE / 'src/paper_comparison/dual_mechanism_replay.py'
    conditions = [(k, method) for k in (3, 4)
                  for method in ('none', 'static', 'motion', 'ab3dmot', 'simpletrack')]
    conditions.append((6, 'none'))
    for k, method in conditions:
        subprocess.run([sys.executable, str(script), '--method', method, '--k', str(k)], check=True)

if __name__ == '__main__':
    main()
