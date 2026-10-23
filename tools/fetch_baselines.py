"""Fetch public upstream code at the exact commits used by the adapters."""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/paper_comparison'))
from review_paths import EXTERNAL

REPOS = {
    'AB3DMOT': ('https://github.com/xinshuoweng/AB3DMOT.git', '61f3bd72574093e367916c757b4747ca445f978c'),
    'SimpleTrack': ('https://github.com/tusen-ai/SimpleTrack.git', '05c96bb7ed98fc179856f327544612a66c839b5e'),
}

def main():
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    for name, (url, commit) in REPOS.items():
        target = EXTERNAL / name
        if not target.exists():
            subprocess.run(['git', 'clone', '--no-checkout', url, str(target)], check=True)
            subprocess.run(['git', '-C', str(target), 'checkout', '--detach', commit], check=True)
        actual = subprocess.check_output(['git', '-C', str(target), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != commit:
            raise RuntimeError(f'{target}: expected {commit}, found {actual}; existing checkout was not changed')
        if subprocess.check_output(['git', '-C', str(target), 'status', '--porcelain'], text=True).strip():
            raise RuntimeError(f'{target}: upstream checkout contains local modifications')
        print(name, actual)

if __name__ == '__main__':
    main()
