"""
Build script for GitHub Pages: copy generated dashboards into docs/dashboards/.

Usage (from python/):
    python -m utils.build_gh_pages

Copies:
    output/predictions/baseline_dashboard.html -> docs/dashboards/baseline.html

Extend DASHBOARDS mapping to add more outputs.
"""

import shutil
import sys
from pathlib import Path

# Source paths relative to python/ (cwd when run via python -m)
# Target paths relative to repo root
DASHBOARDS = [
    ("output/predictions/baseline_dashboard.html", "docs/dashboards/baseline.html"),
]


def main() -> int:
    # Resolve paths: support running from python/ or repo root
    cwd = Path.cwd()
    if (cwd / "output").is_dir():
        py_dir = cwd
        repo_root = cwd.parent
    elif (cwd / "python" / "output").is_dir():
        py_dir = cwd / "python"
        repo_root = cwd
    else:
        py_dir = cwd
        repo_root = cwd

    dst_root = repo_root / "docs" / "dashboards"
    dst_root.mkdir(parents=True, exist_ok=True)

    ok = 0
    for src_rel, dst_rel in DASHBOARDS:
        src = py_dir / src_rel
        dst = repo_root / dst_rel
        if not src.exists():
            print(f"[skip] {src} not found", file=sys.stderr)
            continue
        shutil.copy2(src, dst)
        print(f"[OK] {src.name} -> {dst.relative_to(repo_root)}")
        ok += 1

    if ok == 0:
        print("No dashboards copied. Run baseline_results_dashboard first.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
