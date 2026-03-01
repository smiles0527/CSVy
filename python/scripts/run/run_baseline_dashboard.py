#!/usr/bin/env python3
"""Generate baseline Elo results dashboard (goals vs xG) from pipeline outputs."""

import os
import sys
import pathlib

_script = pathlib.Path(__file__).resolve()
_python_dir = _script.parent
while True:
    if (_python_dir / 'utils').is_dir():
        break
    parent = _python_dir.parent
    if parent == _python_dir:
        raise RuntimeError('Cannot locate python/')
    _python_dir = parent
os.chdir(_python_dir)
sys.path.insert(0, str(_python_dir))

from utils.baseline_results_dashboard import run_dashboard

if __name__ == "__main__":
    out = pathlib.Path("output/predictions/baseline_dashboard.html")
    sys.exit(run_dashboard(str(out), also_xg=True))
