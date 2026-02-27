#!/usr/bin/env python3
"""Generate baseline Elo results dashboard (goals vs xG) from pipeline outputs."""

import os
import sys
import pathlib

_cwd = pathlib.Path(os.path.abspath("")).resolve()
if (_cwd / "python").is_dir():
    _python_dir = _cwd / "python"
elif _cwd.name == "python":
    _python_dir = _cwd
elif (_cwd / "data").is_dir():
    _python_dir = _cwd
else:
    _python_dir = _cwd

os.chdir(_python_dir)
sys.path.insert(0, str(_python_dir))

from utils.baseline_results_dashboard import run_dashboard

if __name__ == "__main__":
    out = pathlib.Path("output/predictions/baseline_dashboard.html")
    sys.exit(run_dashboard(str(out)))
