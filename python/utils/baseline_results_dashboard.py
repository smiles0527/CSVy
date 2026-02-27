"""
Baseline Results Dashboard - Post-run visualization of baseline_elo and baseline_elo_xg outputs.

Loads outputs from baseline_elo (goals) and baseline_elo_xg (xG) pipelines and displays:
- Goals vs xG k-metrics comparison
- Round 1 predictions side-by-side
- Validation summaries

Usage:
    from utils.baseline_results_dashboard import BaselineResultsDashboard
    dash = BaselineResultsDashboard()
    dash.load()
    dash.save_html('output/predictions/baseline_dashboard.html')

    # Or from CLI:
    python -m utils.baseline_results_dashboard
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# QoL: Copy buttons, download, section nav
QOL_CSS = """
.data-toolbar { display: flex; gap: 0.5rem; align-items: center; margin: 0.5em 0; flex-wrap: wrap; }
.data-toolbar button, .data-toolbar a.btn { padding: 0.35em 0.7em; font-size: 0.85em; cursor: pointer; border: 1px solid #888; border-radius: 4px; background: #f8f8f8; }
.data-toolbar button:hover, .data-toolbar a.btn:hover { background: #eee; }
.table-wrap { position: relative; margin: 0.5em 0; }
.table-wrap .copy-btn { position: absolute; top: 4px; right: 4px; padding: 0.25em 0.5em; font-size: 0.8em; cursor: pointer; background: #e8e8e8; border: 1px solid #ccc; border-radius: 3px; }
.table-wrap .copy-btn:hover { background: #ddd; }
.table-wrap .copy-btn.copied { background: #8f8; }
.section-nav { position: sticky; top: 0; background: #fff; padding: 0.5em 0; border-bottom: 1px solid #ddd; margin-bottom: 1em; z-index: 10; }
.section-nav a { margin-right: 1em; font-size: 0.9em; color: #06c; }
.section-nav a:hover { text-decoration: underline; }
#toast { position: fixed; bottom: 1em; right: 1em; padding: 0.5em 1em; background: #333; color: #fff; border-radius: 4px; font-size: 0.9em; z-index: 1000; display: none; }
.table-scroll { max-height: 70vh; overflow-y: auto; margin: 0.5em 0; }
"""

QOL_JS = """
function copyTableTSV(btn) {
  var tbl = btn.closest('.table-wrap').querySelector('table');
  var rows = tbl.querySelectorAll('tr');
  var lines = [];
  for (var i = 0; i < rows.length; i++) {
    var cells = rows[i].querySelectorAll('th, td');
    lines.push(Array.from(cells).map(function(c){ return c.textContent.trim(); }).join('\\t'));
  }
  var tsv = lines.join('\\n');
  navigator.clipboard.writeText(tsv).then(function(){
    btn.textContent = 'Copied!';
    btn.classList.add('copied');
    setTimeout(function(){ btn.textContent = 'Copy TSV'; btn.classList.remove('copied'); }, 1500);
  });
}

function showToast(msg) {
  var el = document.getElementById('toast');
  el.textContent = msg;
  el.style.display = 'block';
  setTimeout(function(){ el.style.display = 'none'; }, 2000);
}

function downloadCSV(data, filename) {
  var blob = new Blob([data], {type: 'text/csv;charset=utf-8'});
  var a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}

document.addEventListener('DOMContentLoaded', function(){
  document.querySelectorAll('.copy-btn').forEach(function(btn){
    btn.onclick = function(){ copyTableTSV(btn); };
  });
  var kData = document.getElementById('k-metrics-data');
  if (kData) {
    var data = JSON.parse(kData.textContent);
    Object.keys(data).forEach(function(key){
      var btn = document.getElementById('dl-' + key);
      if (btn) {
        btn.onclick = function(){
          var df = data[key];
          if (!df || df.length === 0) return;
          var cols = Object.keys(df[0]);
          var csv = cols.join(',') + '\\n' + df.map(function(r){
            return cols.map(function(c){
              var v = r[c];
              if (v == null) return '';
              return typeof v === 'number' ? String(v) : '"' + String(v).replace(/"/g,'""') + '"';
            }).join(',');
          }).join('\\n');
          downloadCSV(csv, key + '.csv');
          showToast('Downloaded ' + key + '.csv');
        };
      }
    });
  }
});
"""

# Visualization imports
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class BaselineResultsDashboard:
    """
    Post-run dashboard that reads baseline_elo and baseline_elo_xg outputs
    and creates interactive comparison visualizations.
    """

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        goals_sweep_path: Optional[Path] = None,
        xg_sweep_path: Optional[Path] = None,
        goals_validation_path: Optional[Path] = None,
        xg_validation_path: Optional[Path] = None,
    ):
        self.base_dir = Path(base_dir or "output/predictions")
        self.goals_sweep = Path(goals_sweep_path or self.base_dir / "baseline_elo" / "sweep")
        self.xg_sweep = Path(xg_sweep_path or self.base_dir / "baseline_elo_xg" / "sweep")
        self.goals_validation = Path(goals_validation_path or self.base_dir / "baseline_elo" / "validation")
        self.xg_validation = Path(xg_validation_path or self.base_dir / "baseline_elo_xg" / "validation")

        self.goals_k_metrics: Optional[pd.DataFrame] = None
        self.goals_k_metrics_full: Optional[pd.DataFrame] = None  # All iterations 1.0, 1.1, 2.0
        self.xg_k_metrics: Optional[pd.DataFrame] = None
        self.goals_r1: Optional[pd.DataFrame] = None
        self.xg_r1: Optional[pd.DataFrame] = None
        self.goals_val_comparison: Optional[pd.DataFrame] = None
        self.xg_val_comparison: Optional[pd.DataFrame] = None
        self.goals_val_summary: Optional[str] = None
        self.xg_val_summary: Optional[str] = None
        self.goals_sweep_comparison: Optional[pd.DataFrame] = None
        self.xg_sweep_comparison: Optional[pd.DataFrame] = None
        self.goals_summary: Optional[Dict] = None
        self.xg_summary: Optional[Dict] = None
        self.goals_baselines: Optional[pd.DataFrame] = None
        self.xg_baselines: Optional[pd.DataFrame] = None
        self.goals_significance: Optional[pd.DataFrame] = None
        self.xg_significance: Optional[pd.DataFrame] = None
        self.goals_calibration: Optional[pd.DataFrame] = None
        self.xg_calibration: Optional[pd.DataFrame] = None
        self.goals_elo_vs_standings: Optional[pd.DataFrame] = None
        self.xg_elo_vs_standings: Optional[pd.DataFrame] = None

    def load(self) -> bool:
        """Load all available output files. Returns True if at least one k_metrics was loaded."""
        loaded = False

        # Goals: from baseline_elo sweep, use iteration 1.0 (goals)
        k_csv = self.goals_sweep / "k_metrics.csv"
        if k_csv.exists():
            df = pd.read_csv(k_csv)
            if "model_iteration" in df.columns:
                self.goals_k_metrics = df[df["model_iteration"] == "1.0"].copy()
                self.goals_k_metrics_full = df.copy()
            else:
                self.goals_k_metrics = df.copy()
                self.goals_k_metrics_full = df.copy()
            loaded = len(self.goals_k_metrics) > 0
        else:
            alt = self.base_dir / "baseline_elo" / "goals" / "k_metrics.csv"
            if alt.exists():
                self.goals_k_metrics = pd.read_csv(alt)
                if "k_factor" in self.goals_k_metrics.columns and "k" not in self.goals_k_metrics.columns:
                    self.goals_k_metrics = self.goals_k_metrics.rename(columns={"k_factor": "k"})
                self.goals_k_metrics_full = self.goals_k_metrics.copy()
                loaded = True

        # xG sweep k_metrics
        k_csv_xg = self.xg_sweep / "k_metrics.csv"
        if k_csv_xg.exists():
            self.xg_k_metrics = pd.read_csv(k_csv_xg)
            loaded = loaded or len(self.xg_k_metrics) > 0
        else:
            alt_xg = self.base_dir / "baseline_elo_xg" / "xg" / "k_metrics.csv"
            if alt_xg.exists():
                self.xg_k_metrics = pd.read_csv(alt_xg)
                if self.xg_k_metrics is not None and "k_factor" in self.xg_k_metrics.columns:
                    self.xg_k_metrics = self.xg_k_metrics.rename(columns={"k_factor": "k"})
                loaded = True

        # Round 1 predictions
        for name, path in [
            ("goals_r1", self.goals_sweep / "round1_predictions.csv"),
            ("goals_r1_alt", self.base_dir / "baseline_elo" / "goals" / "round1_predictions.csv"),
        ]:
            if path.exists():
                self.goals_r1 = pd.read_csv(path)
                break

        for name, path in [
            ("xg_r1", self.xg_sweep / "round1_predictions.csv"),
            ("xg_r1_alt", self.base_dir / "baseline_elo_xg" / "xg" / "round1_predictions.csv"),
        ]:
            if path.exists():
                self.xg_r1 = pd.read_csv(path)
                break

        # Validation comparison
        for path in [self.goals_validation / "comparison.csv", self.goals_validation / "summary.md"]:
            if path.exists():
                if path.suffix == ".csv":
                    self.goals_val_comparison = pd.read_csv(path)
                else:
                    self.goals_val_summary = path.read_text(encoding="utf-8")
                break

        for path in [self.xg_validation / "comparison.csv", self.xg_validation / "summary.md"]:
            if path.exists():
                if path.suffix == ".csv":
                    self.xg_val_comparison = pd.read_csv(path)
                else:
                    self.xg_val_summary = path.read_text(encoding="utf-8")
                break

        if self.goals_val_summary is None and (self.goals_validation / "summary.md").exists():
            self.goals_val_summary = (self.goals_validation / "summary.md").read_text(encoding="utf-8")
        if self.xg_val_summary is None and (self.xg_validation / "summary.md").exists():
            self.xg_val_summary = (self.xg_validation / "summary.md").read_text(encoding="utf-8")

        # Sweep comparison (full k-sweep for hyperparam comparison)
        for p in [self.goals_sweep / "comparison.csv", self.base_dir / "baseline_elo" / "goals" / "comparison.csv"]:
            if p.exists():
                self.goals_sweep_comparison = pd.read_csv(p)
                break
        for p in [self.xg_sweep / "comparison.csv", self.base_dir / "baseline_elo_xg" / "xg" / "comparison.csv"]:
            if p.exists():
                self.xg_sweep_comparison = pd.read_csv(p)
                break

        # Pipeline summary (best_params, team_rankings)
        for p in [self.goals_sweep / "pipeline_summary.json", self.base_dir / "baseline_elo" / "goals" / "pipeline_summary.json"]:
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    self.goals_summary = json.load(f)
                break
        for p in [self.xg_sweep / "pipeline_summary.json", self.base_dir / "baseline_elo_xg" / "xg" / "pipeline_summary.json"]:
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    self.xg_summary = json.load(f)
                break

        # Validation files
        if (self.goals_validation / "baselines.csv").exists():
            self.goals_baselines = pd.read_csv(self.goals_validation / "baselines.csv")
        if (self.xg_validation / "baselines.csv").exists():
            self.xg_baselines = pd.read_csv(self.xg_validation / "baselines.csv")
        if (self.goals_validation / "significance.csv").exists():
            self.goals_significance = pd.read_csv(self.goals_validation / "significance.csv")
        if (self.xg_validation / "significance.csv").exists():
            self.xg_significance = pd.read_csv(self.xg_validation / "significance.csv")
        if (self.goals_validation / "calibration_stats.csv").exists():
            self.goals_calibration = pd.read_csv(self.goals_validation / "calibration_stats.csv")
        if (self.xg_validation / "calibration_stats.csv").exists():
            self.xg_calibration = pd.read_csv(self.xg_validation / "calibration_stats.csv")
        if (self.goals_validation / "elo_vs_standings.csv").exists():
            self.goals_elo_vs_standings = pd.read_csv(self.goals_validation / "elo_vs_standings.csv")
        if (self.xg_validation / "elo_vs_standings.csv").exists():
            self.xg_elo_vs_standings = pd.read_csv(self.xg_validation / "elo_vs_standings.csv")

        return loaded

    def _ensure_k_column(self, df: pd.DataFrame) -> pd.DataFrame:
        if "k" not in df.columns and "k_factor" in df.columns:
            return df.rename(columns={"k_factor": "k"})
        return df

    def _wrap_table(self, html: str) -> str:
        """Wrap table in div with Copy TSV button for Vernier/Excel paste."""
        if "<table" not in html:
            return html
        return f'<div class="table-wrap"><button class="copy-btn">Copy TSV</button>{html}</div>'

    def _df_to_records(self, df: pd.DataFrame) -> List[Dict]:
        """Convert DataFrame to list of dicts for JSON embedding (handles NaN)."""
        return json.loads(df.to_json(orient="records", date_format="iso", default_handler=str))

    def _section_nav(self) -> str:
        """Sticky nav with jump links for quick navigation."""
        return """<nav class="section-nav">
<a href="#overview">Overview</a>
<a href="#hyperparams">Hyperparameters</a>
<a href="#k-metrics">K vs Metrics</a>
<a href="#validation">Validation</a>
<a href="#calibration">Calibration</a>
<a href="#rankings">Rankings</a>
<a href="#round1">Round 1</a>
<span style="color:#666;font-size:0.85em;margin-left:1em">Tip: Copy TSV → paste in Vernier/Excel</span>
</nav>"""

    def _build_k_metrics_json(self, xg_only: bool = False, iteration: Optional[str] = None) -> Dict[str, List[Dict]]:
        """Build k_metrics data for CSV download (Vernier-friendly)."""
        out: Dict[str, List[Dict]] = {}
        goals_df = None if xg_only else (self.goals_k_metrics_full if self.goals_k_metrics_full is not None else self.goals_k_metrics)
        if goals_df is not None and len(goals_df) > 0:
            if iteration and "model_iteration" in goals_df.columns:
                goals_df = goals_df[goals_df["model_iteration"].astype(str) == str(iteration)]
            if len(goals_df) > 0:
                gdf = self._ensure_k_column(goals_df).sort_values("k" if "k" in goals_df.columns else "k_factor")
                out["k_metrics_goals"] = self._df_to_records(gdf)
        if self.xg_k_metrics is not None and len(self.xg_k_metrics) > 0 and not iteration:
            xdf = self._ensure_k_column(self.xg_k_metrics).sort_values("k" if "k" in self.xg_k_metrics.columns else "k_factor")
            out["k_metrics_xg"] = self._df_to_records(xdf)
        return out

    def _k_metrics_download_section(self, k_metrics_data: Dict[str, List[Dict]]) -> str:
        """Download buttons for k_metrics CSV (Vernier, Excel)."""
        parts = ['<div class="data-toolbar">']
        for key in k_metrics_data:
            if k_metrics_data[key]:
                safe_key = key.replace(" ", "_")
                parts.append(f'<button class="btn" id="dl-{safe_key}">Download {key}.csv</button>')
        parts.append("</div>")
        return "".join(parts)

    def create_figure(self, xg_only: bool = False, iteration: Optional[str] = None) -> Optional[Any]:
        """Create Plotly figure with goals vs xG comparison (all iterations 1.0, 1.1, 2.0).
        If iteration is '1.0', '1.1', or '2.0', show only that iteration's data."""
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available for BaselineResultsDashboard")
            return None

        goals_df = None if xg_only else (self.goals_k_metrics_full if self.goals_k_metrics_full is not None else self.goals_k_metrics)
        if goals_df is not None and iteration and "model_iteration" in goals_df.columns:
            goals_df = goals_df[goals_df["model_iteration"].astype(str) == str(iteration)].copy()
        if goals_df is not None and len(goals_df) == 0:
            goals_df = None
        has_goals = goals_df is not None and len(goals_df) > 0
        has_xg = self.xg_k_metrics is not None and len(self.xg_k_metrics) > 0
        if not has_goals and not has_xg:
            logger.warning("No k_metrics data loaded")
            return None

        metrics = ["accuracy", "brier_loss", "log_loss", "combined_rmse"]
        available = []
        for m in metrics:
            if has_goals and m in goals_df.columns:
                available.append(m)
            elif has_xg and m in self.xg_k_metrics.columns:
                available.append(m)
        if not available:
            df = goals_df if goals_df is not None else self.xg_k_metrics
            if df is not None:
                available = [c for c in df.columns if c not in ("k", "k_factor", "model_iteration")]

        n_metrics = len(available)
        n_cols = min(2, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=available,
            horizontal_spacing=0.12,
            vertical_spacing=0.15,
        )

        colors = {"1.0": "#1f77b4", "1.1": "#ff7f0e", "2.0": "#2ca02c"}

        for idx, metric in enumerate(available):
            row = idx // n_cols + 1
            col = idx % n_cols + 1

            if has_goals and goals_df is not None and metric in goals_df.columns:
                if "model_iteration" in goals_df.columns:
                    for it in ["1.0", "1.1", "2.0"]:
                        sub = goals_df[goals_df["model_iteration"] == it]
                        if len(sub) == 0:
                            continue
                        gdf = self._ensure_k_column(sub)
                        k_col = "k" if "k" in gdf.columns else "k_factor"
                        gdf = gdf.sort_values(k_col)
                        label = {"1.0": "Goals (1.0)", "1.1": "xG (1.1)", "2.0": "Off/Def (2.0)"}.get(it, it)
                        fig.add_trace(
                            go.Scatter(
                                x=gdf[k_col].tolist(),
                                y=gdf[metric].tolist(),
                                name=label,
                                mode="lines+markers",
                                line=dict(color=colors.get(it, "#333")),
                            ),
                            row=row, col=col,
                        )
                else:
                    gdf = self._ensure_k_column(goals_df).sort_values("k" if "k" in goals_df.columns else "k_factor")
                    k_col = "k" if "k" in gdf.columns else "k_factor"
                    fig.add_trace(
                        go.Scatter(
                            x=gdf[k_col].tolist(),
                            y=gdf[metric].tolist(),
                            name="Goals",
                            mode="lines+markers",
                            line=dict(color="#1f77b4"),
                        ),
                        row=row, col=col,
                    )
            goals_has_iter = goals_df is not None and "model_iteration" in goals_df.columns and not iteration
            if has_xg and self.xg_k_metrics is not None and metric in self.xg_k_metrics.columns and not goals_has_iter and not iteration:
                xdf = self._ensure_k_column(self.xg_k_metrics).sort_values("k" if "k" in self.xg_k_metrics.columns else "k_factor")
                k_col = "k" if "k" in xdf.columns else "k_factor"
                fig.add_trace(
                    go.Scatter(
                        x=xdf[k_col].tolist(),
                        y=xdf[metric].tolist(),
                        name="xG",
                        mode="lines+markers",
                        line=dict(color="#ff7f0e"),
                    ),
                    row=row, col=col,
                )
            elif has_xg and self.xg_k_metrics is not None and metric in self.xg_k_metrics.columns and goals_has_iter and not iteration:
                xdf = self._ensure_k_column(self.xg_k_metrics).sort_values("k" if "k" in self.xg_k_metrics.columns else "k_factor")
                k_col = "k" if "k" in xdf.columns else "k_factor"
                fig.add_trace(
                    go.Scatter(
                        x=xdf[k_col].tolist(),
                        y=xdf[metric].tolist(),
                        name="xG-standalone",
                        mode="lines+markers",
                        line=dict(color="#d62728"),
                    ),
                    row=row, col=col,
                )

        fig.update_layout(
            title="K vs Metrics",
            height=400 * max(1, n_rows),
            showlegend=True,
        )
        fig.update_xaxes(title_text="k")
        k_min, k_max = None, None
        for df in [goals_df, self.xg_k_metrics]:
            if df is None or len(df) == 0:
                continue
            gdf = self._ensure_k_column(df)
            k_col = "k" if "k" in gdf.columns else "k_factor"
            if k_col in gdf.columns:
                km = float(gdf[k_col].min())
                kx = float(gdf[k_col].max())
                k_min = km if k_min is None else min(k_min, km)
                k_max = kx if k_max is None else max(k_max, kx)
        for idx, metric in enumerate(available):
            row, col = idx // n_cols + 1, idx % n_cols + 1
            if k_min is not None and k_max is not None:
                pad = max(0.5, (k_max - k_min) * 0.02)
                fig.update_xaxes(range=[k_min - pad, k_max + pad], row=row, col=col)
            if metric == "accuracy":
                fig.update_yaxes(range=[0, 1.05], row=row, col=col)
            elif metric == "combined_rmse":
                fig.update_yaxes(rangemode="tozero", row=row, col=col)
        return fig

    def create_calibration_figure(self, xg_only: bool = False, iteration: Optional[str] = None) -> Optional[Any]:
        """Create Plotly reliability diagram from calibration_stats."""
        if not PLOTLY_AVAILABLE:
            return None
        cal_list = []
        if xg_only:
            if self.xg_calibration is not None and len(self.xg_calibration) > 0:
                cal_list.append(("xG", self.xg_calibration))
        elif iteration:
            if self.goals_calibration is not None and len(self.goals_calibration) > 0:
                lab = {"1.0": "Goals", "1.1": "xG", "2.0": "Off/Def"}.get(iteration, iteration)
                cal_list.append((lab, self.goals_calibration))
        else:
            if self.goals_calibration is not None and len(self.goals_calibration) > 0:
                cal_list.append(("Goals", self.goals_calibration))
            if self.xg_calibration is not None and len(self.xg_calibration) > 0:
                cal_list.append(("xG", self.xg_calibration))
        if not cal_list:
            return None

        n = len(cal_list)
        fig = make_subplots(rows=1, cols=n, subplot_titles=[c[0] for c in cal_list])
        colors = ["#1f77b4", "#ff7f0e"]
        for idx, (label, cal_df) in enumerate(cal_list):
            fig.add_trace(
                go.Scatter(
                    x=cal_df["pred_avg"],
                    y=cal_df["actual_rate"],
                    mode="markers",
                    name=label,
                    marker=dict(size=cal_df["count"].clip(upper=100), opacity=0.7),
                    text=[f"n={int(c)}" for c in cal_df["count"]],
                ),
                row=1, col=idx + 1,
            )
            fig.add_trace(
                go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Ideal", line=dict(dash="dash", color="gray")),
                row=1, col=idx + 1,
            )
        fig.update_layout(title="Calibration", height=400)
        for c in range(1, n + 1):
            fig.update_xaxes(title_text="Predicted prob", range=[0, 1], row=1, col=c)
            fig.update_yaxes(title_text="Actual rate", range=[0, 1], scaleanchor="x", scaleratio=1, row=1, col=c)
        return fig

    def _section_overview(self, xg_only: bool = False, iteration: Optional[str] = None) -> str:
        """Overview: best k, key metrics, output paths."""
        parts = []
        if not xg_only and self.goals_summary and (iteration is None or iteration in ("1.0", "1.1", "2.0")):
            labels = {"1.0": "Goals", "1.1": "xG", "2.0": "Off/Def"}
            lab = labels.get(iteration) if iteration else "Goals"
            if iteration and self.goals_sweep_comparison is not None and "model_iteration" in self.goals_sweep_comparison.columns:
                sub = self.goals_sweep_comparison[self.goals_sweep_comparison["model_iteration"].astype(str) == str(iteration)]
                if len(sub) > 0 and "combined_rmse" in sub.columns:
                    best_row = sub.loc[sub["combined_rmse"].idxmin()]
                    best_k = best_row.get("k", best_row.get("k_factor", "?"))
                else:
                    best_k = self.goals_summary.get("best_params", {}).get("k_factor", "?")
            else:
                best_k = self.goals_summary.get("best_params", {}).get("k_factor", self.goals_summary.get("best_k", "?"))
            parts.append(f"<p><b>{lab}</b>: best k={best_k} | {self.base_dir / 'baseline_elo'}</p>")
        if self.xg_summary and not iteration:
            bp = self.xg_summary.get("best_params", {})
            best_k = bp.get("k_factor", self.xg_summary.get("best_k", "?"))
            parts.append(f"<p><b>xG</b>: best k={best_k} | {self.base_dir / 'baseline_elo_xg'}</p>")
        if not xg_only and self.goals_val_comparison is not None and len(self.goals_val_comparison) > 0 and not iteration:
            r = self.goals_val_comparison.iloc[0]
            parts.append(f"<p>Goals best: acc={r.get('win_accuracy', 0):.1%} brier={r.get('brier_loss', 0):.4f}</p>")
        if iteration:
            lab = {"1.0": "Goals", "1.1": "xG", "2.0": "Off/Def"}.get(iteration, iteration)
            parts.append(f"<p>{lab} (iteration {iteration})</p>")
        if self.xg_val_comparison is not None and len(self.xg_val_comparison) > 0 and not iteration:
            r = self.xg_val_comparison.iloc[0]
            parts.append(f"<p>xG best: acc={r.get('win_accuracy', 0):.1%} brier={r.get('brier_loss', 0):.4f}</p>")
        return '<h2 id="overview">Overview</h2>' + ("".join(parts) if parts else "<p>No data.</p>")

    def _section_hyperparams(self, xg_only: bool = False, iteration: Optional[str] = None) -> str:
        """Hyperparameter comparison: best params table, full sweep table."""
        parts = ['<h2 id="hyperparams">Hyperparameters</h2>']
        if (not xg_only and self.goals_summary) or (self.xg_summary and not iteration):
            rows = []
            if not xg_only and self.goals_summary:
                lab = {"1.0": "Goals", "1.1": "xG", "2.0": "Off/Def"}.get(iteration) if iteration else "Goals"
                bp = self.goals_summary.get("best_params", {})
                if iteration and self.goals_sweep_comparison is not None and "model_iteration" in self.goals_sweep_comparison.columns:
                    sub = self.goals_sweep_comparison[self.goals_sweep_comparison["model_iteration"].astype(str) == str(iteration)]
                    if len(sub) > 0 and "combined_rmse" in sub.columns:
                        best_row = sub.loc[sub["combined_rmse"].idxmin()]
                        bp = {k: best_row[k] for k in ["k", "win_accuracy", "brier_loss", "combined_rmse"] if k in best_row}
                rows.append({"Pipeline": lab, **{k: v for k, v in bp.items()}})
            if self.xg_summary and not iteration:
                bp = self.xg_summary.get("best_params", {})
                rows.append({"Pipeline": "xG", **{k: v for k, v in bp.items()}})
            if rows:
                df = pd.DataFrame(rows)
                parts.append(f"<h3>Params</h3>{self._wrap_table(df.to_html(index=False))}")
        combined = []
        if not xg_only and self.goals_sweep_comparison is not None and len(self.goals_sweep_comparison) > 0:
            df = self.goals_sweep_comparison.copy()
            if iteration and "model_iteration" in df.columns:
                df = df[df["model_iteration"].astype(str) == str(iteration)]
            if len(df) > 0:
                df["pipeline"] = "goals" if not iteration else {"1.0": "Goals", "1.1": "xG", "2.0": "Off/Def"}.get(iteration, iteration)
                if "model_iteration" in df.columns:
                    df["config"] = df["model_iteration"].astype(str) + "(k=" + df["k"].astype(str) + ")"
                combined.append(df)
        if self.xg_sweep_comparison is not None and len(self.xg_sweep_comparison) > 0 and not iteration:
            df = self.xg_sweep_comparison.copy()
            df["pipeline"] = "xG"
            if "model_iteration" not in df.columns:
                df["model_iteration"] = "xG"
            combined.append(df)
        if combined:
            sweep_df = pd.concat(combined, ignore_index=True)
            sort_cols = [c for c in ["pipeline", "model_iteration", "k"] if c in sweep_df.columns]
            if sort_cols:
                sweep_df = sweep_df.sort_values(sort_cols)
            cols = [c for c in ["pipeline", "model_iteration", "k", "config", "win_accuracy", "brier_loss", "log_loss", "combined_rmse"] if c in sweep_df.columns]
            parts.append(f"<h3>K-Sweep ({len(sweep_df)} rows)</h3><div class='table-scroll'>{self._wrap_table(sweep_df[cols].to_html(index=False, max_rows=None))}</div>")
        return "".join(parts)

    def _section_validation(self, xg_only: bool = False, iteration: Optional[str] = None) -> str:
        """Validation: baselines, significance, comparison, elo vs standings."""
        parts = ['<h2 id="validation">Validation</h2>']
        if not xg_only and self.goals_baselines is not None:
            parts.append(f"<h3>Baselines (Goals)</h3>{self._wrap_table(self.goals_baselines.to_html(index=False))}")
        if self.xg_baselines is not None and not iteration:
            parts.append(f"<h3>Baselines (xG)</h3>{self._wrap_table(self.xg_baselines.to_html(index=False))}")
        if not xg_only and self.goals_significance is not None:
            df = self.goals_significance.copy()
            df["significant"] = df["p_value"] < 0.05
            parts.append(f"<h3>Significance (Goals)</h3>{self._wrap_table(df.to_html(index=False))}")
        if self.xg_significance is not None and not iteration:
            df = self.xg_significance.copy()
            df["significant"] = df["p_value"] < 0.05
            parts.append(f"<h3>Significance (xG)</h3>{self._wrap_table(df.to_html(index=False))}")
        if not xg_only and self.goals_val_comparison is not None:
            parts.append(f"<h3>Elo vs Baselines (Goals)</h3>{self._wrap_table(self.goals_val_comparison.to_html(index=False))}")
        if self.xg_val_comparison is not None and not iteration:
            parts.append(f"<h3>Elo vs Baselines (xG)</h3>{self._wrap_table(self.xg_val_comparison.to_html(index=False))}")
        if not xg_only and self.goals_elo_vs_standings is not None:
            parts.append(f"<h3>Elo vs Standings (Goals)</h3>{self._wrap_table(self.goals_elo_vs_standings.to_html(index=False))}")
        if self.xg_elo_vs_standings is not None and not iteration:
            parts.append(f"<h3>Elo vs Standings (xG)</h3>{self._wrap_table(self.xg_elo_vs_standings.to_html(index=False))}")
        return "".join(parts) if len(parts) > 1 else parts[0]

    def _section_rankings(self, xg_only: bool = False, iteration: Optional[str] = None) -> str:
        """Team rankings: all teams side-by-side with rank."""
        parts = ['<h2 id="rankings">Rankings</h2><div style="display:flex;gap:2em">']
        lab = {"1.0": "Goals", "1.1": "xG", "2.0": "Off/Def"}.get(iteration) if iteration else "Goals"
        if not xg_only and self.goals_summary:
            # Use iteration-specific rankings when viewing 1.0/1.1/2.0 dashboards
            by_it = (self.goals_summary or {}).get("team_rankings_by_iteration") or {}
            tr = by_it.get(str(iteration)) if iteration else self.goals_summary.get("team_rankings")
            if tr is None:
                tr = self.goals_summary.get("team_rankings", {})
        else:
            tr = {}
        if tr:
            ranked = sorted(tr.items(), key=lambda x: -x[1])
            n = len(ranked)
            tbl = "<table><tr><th>Rank</th><th>Team</th><th>Rating</th></tr>"
            for i, (t, r) in enumerate(ranked, 1):
                tbl += f"<tr><td>{i}</td><td>{t}</td><td>{r:.1f}</td></tr>"
            tbl += "</table>"
            parts.append(f'<div><h3>{lab} ({n} teams)</h3>{self._wrap_table(tbl)}</div>')
        if self.xg_summary and "team_rankings" in self.xg_summary and not iteration:
            tr = self.xg_summary["team_rankings"]
            ranked = sorted(tr.items(), key=lambda x: -x[1])
            n = len(ranked)
            tbl = "<table><tr><th>Rank</th><th>Team</th><th>Rating</th></tr>"
            for i, (t, r) in enumerate(ranked, 1):
                tbl += f"<tr><td>{i}</td><td>{t}</td><td>{r:.1f}</td></tr>"
            tbl += "</table>"
            parts.append(f'<div><h3>xG ({n} teams)</h3>{self._wrap_table(tbl)}</div>')
        parts.append("</div>")
        return "".join(parts)

    def _section_all_teams(self, r1: pd.DataFrame, xg_only: bool = False, iteration: Optional[str] = None) -> str:
        """List every team in Round 1 with rank(s), so user can verify all teams are present."""
        home_col = "home_team" if "home_team" in r1.columns else "home"
        away_col = "away_team" if "away_team" in r1.columns else "away"
        if home_col not in r1.columns or away_col not in r1.columns:
            return ""
        teams = set()
        for _, row in r1.iterrows():
            h = str(row[home_col]).strip() if pd.notna(row[home_col]) else ""
            a = str(row[away_col]).strip() if pd.notna(row[away_col]) else ""
            if h:
                teams.add(h)
            if a:
                teams.add(a)
        if not teams:
            return ""

        def _rank_and_rating(tr: dict, t: str, t_lower: str) -> Tuple[Optional[int], Optional[float]]:
            rating = tr.get(t_lower) or tr.get(t)
            if rating is None:
                return (999, None)
            rl = sorted(tr.items(), key=lambda x: -x[1])
            rank = next((i + 1 for i, (k, _) in enumerate(rl) if k == t_lower or k == t), 999)
            return (rank, rating)

        by_it = (self.goals_summary or {}).get("team_rankings_by_iteration") or {}
        tr_goals = by_it.get(str(iteration)) if iteration else (self.goals_summary or {}).get("team_rankings") or {}
        if not tr_goals and self.goals_summary:
            tr_goals = self.goals_summary.get("team_rankings") or {}
        tr_xg = (self.xg_summary or {}).get("team_rankings") or {}
        has_both = tr_goals and tr_xg and not xg_only and not iteration
        use_xg_for_single = xg_only or iteration == "1.1"

        rows = []
        for t in teams:
            t_lower = t.lower().replace(" ", "_")
            r_g, rat_g = _rank_and_rating(tr_goals, t, t_lower)
            r_x, rat_x = _rank_and_rating(tr_xg, t, t_lower)
            if has_both:
                sort_rank = r_g
            else:
                use_r = r_x if use_xg_for_single and tr_xg else r_g
                sort_rank = use_r if (tr_xg if use_xg_for_single else tr_goals) else 999
            if has_both:
                rows.append({
                    "Team": t,
                    "Rank_G": r_g, "Rating_G": rat_g,
                    "Rank_xG": r_x, "Rating_xG": rat_x,
                    "_sort": sort_rank,
                })
            else:
                use_r = r_x if (use_xg_for_single and tr_xg) else r_g
                use_rat = rat_x if (use_xg_for_single and tr_xg) else rat_g
                rows.append({
                    "Team": t,
                    "Rank": use_r,
                    "Rating": use_rat,
                    "_sort": sort_rank,
                })
        df = pd.DataFrame(rows)
        df = df.sort_values("_sort").reset_index(drop=True)
        df = df.drop(columns=["_sort"], errors="ignore")

        if has_both:
            df["Rank_G"] = df["Rank_G"].apply(lambda x: "—" if x == 999 else int(x))
            df["Rating_G"] = df["Rating_G"].apply(lambda x: "—" if x is None else f"{x:.1f}")
            df["Rank_xG"] = df["Rank_xG"].apply(lambda x: "—" if x == 999 else int(x))
            df["Rating_xG"] = df["Rating_xG"].apply(lambda x: "—" if x is None else f"{x:.1f}")
        else:
            df["Rank"] = df["Rank"].apply(lambda x: "—" if x == 999 else int(x))
            df["Rating"] = df["Rating"].apply(lambda x: "—" if x is None else f"{x:.1f}")
        n = len(rows)
        return f'<h3>All teams in Round 1 ({n})</h3>{self._wrap_table(df.to_html(index=False))}'

    def create_r1_comparison_table(self) -> Optional[pd.DataFrame]:
        """Merge Round 1 predictions from goals and xG for side-by-side comparison."""
        if self.goals_r1 is None and self.xg_r1 is None:
            return None

        if self.goals_r1 is None:
            return self.xg_r1
        if self.xg_r1 is None:
            return self.goals_r1

        # Align by matchup (home vs away)
        g = self.goals_r1.copy()
        x = self.xg_r1.copy()

        # Common columns for matching
        home_col = "home_team" if "home_team" in g.columns else "home"
        away_col = "away_team" if "away_team" in g.columns else "away"
        if home_col not in g.columns and "matchup" in g.columns:
            return pd.concat([g.add_suffix("_goals"), x.add_suffix("_xg")], axis=1)

        if "predicted_winner" in g.columns:
            g = g.rename(columns={"predicted_winner": "pred_winner_goals"})
        elif "pred_winner" in g.columns:
            g = g.rename(columns={"pred_winner": "pred_winner_goals"})
        if "confidence" in g.columns:
            g = g.rename(columns={"confidence": "win_prob_goals"})
        elif "win_prob" in g.columns:
            g = g.rename(columns={"win_prob": "win_prob_goals"})
        if "predicted_winner" in x.columns:
            x = x.rename(columns={"predicted_winner": "pred_winner_xg"})
        elif "pred_winner" in x.columns:
            x = x.rename(columns={"pred_winner": "pred_winner_xg"})
        if "confidence" in x.columns:
            x = x.rename(columns={"confidence": "win_prob_xg"})
        elif "win_prob" in x.columns:
            x = x.rename(columns={"win_prob": "win_prob_xg"})

        if home_col in g.columns and home_col in x.columns:
            merged = pd.merge(
                g[[c for c in g.columns if c in (home_col, away_col, "pred_winner_goals", "win_prob_goals")]],
                x[[c for c in x.columns if c in (home_col, away_col, "pred_winner_xg", "win_prob_xg")]],
                on=[home_col, away_col],
                how="outer",
            )
            return merged
        return pd.concat([g, x], axis=1)

    def save_html(self, path: str, xg_only: bool = False, iteration: Optional[str] = None) -> bool:
        """Save comprehensive interactive HTML dashboard. If xg_only, show only xG pipeline content.
        If iteration is '1.0', '1.1', or '2.0', show only that iteration's data from the unified sweep."""
        fig = self.create_figure(xg_only=xg_only, iteration=iteration)
        if fig is None:
            return False

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        k_metrics_data = self._build_k_metrics_json(xg_only=xg_only, iteration=iteration)
        k_download_html = self._k_metrics_download_section(k_metrics_data) if k_metrics_data else ""

        sections = []
        sections.append(self._section_nav())
        sections.append(self._section_overview(xg_only=xg_only, iteration=iteration))
        sections.append(self._section_hyperparams(xg_only=xg_only, iteration=iteration))
        sections.append(f'<h2 id="k-metrics">K vs Metrics</h2>{k_download_html}' + fig.to_html(full_html=False, include_plotlyjs=False))
        sections.append(self._section_validation(xg_only=xg_only, iteration=iteration))

        cal_fig = self.create_calibration_figure(xg_only=xg_only, iteration=iteration)
        if cal_fig is not None:
            sections.append('<h2 id="calibration">Calibration</h2>' + cal_fig.to_html(full_html=False, include_plotlyjs=False))

        sections.append(self._section_rankings(xg_only=xg_only, iteration=iteration))

        r1 = self.create_r1_comparison_table()
        if r1 is not None and len(r1) > 0:
            all_teams_html = self._section_all_teams(r1, xg_only=xg_only, iteration=iteration)
            sections.append(f'<h2 id="round1">Round 1</h2>{all_teams_html}{self._wrap_table(r1.to_html(classes="table", index=False))}')

        _titles = {"1.0": "Baseline Elo 1.0 (Goals)", "1.1": "Baseline Elo 1.1 (xG)", "2.0": "Baseline Elo 2.0 (Off/Def)"}
        title = _titles.get(str(iteration)) if iteration else ("Baseline Elo xG" if xg_only else "Baseline Elo")
        css = """
        <style>
        body { font-family: sans-serif; margin: 1.5em; max-width: 1400px; }
        h1 { border-bottom: 1px solid #ccc; padding-bottom: 0.3em; }
        h2 { margin-top: 1.5em; color: #333; }
        h3 { margin-top: 1em; font-size: 1em; }
        table { border-collapse: collapse; margin: 0.5em 0; font-size: 0.9em; }
        th, td { border: 1px solid #ddd; padding: 4px 8px; text-align: left; }
        th { background: #f5f5f5; }
        tr:nth-child(even) { background: #fafafa; }
        """ + QOL_CSS + """
        </style>
        """

        k_data_script = ""
        if k_metrics_data:
            k_data_script = f'<script type="application/json" id="k-metrics-data">{json.dumps(k_metrics_data)}</script>'

        full_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
{css}
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
<h1>{title}</h1>
{"".join(sections)}
{k_data_script}
<div id="toast"></div>
<script>
{QOL_JS}
</script>
</body>
</html>"""

        out.write_text(full_html, encoding="utf-8")
        logger.info(f"Dashboard saved to {out}")
        return True

    def save_matplotlib(self, path: str) -> bool:
        """Save matplotlib figure as fallback when Plotly is unavailable."""
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib not available")
            return False

        has_goals = self.goals_k_metrics is not None and len(self.goals_k_metrics) > 0
        has_xg = self.xg_k_metrics is not None and len(self.xg_k_metrics) > 0
        if not has_goals and not has_xg:
            return False

        metrics = ["accuracy", "brier_loss", "log_loss", "combined_rmse"]
        n = sum(1 for m in metrics
                if (has_goals and m in (self.goals_k_metrics or pd.DataFrame()).columns)
                or (has_xg and m in (self.xg_k_metrics or pd.DataFrame()).columns))
        if n == 0:
            return False

        n_cols = min(2, n)
        n_rows = (n + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 4 * n_rows))
        if n == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        idx = 0
        for metric in metrics:
            if (has_goals and self.goals_k_metrics is not None and metric in self.goals_k_metrics.columns) or \
               (has_xg and self.xg_k_metrics is not None and metric in self.xg_k_metrics.columns):
                ax = axes[idx] if n > 1 else axes[0]
                if has_goals and metric in self.goals_k_metrics.columns:
                    gdf = self._ensure_k_column(self.goals_k_metrics)
                    k_col = "k" if "k" in gdf.columns else "k_factor"
                    ax.plot(gdf[k_col], gdf[metric], "o-", label="Goals", color="#1f77b4")
                if has_xg and metric in self.xg_k_metrics.columns:
                    xdf = self._ensure_k_column(self.xg_k_metrics)
                    k_col = "k" if "k" in xdf.columns else "k_factor"
                    ax.plot(xdf[k_col], xdf[metric], "s-", label="xG", color="#ff7f0e")
                ax.set_xlabel("k")
                ax.set_ylabel(metric)
                ax.set_title(metric)
                ax.legend()
                ax.grid(True, alpha=0.3)
                idx += 1

        for j in range(idx, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Baseline Elo: Goals vs xG K-Metrics")
        plt.tight_layout()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        return True


def run_dashboard(
    output_path: str = "output/predictions/baseline_dashboard.html",
    xg_only: bool = False,
    also_xg: bool = False,
    also_iterations: bool = True,
) -> int:
    """CLI entry point: load and save dashboard. If also_xg, save xG-only. If also_iterations, save 1.0, 1.1, 2.0."""
    dash = BaselineResultsDashboard()
    if not dash.load():
        print("No baseline k_metrics found. Run _run_baseline_elo_sweep.py and _run_baseline_elo_xg_sweep.py first.")
        return 1
    ok = False
    if PLOTLY_AVAILABLE:
        ok = dash.save_html(output_path, xg_only=xg_only)
        if also_xg and dash.xg_k_metrics is not None:
            xg_path = str(Path(output_path).parent / "baseline_elo_xg_dashboard.html")
            ok_xg = dash.save_html(xg_path, xg_only=True)
            ok = ok or ok_xg
            if ok_xg:
                print(f"[OK] xG dashboard saved to {xg_path}")
        if also_iterations and dash.goals_k_metrics_full is not None and "model_iteration" in dash.goals_k_metrics_full.columns:
            parent = Path(output_path).parent
            for it in ("1.0", "1.1", "2.0"):
                sub = dash.goals_k_metrics_full[dash.goals_k_metrics_full["model_iteration"].astype(str) == it]
                if len(sub) > 0:
                    p = str(parent / f"baseline_elo_{it.replace('.', '_')}_dashboard.html")
                    if dash.save_html(p, iteration=it):
                        ok = True
                        print(f"[OK] {it} dashboard saved to {p}")
    else:
        p = Path(output_path)
        ok = dash.save_matplotlib(str(p.with_suffix(".png")))
    print(f"[OK] Dashboard saved" if ok else "[WARN] Could not save dashboard")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    import os
    _cwd = Path(os.path.abspath("")).resolve()
    if (_cwd / "python").is_dir():
        os.chdir(_cwd / "python")
    also = "--xg" in sys.argv or "--also-xg" in sys.argv
    sys.exit(run_dashboard(also_xg=also))
