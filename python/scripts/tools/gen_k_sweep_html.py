#!/usr/bin/env python3
"""Generate docs/k_sweep_2_0.html from k_metrics_2_0.csv."""
import os
import pandas as pd
from pathlib import Path

_script = Path(__file__).resolve()
_python_dir = _script.parent
while True:
    if (_python_dir / 'utils').is_dir():
        break
    parent = _python_dir.parent
    if parent == _python_dir:
        raise RuntimeError('Cannot locate python/')
    _python_dir = parent
os.chdir(_python_dir)

csv_path = Path('output/predictions/baseline_elo/sweep/k_metrics_2_0.csv')
docs_dir = _python_dir.parent / 'docs'

df = pd.read_csv(csv_path)
df = df.sort_values('accuracy', ascending=False).reset_index(drop=True)
best = df.iloc[0]
best_k = best['k']

rows = []
for _, r in df.iterrows():
    rows.append(f"<tr><td>{r['k']}</td><td>{r['accuracy']:.4f}</td><td>{r['brier_loss']:.4f}</td><td>{r['log_loss']:.4f}</td><td>{r['combined_rmse']:.4f}</td></tr>")

html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>2.0 k-sweep</title>
<style>body{{font-family:system-ui,sans-serif;margin:1rem;max-width:900px}} table{{border-collapse:collapse;font-size:0.9em}} th,td{{border:1px solid #ddd;padding:4px 8px;text-align:right}} th{{background:#f5f5f5}} .nav{{margin-bottom:1rem}} .scroll{{max-height:70vh;overflow-y:auto}}</style></head>
<body>
<nav class="nav"><a href="index.html">← Dashboards</a> | <a href="model_values.html">Model values</a></nav>
<h1>2.0 Off/Def k-sweep</h1>
<p>{len(df)} k values (0.1 to 100). Sorted by accuracy (best first). Best: k={best_k}.</p>
<div class="scroll"><table>
<tr><th>k</th><th>accuracy</th><th>brier_loss</th><th>log_loss</th><th>combined_rmse</th></tr>
{''.join(rows)}
</table></div>
</body></html>"""

out = docs_dir / 'k_sweep_2_0.html'
out.write_text(html, encoding='utf-8')
print(f'[OK] {out}')
