"""
Offensive Disparity vs Team Strength — Competition Visualization
================================================================
Answers: "Are teams with more evenly-matched offensive lines more likely to succeed?"

X-axis: Offensive Line Disparity |O_L1 - O_L2| (absolute gap between lines)
Y-axis: Overall Team Elo

Location: python/scripts/elo_disparity_plot.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from adjustText import adjust_text
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE = Path(__file__).resolve().parent.parent  # python/

FULL_RATINGS = (
    BASE / "output" / "predictions" / "baseline_elo" / "sweep"
    / "offdef_line_ratings_full.csv"
)
FALLBACK_RATINGS = (
    BASE / "output" / "predictions" / "baseline_elo" / "sweep"
    / "offdef_line_ratings.csv"
)

OUTPUT_DIR = BASE / "output" / "predictions" / "baseline_elo" / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# LOAD DATA
# ============================================================================

print("Loading Off/Def line ratings...")
if FULL_RATINGS.exists():
    df = pd.read_csv(FULL_RATINGS)
    print(f"  Using full ratings: {FULL_RATINGS.name}")
elif FALLBACK_RATINGS.exists():
    df = pd.read_csv(FALLBACK_RATINGS)
    df['O1_O2'] = df['O_first_off'] - df['O_second_off']
    df['Overall'] = 1200 + (df['net_L1'] + df['net_L2']) / 2
    print(f"  Using fallback ratings: {FALLBACK_RATINGS.name}")
else:
    print("ERROR: No ratings file found")
    raise SystemExit(1)

print(f"  Loaded {len(df)} teams")

# ============================================================================
# CALCULATE METRICS
# ============================================================================

# Line quality disparity = |(O_L1 + D_L1) - (O_L2 + D_L2)|
# = |total L1 rating - total L2 rating|
df['line_disparity'] = (df['O1_O2'] + df['D1_D2']).abs()
ABBREVIATIONS = {'Uk': 'UK', 'Usa': 'USA', 'Uae': 'UAE'}
df['team_display'] = (
    df['team'].str.replace('_', ' ').str.title()
    .replace(ABBREVIATIONS, regex=False)
)

print(f"  Overall Elo range: {df['Overall'].min():.1f} to {df['Overall'].max():.1f}")
print(f"  Line quality disparity range: {df['line_disparity'].min():.1f} to {df['line_disparity'].max():.1f}")

r = np.corrcoef(df['line_disparity'], df['Overall'])[0, 1]
r2 = r ** 2
print(f"  Correlation: r = {r:.3f}, R^2 = {r2:.3f}")

# ============================================================================
# FIGURE
# ============================================================================

print("\nGenerating visualization...")

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Segoe UI', 'Helvetica Neue', 'Arial', 'sans-serif'],
    'axes.spines.top': False,
    'axes.spines.right': False,
})

fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor('white')
ax.set_facecolor('#fafafa')

league_avg = df['Overall'].mean()
above = df['Overall'] >= league_avg

ax.scatter(
    df.loc[above, 'line_disparity'], df.loc[above, 'Overall'],
    s=120, color='#2e7d32', alpha=0.80, edgecolors='white', linewidth=0.9,
    zorder=3, label='Above league average',
)
ax.scatter(
    df.loc[~above, 'line_disparity'], df.loc[~above, 'Overall'],
    s=120, color='#c62828', alpha=0.80, edgecolors='white', linewidth=0.9,
    zorder=3, label='Below league average',
)

# Regression line
z = np.polyfit(df['line_disparity'], df['Overall'], 1)
p = np.poly1d(z)
x_line = np.linspace(0, df['line_disparity'].max() + 2, 100)
slope_dir = "negative" if z[0] < 0 else "positive"
ax.plot(x_line, p(x_line), color='#37474f', linestyle='--', linewidth=1.5,
        alpha=0.55, zorder=2)

# League average
ax.axhline(league_avg, color='#9e9e9e', linestyle=':', linewidth=1, alpha=0.5,
           zorder=1)
ax.text(
    -0.5, league_avg + 0.4,
    f'League avg ({league_avg:.1f})',
    fontsize=7, color='#757575', ha='left', va='bottom',
)

# Team labels (using adjustText to avoid overlaps)
texts = []
for _, row in df.iterrows():
    clr = '#1b5e20' if row['Overall'] >= league_avg else '#b71c1c'
    texts.append(
        ax.text(
            row['line_disparity'], row['Overall'],
            row['team_display'],
            fontsize=6.5, color=clr, alpha=0.85, zorder=4,
        )
    )
adjust_text(
    texts, ax=ax,
    arrowprops=dict(arrowstyle='-', color='#bdbdbd', lw=0.4),
    expand=(1.4, 1.6),
    force_text=(0.6, 0.8),
    force_points=(0.4, 0.5),
    lim=500,
)

# Title & subtitle
ax.set_title(
    'Offensive Line Quality Disparity vs. Team Strength',
    fontsize=14, fontweight='bold', pad=28, loc='left', color='#212121',
)
ax.text(
    0.0, 1.04,
    f'WHL 2025  |  {len(df)} teams  |  Baseline Elo Off/Def 2.0  |  r = {r:.2f}',
    transform=ax.transAxes, fontsize=9.5, color='#616161',
    verticalalignment='bottom',
)

# Axes
ax.set_xlabel(
    'Line Quality Disparity  (Elo points)',
    fontsize=10.5, labelpad=8, color='#424242',
)
ax.set_ylabel(
    'Overall Team Elo Rating',
    fontsize=10.5, labelpad=8, color='#424242',
)
ax.set_xlim(-1, df['line_disparity'].max() + 3)
y_margin = (df['Overall'].max() - df['Overall'].min()) * 0.08
ax.set_ylim(df['Overall'].min() - y_margin - 1, df['Overall'].max() + y_margin + 1)

ax.grid(True, alpha=0.20, linestyle='-', linewidth=0.4, zorder=0)
ax.tick_params(labelsize=9, colors='#424242')

# Legend
trend_line = Line2D([0], [0], color='#37474f', linestyle='--', lw=1.5, alpha=0.55)
legend_handles = [
    plt.scatter([], [], s=80, color='#2e7d32', edgecolors='white', lw=0.9),
    plt.scatter([], [], s=80, color='#c62828', edgecolors='white', lw=0.9),
    trend_line,
]
legend_labels = [
    'Above league average',
    'Below league average',
    f'Linear trend (r = {r:.2f}, R$^2$ = {r2:.3f})',
]
ax.legend(
    legend_handles, legend_labels,
    loc='upper right', fontsize=8, framealpha=0.92,
    edgecolor='#e0e0e0', fancybox=True,
)

# Finding callout
if abs(r) < 0.15:
    finding_lines = [
        f'r = {r:.2f},  R$^2$ = {r2:.3f}',
        'Line quality disparity shows no',
        'meaningful relationship with',
        'overall team strength.',
    ]
else:
    finding_lines = [
        f'r = {r:.2f},  R$^2$ = {r2:.3f}',
        'Teams with greater line disparity',
        'tend to perform slightly worse,',
        'favoring balanced deployment.',
    ]
props = dict(boxstyle='round,pad=0.5', facecolor='white',
             edgecolor='#b0bec5', alpha=0.92, linewidth=0.8)
ax.text(
    0.02, 0.04, '\n'.join(finding_lines),
    transform=ax.transAxes, fontsize=8, verticalalignment='bottom',
    bbox=props, color='#37474f', linespacing=1.4,
)

# Caption
caption = (
    'Disparity = |(O$_{L1}$ + D$_{L1}$) $-$ (O$_{L2}$ + D$_{L2}$)|, '
    'the absolute gap in combined offensive-defensive Elo between a team\'s two lines.  '
    f'Slope = {z[0]:.2f} Elo per disparity point.'
)
fig.text(
    0.10, 0.005, caption,
    fontsize=7, color='#9e9e9e', ha='left', va='bottom', style='italic',
)

plt.tight_layout(rect=[0, 0.03, 1, 1])

output_png = OUTPUT_DIR / "elo_offensive_disparity.png"
output_pdf = OUTPUT_DIR / "elo_offensive_disparity.pdf"
plt.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(output_pdf, bbox_inches='tight', facecolor='white')
print(f"  PNG: {output_png}")
print(f"  PDF: {output_pdf}")

# ============================================================================
# WEBSITE TABLE
# ============================================================================

print("\nGenerating website table...")

table_df = df[[
    'team_display', 'O_first_off', 'O_second_off', 'O1_O2',
    'D_first_off', 'D_second_off', 'D1_D2',
    'net_L1', 'net_L2', 'line_disparity', 'Overall',
]].copy()
table_df.columns = [
    'Team', 'O (L1)', 'O (L2)', 'O gap',
    'D (L1)', 'D (L2)', 'D gap',
    'Net L1', 'Net L2', 'Line Disp', 'Overall Elo',
]
table_df = table_df.sort_values('Overall Elo', ascending=False).reset_index(drop=True)

rows_html = []
for i, row in table_df.iterrows():
    elo = row['Overall Elo']
    elo_class = 'above' if elo >= league_avg else 'below'
    disp = row['Line Disp']
    disp_class = 'high-disp' if disp >= 25 else ('mid-disp' if disp >= 15 else 'low-disp')
    rows_html.append(
        f"<tr class='{elo_class} {disp_class}'>"
        f"<td>{i+1}</td>"
        f"<td class='team'>{row['Team']}</td>"
        f"<td>{row['O (L1)']:.1f}</td>"
        f"<td>{row['O (L2)']:.1f}</td>"
        f"<td class='disp'>{row['O gap']:+.1f}</td>"
        f"<td>{row['D (L1)']:.1f}</td>"
        f"<td>{row['D (L2)']:.1f}</td>"
        f"<td class='disp'>{row['D gap']:+.1f}</td>"
        f"<td>{row['Net L1']:+.1f}</td>"
        f"<td>{row['Net L2']:+.1f}</td>"
        f"<td class='abs-disp'>{disp:.1f}</td>"
        f"<td class='elo'>{elo:.1f}</td>"
        f"</tr>"
    )

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Offensive Disparity Analysis</title>
<style>
  body {{ font-family: system-ui, -apple-system, sans-serif; margin: 0; padding: 1rem 1.5rem; background: #fafafa; color: #222; }}
  .nav {{ margin-bottom: 1.5rem; font-size: 0.9em; }}
  .nav a {{ color: #06c; text-decoration: none; }}
  .nav a:hover {{ text-decoration: underline; }}
  h1 {{ font-size: 1.5rem; margin: 0 0 0.25rem; }}
  .subtitle {{ color: #666; font-size: 0.95rem; margin-bottom: 1rem; }}
  .finding {{ background: #e8f4ec; border-left: 4px solid #2a9d4e; padding: 0.75rem 1rem; margin-bottom: 1.5rem; border-radius: 4px; }}
  .finding strong {{ color: #1a6b32; }}
  .stats {{ display: flex; gap: 1.5rem; margin-bottom: 1.5rem; flex-wrap: wrap; }}
  .stat-card {{ background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 0.75rem 1rem; min-width: 140px; }}
  .stat-card .label {{ font-size: 0.75rem; color: #888; text-transform: uppercase; letter-spacing: 0.05em; }}
  .stat-card .value {{ font-size: 1.4rem; font-weight: 700; margin-top: 0.2rem; }}
  .scroll {{ max-height: 70vh; overflow-y: auto; border: 1px solid #ddd; border-radius: 6px; background: #fff; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.85em; }}
  th {{ position: sticky; top: 0; background: #f5f5f5; border-bottom: 2px solid #ccc; padding: 8px 10px; text-align: right; font-weight: 600; white-space: nowrap; }}
  th:nth-child(1), th:nth-child(2) {{ text-align: left; }}
  td {{ padding: 6px 10px; border-bottom: 1px solid #eee; text-align: right; font-variant-numeric: tabular-nums; }}
  td:nth-child(1), td:nth-child(2) {{ text-align: left; }}
  td.team {{ font-weight: 500; }}
  td.disp {{ font-weight: 600; }}
  td.elo {{ font-weight: 700; }}
  tr.above td.elo {{ color: #1a7a2e; }}
  tr.below td.elo {{ color: #a63030; }}
  tr.high-disp td.abs-disp {{ background: #fff3e0; color: #e65100; font-weight: 700; }}
  tr.low-disp td.abs-disp {{ background: #e8f5e9; color: #2e7d32; }}
  tr:hover {{ background: #f5faff; }}
  .caption {{ margin-top: 1rem; font-size: 0.8rem; color: #888; line-height: 1.5; }}
  .img-wrap {{ margin: 1.5rem 0; text-align: center; }}
  .img-wrap img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 6px; }}
</style>
</head>
<body>
<nav class="nav"><a href="index.html">&larr; Dashboards</a> | <a href="model_values.html">Model values</a></nav>

<h1>Offensive Line Disparity Analysis</h1>
<p class="subtitle">Does offensive line balance predict team success? &mdash; Baseline Elo Off/Def 2.0</p>

<div class="finding">
  <strong>Finding:</strong> r&nbsp;=&nbsp;{r:.2f}, R&sup2;&nbsp;=&nbsp;{r2:.3f} &mdash;
  {"Teams with larger offensive line disparity tend to perform slightly worse, suggesting balanced line deployment may be beneficial."
   if r < -0.15 else
   "Offensive line balance shows little meaningful relationship with overall team strength. Teams succeed or fail regardless of how they distribute offensive talent."}
</div>

<div class="stats">
  <div class="stat-card"><div class="label">Correlation (r)</div><div class="value">{r:.2f}</div></div>
  <div class="stat-card"><div class="label">R&sup2;</div><div class="value">{r2:.3f}</div></div>
  <div class="stat-card"><div class="label">Teams</div><div class="value">{len(df)}</div></div>
  <div class="stat-card"><div class="label">Avg |Disparity|</div><div class="value">{df['line_disparity'].mean():.1f}</div></div>
  <div class="stat-card"><div class="label">Elo Range</div><div class="value">{df['Overall'].min():.0f}&ndash;{df['Overall'].max():.0f}</div></div>
</div>

<div class="img-wrap">
  <img src="../output/predictions/baseline_elo/plots/elo_offensive_disparity.png"
       alt="Offensive Disparity vs Team Elo scatter plot">
</div>

<h2 style="font-size:1.1rem; margin-top:2rem;">Team Line Ratings &amp; Disparity</h2>
<p class="subtitle">Sorted by Overall Elo (strongest first). Green = above league avg; red = below.</p>

<div class="scroll">
<table>
<tr><th>#</th><th>Team</th><th>O (L1)</th><th>O (L2)</th><th>O gap</th><th>D (L1)</th><th>D (L2)</th><th>D gap</th><th>Net L1</th><th>Net L2</th><th>Line Disp</th><th>Overall Elo</th></tr>
{''.join(rows_html)}
</table>
</div>

<p class="caption">
  <strong>O (L1/L2)</strong> = Offensive Elo for Line 1 / Line 2. Higher = stronger offense.<br>
  <strong>D (L1/L2)</strong> = Defensive Elo. Lower = stronger defense.<br>
  <strong>O gap / D gap</strong> = L1 &minus; L2 for offense / defense.<br>
  <strong>Line Disp</strong> = |(O+D)_L1 - (O+D)_L2|. Total quality gap between lines.<br>
  <strong>Net</strong> = O &minus; D per line.<br>
  <em>Source: Baseline Elo Off/Def 2.0, WHL 2025 shift-level data.</em>
</p>
</body>
</html>"""

docs_dir = BASE.parent / 'docs'
if not docs_dir.is_dir():
    docs_dir = BASE / 'docs'
html_path = docs_dir / 'offensive_disparity.html'
html_path.write_text(html, encoding='utf-8')
print(f"  HTML: {html_path}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"\nr = {r:.3f}, R^2 = {r2:.3f}, slope = {z[0]:.2f}")
print(f"|Disparity| range: {df['line_disparity'].min():.1f} to {df['line_disparity'].max():.1f}")
print(f"Overall Elo range: {df['Overall'].min():.1f} to {df['Overall'].max():.1f}")

print(f"\nMost balanced (lowest |disparity|):")
for _, row in df.nsmallest(5, 'line_disparity').iterrows():
    print(f"  {row['team_display']:20s}  |disp|={row['line_disparity']:.1f}  Elo={row['Overall']:.1f}")

print(f"\nMost top-heavy (highest |disparity|):")
for _, row in df.nlargest(5, 'line_disparity').iterrows():
    print(f"  {row['team_display']:20s}  |disp|={row['line_disparity']:.1f}  Elo={row['Overall']:.1f}")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
