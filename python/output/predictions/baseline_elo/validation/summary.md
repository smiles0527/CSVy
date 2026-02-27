# Validation Summary

## Naive Baselines
- Constant 50%: Brier~0.25, Log~0.6931
- League avg: Brier~0.25, Log~0.6931

## Best per iteration vs baselines
- **1.0** (k=5.0): acc=53.4%, brier=0.2495, log=0.6922
  - Beats constant 50% acc: True
  - Beats Brier baseline: True
- **1.1** (k=5.0): acc=62.8%, brier=0.2358, log=0.6646
  - Beats constant 50% acc: True
  - Beats Brier baseline: True
- **2.0** (k=5.0): acc=66.9%, brier=0.2361, log=0.6651
  - Beats constant 50% acc: True
  - Beats Brier baseline: True

## Statistical significance (win accuracy > 50%)
- 1.0: p=0.0948 (not significant)
- 1.1: p=0.0000 (significant)
- 2.0: p=0.0000 (significant)

## Elo vs standings (Spearman)
- 1.0: r=0.8625, p=0.0000
- 1.1: r=0.5652, p=0.0007
- 2.0: r=0.1052, p=0.5666