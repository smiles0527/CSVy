# Validation Summary

## Naive Baselines
- Constant 50%: Brier~0.25, Log~0.6931
- League avg: Brier~0.25, Log~0.6931

## Best per iteration vs baselines
- **2.0** (k=100.0): acc=64.9%, brier=0.2474, log=0.6880
  - Beats constant 50% acc: True
  - Beats Brier baseline: True

## Statistical significance (win accuracy > 50%)
- 2.0: p=0.0000 (significant)

## Elo vs standings (Spearman)
- 2.0: r=0.0271, p=0.8828