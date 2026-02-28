# Validation Summary

## Naive Baselines
- Constant 50%: Brier~0.25, Log~0.6931
- League avg: Brier~0.25, Log~0.6931

## Best per iteration vs baselines
- **2.0** (k=25.1): acc=66.4%, brier=0.2332, log=0.6593
  - Beats constant 50% acc: True
  - Beats Brier baseline: True

## Statistical significance (win accuracy > 50%)
- 2.0: p=0.0000 (significant)

## Elo vs standings (Spearman)
- 2.0: r=0.0367, p=0.8421