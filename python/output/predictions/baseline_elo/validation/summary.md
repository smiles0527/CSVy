# Validation Summary

## Naive Baselines
- Constant 50%: Brier~0.25, Log~0.6931
- League avg: Brier~0.25, Log~0.6931

## Best per iteration vs baselines
- **1.0** (k=3.2): acc=53.2%, brier=0.2491, log=0.6913
  - Beats constant 50% acc: True
  - Beats Brier baseline: True
- **1.1** (k=3.6): acc=62.6%, brier=0.2386, log=0.6702
  - Beats constant 50% acc: True
  - Beats Brier baseline: True
- **2.0** (k=4.1): acc=66.9%, brier=0.2370, log=0.6671
  - Beats constant 50% acc: True
  - Beats Brier baseline: True

## Statistical significance (win accuracy > 50%)
- 1.0: p=0.1130 (not significant)
- 1.1: p=0.0000 (significant)
- 2.0: p=0.0000 (significant)

## Elo vs standings (Spearman)
- 1.0: r=0.8677, p=0.0000
- 1.1: r=0.5656, p=0.0007
- 2.0: r=0.0971, p=0.5969