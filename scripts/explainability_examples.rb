#!/usr/bin/env ruby
# Example: Complete explainability and debugging workflow

require_relative '../lib/model_explainer'

# Example 1: Generate SHAP Report
puts "=" * 60
puts "Example 1: SHAP Explainability Report"
puts "=" * 60
puts ""
puts "Command:"
puts "  ruby cli.rb explain-model models/xgboost_model.pkl data/test_features.csv"
puts ""
puts "This generates:"
puts "  - Feature importance bar chart"
puts "  - SHAP summary beeswarm plot"
puts "  - Dependence plots for top features"
puts "  - Interactive HTML report"
puts "  - Raw SHAP values CSV"
puts ""

# Example 2: Debug Features
puts "=" * 60
puts "Example 2: Feature Quality Debugging"
puts "=" * 60
puts ""
puts "Command:"
puts "  ruby cli.rb debug-features data/train.csv"
puts ""
puts "Detects:"
puts "  ✓ Missing values per feature"
puts "  ✓ Constant features (should remove)"
puts "  ✓ Outliers (>3 standard deviations)"
puts "  ✓ High correlations (multicollinearity)"
puts "  ✓ Feature quality scores"
puts ""

# Example 3: Error Analysis
puts "=" * 60
puts "Example 3: Prediction Error Analysis"
puts "=" * 60
puts ""
puts "Command:"
puts "  ruby cli.rb debug-errors predictions.csv actuals.csv features.csv"
puts ""
puts "Finds:"
puts "  ✓ Overall MAE, RMSE, mean error"
puts "  ✓ Error distribution (excellent/good/fair/poor)"
puts "  ✓ Systematic bias (over/underestimation)"
puts "  ✓ Worst predictions"
puts "  ✓ Error patterns by feature ranges"
puts ""

# Example 4: Single Prediction
puts "=" * 60
puts "Example 4: Explain Single Prediction"
puts "=" * 60
puts ""
puts "Command:"
puts "  ruby cli.rb explain-prediction models/xgboost.pkl \\"
puts "    --features GF:250 GA:180 DIFF:70 PTS:98 P%:0.65 rest_time:2 \\"
puts "    -o game_123_explanation"
puts ""
puts "Shows:"
puts "  ✓ Final prediction value"
puts "  ✓ Base value (average prediction)"
puts "  ✓ Top 5 positive contributors"
puts "  ✓ Top 5 negative contributors"
puts "  ✓ Feature contribution breakdown"
puts ""

# Example 5: Complete Workflow
puts "=" * 60
puts "Example 5: Complete Debugging Workflow"
puts "=" * 60
puts ""
puts "Step 1: Check data quality"
puts "  $ ruby cli.rb debug-features data/train.csv"
puts "  → Fix any issues (missing values, outliers, etc.)"
puts ""
puts "Step 2: Train your model"
puts "  (Use Python/DeepNote/Jupyter)"
puts "  $ python train_model.py"
puts ""
puts "Step 3: Make predictions"
puts "  $ python predict.py test.csv > predictions.csv"
puts ""
puts "Step 4: Analyze errors"
puts "  $ ruby cli.rb debug-errors predictions.csv actuals.csv test.csv"
puts "  → Identify systematic issues"
puts ""
puts "Step 5: Explain model behavior"
puts "  $ ruby cli.rb explain-model models/xgboost.pkl test.csv"
puts "  → Understand feature importance"
puts ""
puts "Step 6: Debug specific bad predictions"
puts "  $ ruby cli.rb explain-prediction models/xgboost.pkl \\"
puts "    --features [values from worst prediction]"
puts "  → Understand what went wrong"
puts ""
puts "Step 7: Iterate and improve"
puts "  - Remove low-quality features"
puts "  - Engineer better features for error-prone ranges"
puts "  - Adjust model to fix systematic bias"
puts ""

# Example 6: Model Comparison
puts "=" * 60
puts "Example 6: Compare Multiple Models"
puts "=" * 60
puts ""
puts "Generate SHAP for each model:"
puts "  $ ruby cli.rb explain-model models/xgboost.pkl test.csv -o exp/xgb/"
puts "  $ ruby cli.rb explain-model models/rf.pkl test.csv -o exp/rf/"
puts "  $ ruby cli.rb explain-model models/linear.pkl test.csv -o exp/linear/"
puts ""
puts "Compare feature_importance.csv files:"
puts "  - Which features are consistently important?"
puts "  - Where do models disagree? → Good for ensembles"
puts "  - Are important features domain-sensible?"
puts ""

# Tips
puts "=" * 60
puts "💡 Tips & Best Practices"
puts "=" * 60
puts ""
puts "1. Always debug features BEFORE training"
puts "   → Garbage in = garbage out"
puts ""
puts "2. Check for systematic bias in errors"
puts "   → Mean error should be ~0"
puts ""
puts "3. Validate feature importance makes sense"
puts "   → If random features are important, you're overfitting"
puts ""
puts "4. Use SHAP to build trust in your model"
puts "   → Show stakeholders WHY predictions make sense"
puts ""
puts "5. Iterate based on error analysis"
puts "   → Engineer features for high-error ranges"
puts ""
puts "6. Save HTML reports for documentation"
puts "   → Track insights over time"
puts ""

puts "=" * 60
puts " Full Documentation"
puts "=" * 60
puts ""
puts "See: docs/guides/EXPLAINABILITY_DEBUG_GUIDE.md"
puts ""
