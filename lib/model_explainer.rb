require 'csv'
require 'json'
require 'fileutils'
require 'logger'

# ModelExplainer - SHAP-based model explainability and debugging
class ModelExplainer
  attr_reader :logger

  def initialize(logger: nil)
    @logger = logger || Logger.new(STDOUT)
  end

  # Generate SHAP values for a trained model
  def explain_predictions(model_path:, data_path:, output_dir:, model_type: 'xgboost', top_n: 20)
    validate_inputs(model_path, data_path)
    FileUtils.mkdir_p(output_dir)

    python_script = generate_shap_script(
      model_path: model_path,
      data_path: data_path,
      output_dir: output_dir,
      model_type: model_type,
      top_n: top_n
    )

    script_path = File.join(output_dir, 'run_shap.py')
    File.write(script_path, python_script)

    @logger.info "Running SHAP analysis..."
    result = system('python', script_path)

    unless result
      raise "SHAP analysis failed. Check #{script_path} for errors."
    end

    {
      summary_plot: File.join(output_dir, 'shap_summary.png'),
      importance_plot: File.join(output_dir, 'shap_importance.png'),
      dependence_plots: File.join(output_dir, 'dependence'),
      values_csv: File.join(output_dir, 'shap_values.csv'),
      report: File.join(output_dir, 'shap_report.html')
    }
  end

  # Analyze prediction errors and identify patterns
  def analyze_errors(predictions_path:, actuals_path:, features_path:, output_path:)
    predictions = load_csv(predictions_path)
    actuals = load_csv(actuals_path)
    features = load_csv(features_path)
    
    # Validate row counts match
    unless predictions.size == actuals.size && actuals.size == features.size
      raise "Row count mismatch: predictions=#{predictions.size}, actuals=#{actuals.size}, features=#{features.size}"
    end

    errors = calculate_errors(predictions, actuals, features)
    error_analysis = {
      overall: overall_statistics(errors),
      by_magnitude: group_by_error_magnitude(errors, features),
      by_feature_range: analyze_feature_ranges(errors, features),
      worst_predictions: identify_worst_predictions(errors, features, top_n: 20),
      systematic_bias: detect_systematic_bias(errors, features)
    }
    
    # Ensure output directory exists
    FileUtils.mkdir_p(File.dirname(output_path))

    save_error_analysis(error_analysis, output_path)
    generate_error_report(error_analysis, output_path)

    error_analysis
  end

  # Debug feature distributions and detect anomalies
  def debug_features(data_path:, output_dir:, threshold: 3.0)
    FileUtils.mkdir_p(output_dir)
    data = load_csv(data_path)
    
    results = {
      missing_values: detect_missing_values(data),
      outliers: detect_outliers(data, threshold: threshold),
      constant_features: detect_constant_features(data),
      high_correlation: detect_high_correlation(data),
      distribution_stats: calculate_distribution_stats(data),
      feature_quality_score: calculate_feature_quality(data)
    }

    report_path = File.join(output_dir, 'feature_debug_report.html')
    generate_feature_debug_report(results, report_path)
    
    csv_path = File.join(output_dir, 'feature_debug.csv')
    save_debug_results(results, csv_path)

    @logger.info "✓ Feature debugging complete: #{report_path}"
    results
  end

  # Compare multiple models and explain differences
  def compare_models(models_config:, test_data:, output_dir:)
    FileUtils.mkdir_p(output_dir)
    
    model_results = models_config.map do |config|
      {
        name: config[:name],
        predictions: load_predictions(config[:predictions_path]),
        shap_values: config[:shap_values_path] ? load_shap_values(config[:shap_values_path]) : nil,
        metrics: calculate_metrics(config[:predictions_path], test_data)
      }
    end

    comparison = {
      prediction_agreement: calculate_prediction_agreement(model_results),
      feature_importance_differences: compare_feature_importance(model_results),
      complementary_analysis: find_complementary_models(model_results),
      ensemble_potential: estimate_ensemble_gain(model_results)
    }

    report_path = File.join(output_dir, 'model_comparison_report.html')
    generate_comparison_report(comparison, model_results, report_path)

    comparison
  end

  # Explain a single prediction in detail
  def explain_single_prediction(model_path:, features:, output_path:, model_type: 'xgboost')
    # Ensure output directory exists
    FileUtils.mkdir_p(File.dirname(output_path))
    
    python_script = generate_single_prediction_script(
      model_path: model_path,
      features: features,
      output_path: output_path,
      model_type: model_type
    )

    script_path = "#{output_path}_script.py"
    File.write(script_path, python_script)

    result = system('python', script_path)
    unless result
      raise "Single prediction explanation failed"
    end

    JSON.parse(File.read("#{output_path}.json"))
  end

  private

  def validate_inputs(model_path, data_path)
    raise "Model file not found: #{model_path}" unless File.exist?(model_path)
    raise "Data file not found: #{data_path}" unless File.exist?(data_path)
  end

  def generate_shap_script(model_path:, data_path:, output_dir:, model_type:, top_n:)
    # Safely escape all paths and values for Python
    model_path_json = model_path.to_json
    data_path_json = data_path.to_json
    output_dir_json = output_dir.to_json
    model_type_json = model_type.to_json
    
    <<~PYTHON
      import pandas as pd
      import numpy as np
      import shap
      import matplotlib.pyplot as plt
      import joblib
      import json
      import pickle
      from pathlib import Path

      # Load paths from JSON-encoded Ruby strings
      model_path = #{model_path_json}
      data_path = #{data_path_json}
      output_dir = #{output_dir_json}
      model_type = #{model_type_json}

      # Load model
      model = joblib.load(model_path)
      
      # Load data
      data = pd.read_csv(data_path)
      
      # Separate features (remove target if present)
      target_cols = ['target', 'actual', 'label', 'y', 'win', 'result']
      X = data.drop(columns=[col for col in target_cols if col in data.columns], errors='ignore')
      
      # Create SHAP explainer
      print("Creating SHAP explainer...")
      if model_type in ['xgboost', 'lightgbm', 'catboost']:
          explainer = shap.TreeExplainer(model)
      elif model_type == 'linear':
          explainer = shap.LinearExplainer(model, X)
      else:
          explainer = shap.KernelExplainer(model.predict, shap.sample(X, 100))
      
      # Calculate SHAP values
      print("Calculating SHAP values...")
      shap_values = explainer.shap_values(X)
      
      # Handle multiclass outputs (shap_values will be a list)
      if isinstance(shap_values, list):
          print("Detected multiclass output, averaging across classes...")
          shap_values = np.mean(shap_values, axis=0)
      
      # Save SHAP values
      shap_df = pd.DataFrame(shap_values, columns=X.columns)
      shap_df.to_csv(output_dir + '/shap_values.csv', index=False)
      
      # Summary plot (bar chart of mean absolute SHAP values)
      plt.figure(figsize=(10, 8))
      shap.summary_plot(shap_values, X, plot_type="bar", show=False, max_display=#{top_n})
      plt.tight_layout()
      plt.savefig(output_dir + '/shap_importance.png', dpi=150, bbox_inches='tight')
      plt.close()
      
      # Detailed summary plot (beeswarm)
      plt.figure(figsize=(10, 8))
      shap.summary_plot(shap_values, X, show=False, max_display=#{top_n})
      plt.tight_layout()
      plt.savefig(output_dir + '/shap_summary.png', dpi=150, bbox_inches='tight')
      plt.close()
      
      # Feature importance data
      feature_importance = np.abs(shap_values).mean(axis=0)
      importance_df = pd.DataFrame({
          'feature': X.columns,
          'importance': feature_importance
      }).sort_values('importance', ascending=False)
      importance_df.to_csv(output_dir + '/feature_importance.csv', index=False)
      
      # Dependence plots for top features
      Path(output_dir + '/dependence').mkdir(exist_ok=True)
      top_features = importance_df.head(10)['feature'].tolist()
      
      for feature in top_features:
          plt.figure(figsize=(8, 6))
          shap.dependence_plot(feature, shap_values, X, show=False)
          plt.tight_layout()
          safe_name = feature.replace('/', '_').replace(' ', '_')
          plt.savefig(output_dir + f'/dependence/{safe_name}.png', dpi=150, bbox_inches='tight')
          plt.close()
      
      # Generate HTML report
      html_content = f'''
      <!DOCTYPE html>
      <html>
      <head>
          <title>SHAP Explainability Report</title>
          <style>
              body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
              h1 {{ color: #2c3e50; }}
              h2 {{ color: #34495e; margin-top: 30px; }}
              .container {{ background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
              img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px; margin: 20px 0; }}
              table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
              th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
              th {{ background-color: #3498db; color: white; }}
              tr:nth-child(even) {{ background-color: #f2f2f2; }}
              .metric {{ display: inline-block; margin: 10px 20px 10px 0; }}
              .metric-value {{ font-size: 24px; font-weight: bold; color: #2980b9; }}
              .metric-label {{ font-size: 14px; color: #7f8c8d; }}
          </style>
      </head>
      <body>
          <div class="container">
              <h1>🔍 SHAP Explainability Report</h1>
              <p><strong>Model:</strong> {model_path}</p>
              <p><strong>Data:</strong> {data_path}</p>
              <p><strong>Samples:</strong> {len(X)}</p>
              <p><strong>Features:</strong> {len(X.columns)}</p>
              
              <h2>Feature Importance (Mean |SHAP|)</h2>
              <img src="shap_importance.png" alt="Feature Importance">
              
              <h2>Feature Impact Summary</h2>
              <img src="shap_summary.png" alt="SHAP Summary">
              <p><em>Red = high feature value, Blue = low feature value</em></p>
              
              <h2>Top {len(top_features)} Features</h2>
              <table>
                  <tr>
                      <th>Rank</th>
                      <th>Feature</th>
                      <th>Mean |SHAP|</th>
                      <th>% of Total</th>
                  </tr>
      '''
      
      total_importance = importance_df['importance'].sum()
      for idx, row in importance_df.head(20).iterrows():
          pct = (row['importance'] / total_importance * 100)
          html_content += f'''
                  <tr>
                      <td>{idx + 1}</td>
                      <td>{row['feature']}</td>
                      <td>{row['importance']:.4f}</td>
                      <td>{pct:.2f}%</td>
                  </tr>
          '''
      
      html_content += '''
              </table>
              
              <h2>Feature Dependence Plots</h2>
              <p>Shows how each feature's value affects predictions:</p>
      '''
      
      for feature in top_features:
          safe_name = feature.replace('/', '_').replace(' ', '_')
          html_content += f'<h3>{feature}</h3><img src="dependence/{safe_name}.png" alt="{feature}">'
      
      html_content += '''
          </div>
      </body>
      </html>
      '''
      
      with open(output_dir + '/shap_report.html', 'w') as f:
          f.write(html_content)
      
      print("✓ SHAP analysis complete!")
      print(f"  - Summary plot: {output_dir}/shap_summary.png")
      print(f"  - Importance plot: {output_dir}/shap_importance.png")
      print(f"  - HTML report: {output_dir}/shap_report.html")
    PYTHON
  end

  def generate_single_prediction_script(model_path:, features:, output_path:, model_type:)
    # Safely escape all values for Python
    model_path_json = model_path.to_json
    output_path_json = output_path.to_json
    model_type_json = model_type.to_json
    features_json = features.to_json
    
    <<~PYTHON
      import pandas as pd
      import numpy as np
      import shap
      import joblib
      import json

      # Load paths and values from JSON-encoded Ruby strings
      model_path = #{model_path_json}
      output_path = #{output_path_json}
      model_type = #{model_type_json}
      features = #{features_json}

      # Load model
      model = joblib.load(model_path)
      
      # Load features
      X = pd.DataFrame([features])
      
      # Make prediction
      prediction = model.predict(X)[0]
      
      # Create SHAP explainer
      if model_type in ['xgboost', 'lightgbm', 'catboost']:
          explainer = shap.TreeExplainer(model)
      else:
          explainer = shap.KernelExplainer(model.predict, X)
      
      # Calculate SHAP values
      shap_values = explainer.shap_values(X)
      
      # Handle multiclass outputs
      if isinstance(shap_values, list):
          shap_values = np.mean(shap_values, axis=0)
      
      # Prepare result
      result = {
          'prediction': float(prediction),
          'base_value': float(explainer.expected_value) if hasattr(explainer, 'expected_value') else None,
          'feature_contributions': {
              feature: float(shap_val)
              for feature, shap_val in zip(X.columns, shap_values[0])
          },
          'top_positive': sorted(
              [(f, float(s)) for f, s in zip(X.columns, shap_values[0]) if s > 0],
              key=lambda x: x[1],
              reverse=True
          )[:5],
          'top_negative': sorted(
              [(f, float(s)) for f, s in zip(X.columns, shap_values[0]) if s < 0],
              key=lambda x: x[1]
          )[:5]
      }
      
      with open(output_path + '.json', 'w') as f:
          json.dump(result, f, indent=2)
      
      print("✓ Single prediction explained!")
    PYTHON
  end

  def load_csv(path)
    CSV.read(path, headers: true, converters: :numeric)
  end

  def calculate_errors(predictions, actuals, features)
    predictions.map.with_index do |pred_row, idx|
      actual_row = actuals[idx]
      feature_row = features[idx]
      pred_val = pred_row['prediction'].to_f
      actual_val = actual_row['actual'].to_f
      
      {
        index: idx,
        predicted: pred_val,
        actual: actual_val,
        error: pred_val - actual_val,
        absolute_error: (pred_val - actual_val).abs,
        squared_error: (pred_val - actual_val) ** 2,
        features: feature_row.to_h
      }
    end
  end

  def overall_statistics(errors)
    mae = errors.sum { |e| e[:absolute_error] } / errors.size
    rmse = Math.sqrt(errors.sum { |e| e[:squared_error] } / errors.size)
    mean_error = errors.sum { |e| e[:error] } / errors.size
    
    {
      count: errors.size,
      mae: mae,
      rmse: rmse,
      mean_error: mean_error,
      std_error: standard_deviation(errors.map { |e| e[:error] }),
      max_error: errors.max_by { |e| e[:absolute_error] }[:absolute_error],
      min_error: errors.min_by { |e| e[:absolute_error] }[:absolute_error]
    }
  end

  def group_by_error_magnitude(errors, features)
    bins = [
      { name: 'excellent', range: 0..0.1, errors: [] },
      { name: 'good', range: 0.1..0.5, errors: [] },
      { name: 'fair', range: 0.5..1.0, errors: [] },
      { name: 'poor', range: 1.0..2.0, errors: [] },
      { name: 'very_poor', range: 2.0..Float::INFINITY, errors: [] }
    ]

    errors.each do |error|
      bin = bins.find { |b| b[:range].cover?(error[:absolute_error]) }
      bin[:errors] << error if bin
    end

    bins.map do |bin|
      {
        name: bin[:name],
        count: bin[:errors].size,
        percentage: (bin[:errors].size.to_f / errors.size * 100).round(2),
        avg_error: bin[:errors].empty? ? 0 : bin[:errors].sum { |e| e[:absolute_error] } / bin[:errors].size
      }
    end
  end

  def analyze_feature_ranges(errors, features)
    # Analyze if errors are concentrated in certain feature value ranges
    feature_columns = features[0].headers - ['prediction', 'actual', 'target', 'label']
    
    feature_columns.first(10).map do |feature|
      values = features.map { |row| row[feature].to_f }
      quartile_bins = calculate_quartiles(values)
      
      {
        feature: feature,
        quartile_errors: quartile_bins.map.with_index do |bin, idx|
          quartile_errors = errors.select { |e| 
            val = e[:features][feature].to_f
            bin[:range].cover?(val)
          }
          {
            quartile: idx + 1,
            count: quartile_errors.size,
            avg_error: quartile_errors.empty? ? 0 : quartile_errors.sum { |e| e[:absolute_error] } / quartile_errors.size
          }
        end
      }
    end
  end

  def identify_worst_predictions(errors, features, top_n:)
    errors.sort_by { |e| -e[:absolute_error] }.first(top_n).map do |error|
      {
        index: error[:index],
        predicted: error[:predicted].round(3),
        actual: error[:actual].round(3),
        error: error[:error].round(3),
        absolute_error: error[:absolute_error].round(3)
      }
    end
  end

  def detect_systematic_bias(errors, features)
    # Check for bias patterns
    mean_error = errors.sum { |e| e[:error] } / errors.size
    
    over_count = errors.count { |e| e[:error] > 0 }
    under_count = errors.count { |e| e[:error] < 0 }
    exact_count = errors.count { |e| e[:error] == 0 }
    
    {
      overall_bias: mean_error,
      overestimation_rate: (over_count.to_f / errors.size * 100).round(2),
      underestimation_rate: (under_count.to_f / errors.size * 100).round(2),
      exact_predictions: exact_count,
      exact_rate: (exact_count.to_f / errors.size * 100).round(2),
      significant_bias: mean_error.abs > 0.1
    }
  end

  def detect_missing_values(data)
    data.headers.map do |header|
      missing_count = data.count { |row| row[header].nil? || row[header].to_s.strip.empty? }
      {
        feature: header,
        missing_count: missing_count,
        missing_percentage: (missing_count.to_f / data.size * 100).round(2)
      }
    end.select { |result| result[:missing_count] > 0 }
  end

  def detect_outliers(data, threshold:)
    numeric_columns = data.headers.select do |header|
      data.first[header].to_s.match?(/^-?\d+\.?\d*$/)
    end

    numeric_columns.map do |col|
      values = data.map { |row| row[col].to_f }
      mean = values.sum / values.size
      std = standard_deviation(values)
      
      # Skip if no variation (std == 0)
      next nil if std.zero?
      
      outliers = values.select { |v| ((v - mean).abs / std) > threshold }
      
      {
        feature: col,
        outlier_count: outliers.size,
        outlier_percentage: (outliers.size.to_f / values.size * 100).round(2),
        threshold_sigmas: threshold
      }
    end.compact.select { |result| result[:outlier_count] > 0 }
  end

  def detect_constant_features(data)
    data.headers.select do |header|
      unique_values = data.map { |row| row[header] }.uniq
      unique_values.size <= 1
    end.map { |feature| { feature: feature, unique_values: 1 } }
  end

  def detect_high_correlation(data)
    # Simple pairwise correlation check (first 20 numeric features)
    numeric_cols = data.headers.select do |header|
      data.first[header].to_s.match?(/^-?\d+\.?\d*$/)
    end.first(20)

    high_correlations = []
    
    numeric_cols.combination(2).each do |col1, col2|
      values1 = data.map { |row| row[col1].to_f }
      values2 = data.map { |row| row[col2].to_f }
      
      correlation = calculate_correlation(values1, values2)
      
      if correlation.abs > 0.9
        high_correlations << {
          feature1: col1,
          feature2: col2,
          correlation: correlation.round(4)
        }
      end
    end

    high_correlations
  end

  def calculate_distribution_stats(data)
    numeric_cols = data.headers.select do |header|
      data.first[header].to_s.match?(/^-?\d+\.?\d*$/)
    end.first(20)

    numeric_cols.map do |col|
      values = data.map { |row| row[col].to_f }
      sorted = values.sort
      
      {
        feature: col,
        mean: (values.sum / values.size).round(4),
        median: sorted[sorted.size / 2].round(4),
        std: standard_deviation(values).round(4),
        min: sorted.first.round(4),
        max: sorted.last.round(4),
        q1: sorted[(sorted.size * 0.25).to_i].round(4),
        q3: sorted[(sorted.size * 0.75).to_i].round(4)
      }
    end
  end

  def calculate_feature_quality(data)
    # Simple quality score based on completeness and variance
    data.headers.map do |header|
      missing_count = data.count { |row| row[header].nil? || row[header].to_s.strip.empty? }
      completeness = 1.0 - (missing_count.to_f / data.size)
      
      values = data.map { |row| row[header].to_s }
      uniqueness = values.uniq.size.to_f / values.size
      
      score = (completeness * 0.7 + uniqueness * 0.3) * 100
      
      {
        feature: header,
        quality_score: score.round(2),
        completeness: (completeness * 100).round(2),
        uniqueness: (uniqueness * 100).round(2)
      }
    end.sort_by { |r| -r[:quality_score] }
  end

  def standard_deviation(values)
    mean = values.sum / values.size
    variance = values.sum { |v| (v - mean) ** 2 } / values.size
    Math.sqrt(variance)
  end

  def calculate_correlation(x, y)
    n = x.size
    sum_x = x.sum
    sum_y = y.sum
    sum_xy = x.zip(y).sum { |a, b| a * b }
    sum_x2 = x.sum { |a| a ** 2 }
    sum_y2 = y.sum { |a| a ** 2 }
    
    numerator = n * sum_xy - sum_x * sum_y
    denominator = Math.sqrt((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2))
    
    return 0 if denominator.zero?
    numerator / denominator
  end

  def calculate_quartiles(values)
    sorted = values.sort
    n = sorted.size
    
    q1_val = sorted[(n * 0.25).to_i]
    q2_val = sorted[(n * 0.50).to_i]
    q3_val = sorted[(n * 0.75).to_i]
    
    # Return 4 quartile ranges with exclusive lower bounds to avoid overlaps
    [
      { range: -Float::INFINITY..q1_val, q1: q1_val },
      { range: q1_val...q2_val, q2: q2_val },  # exclusive lower bound
      { range: q2_val...q3_val, q3: q3_val },  # exclusive lower bound
      { range: q3_val..Float::INFINITY, q3: q3_val }
    ]
  end

  def save_error_analysis(analysis, output_path)
    CSV.open(output_path, 'w') do |csv|
      csv << ['Metric', 'Value']
      analysis[:overall].each { |k, v| csv << [k.to_s, v] }
    end
  end

  def save_debug_results(results, csv_path)
    CSV.open(csv_path, 'w') do |csv|
      csv << ['Feature', 'Quality Score', 'Completeness %', 'Uniqueness %']
      results[:feature_quality_score].each do |row|
        csv << [row[:feature], row[:quality_score], row[:completeness], row[:uniqueness]]
      end
    end
  end

  def generate_error_report(analysis, base_path)
    html_path = base_path.sub('.csv', '_report.html')
    
    html_content = <<~HTML
      <!DOCTYPE html>
      <html>
      <head>
          <title>Error Analysis Report</title>
          <style>
              body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
              .container { background: white; padding: 30px; border-radius: 8px; }
              h1 { color: #e74c3c; }
              h2 { color: #34495e; margin-top: 30px; }
              table { border-collapse: collapse; width: 100%; margin: 20px 0; }
              th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
              th { background-color: #e74c3c; color: white; }
              .metric { font-size: 18px; margin: 10px 0; }
              .good { color: #27ae60; }
              .bad { color: #e74c3c; }
          </style>
      </head>
      <body>
          <div class="container">
              <h1>🐛 Error Analysis Report</h1>
              
              <h2>Overall Statistics</h2>
              <div class="metric">MAE: #{analysis[:overall][:mae].round(4)}</div>
              <div class="metric">RMSE: #{analysis[:overall][:rmse].round(4)}</div>
              <div class="metric">Mean Error: #{analysis[:overall][:mean_error].round(4)}</div>
              
              <h2>Error Distribution</h2>
              <table>
                  <tr><th>Category</th><th>Count</th><th>Percentage</th><th>Avg Error</th></tr>
    HTML

    analysis[:by_magnitude].each do |bin|
      html_content += "<tr><td>#{bin[:name]}</td><td>#{bin[:count]}</td><td>#{bin[:percentage]}%</td><td>#{bin[:avg_error].round(4)}</td></tr>\n"
    end

    html_content += <<~HTML
              </table>
              
              <h2>Systematic Bias Analysis</h2>
              <p>Overall Bias: #{analysis[:systematic_bias][:overall_bias].round(4)}</p>
              <p>Overestimation Rate: #{analysis[:systematic_bias][:overestimation_rate]}%</p>
              <p>Underestimation Rate: #{analysis[:systematic_bias][:underestimation_rate]}%</p>
              
              <h2>Worst Predictions (Top 20)</h2>
              <table>
                  <tr><th>Index</th><th>Predicted</th><th>Actual</th><th>Error</th><th>|Error|</th></tr>
    HTML

    analysis[:worst_predictions].each do |pred|
      html_content += "<tr><td>#{pred[:index]}</td><td>#{pred[:predicted]}</td><td>#{pred[:actual]}</td><td>#{pred[:error]}</td><td>#{pred[:absolute_error]}</td></tr>\n"
    end

    html_content += "</table></div></body></html>"
    
    File.write(html_path, html_content)
    @logger.info "✓ Error analysis report: #{html_path}"
  end

  def generate_feature_debug_report(results, report_path)
    html_content = <<~HTML
      <!DOCTYPE html>
      <html>
      <head>
          <title>Feature Debug Report</title>
          <style>
              body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
              .container { background: white; padding: 30px; border-radius: 8px; }
              h1 { color: #9b59b6; }
              h2 { color: #34495e; margin-top: 30px; }
              table { border-collapse: collapse; width: 100%; margin: 20px 0; }
              th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
              th { background-color: #9b59b6; color: white; }
              .warning { background-color: #fff3cd; padding: 10px; border-left: 4px solid #ffc107; margin: 10px 0; }
              .success { background-color: #d4edda; padding: 10px; border-left: 4px solid #28a745; margin: 10px 0; }
          </style>
      </head>
      <body>
          <div class="container">
              <h1>🔧 Feature Debug Report</h1>
              
              <h2>Data Quality Summary</h2>
    HTML

    if results[:missing_values].empty?
      html_content += '<div class="success">✓ No missing values detected</div>'
    else
      html_content += '<div class="warning">⚠ Missing values detected</div>'
      html_content += '<table><tr><th>Feature</th><th>Missing Count</th><th>Missing %</th></tr>'
      results[:missing_values].each do |mv|
        html_content += "<tr><td>#{mv[:feature]}</td><td>#{mv[:missing_count]}</td><td>#{mv[:missing_percentage]}%</td></tr>"
      end
      html_content += '</table>'
    end

    if results[:constant_features].empty?
      html_content += '<div class="success">✓ No constant features detected</div>'
    else
      html_content += '<div class="warning">⚠ Constant features detected (should be removed)</div>'
      html_content += '<ul>'
      results[:constant_features].each { |cf| html_content += "<li>#{cf[:feature]}</li>" }
      html_content += '</ul>'
    end

    if results[:outliers].empty?
      html_content += '<div class="success">✓ No extreme outliers detected</div>'
    else
      html_content += '<h2>Outliers (>3σ)</h2><table><tr><th>Feature</th><th>Count</th><th>Percentage</th></tr>'
      results[:outliers].each do |outlier|
        html_content += "<tr><td>#{outlier[:feature]}</td><td>#{outlier[:outlier_count]}</td><td>#{outlier[:outlier_percentage]}%</td></tr>"
      end
      html_content += '</table>'
    end

    if results[:high_correlation].any?
      html_content += '<h2>High Correlations (|r| > 0.9)</h2><table><tr><th>Feature 1</th><th>Feature 2</th><th>Correlation</th></tr>'
      results[:high_correlation].each do |corr|
        html_content += "<tr><td>#{corr[:feature1]}</td><td>#{corr[:feature2]}</td><td>#{corr[:correlation]}</td></tr>"
      end
      html_content += '</table>'
    end

    html_content += '<h2>Feature Quality Scores</h2><table><tr><th>Feature</th><th>Score</th><th>Completeness</th><th>Uniqueness</th></tr>'
    results[:feature_quality_score].first(20).each do |fq|
      html_content += "<tr><td>#{fq[:feature]}</td><td>#{fq[:quality_score]}</td><td>#{fq[:completeness]}%</td><td>#{fq[:uniqueness]}%</td></tr>"
    end

    html_content += "</table></div></body></html>"
    
    File.write(report_path, html_content)
  end

  def generate_comparison_report(comparison, model_results, report_path)
    # Placeholder for model comparison report
    File.write(report_path, "<html><body><h1>Model Comparison Report</h1><p>Coming soon...</p></body></html>")
  end

  def load_predictions(path)
    CSV.read(path, headers: true)
  end

  def load_shap_values(path)
    CSV.read(path, headers: true)
  end

  def calculate_metrics(predictions_path, test_data)
    {} # Placeholder
  end

  def calculate_prediction_agreement(model_results)
    {} # Placeholder
  end

  def compare_feature_importance(model_results)
    {} # Placeholder
  end

  def find_complementary_models(model_results)
    {} # Placeholder
  end

  def estimate_ensemble_gain(model_results)
    {} # Placeholder
  end
end
