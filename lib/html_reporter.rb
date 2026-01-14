require 'csv'
require 'json'
require 'logger'

class HTMLReporter
  attr_reader :logger

  def initialize(logger = Logger.new(STDOUT))
    @logger = logger
  end

  # Generate simple HTML report with tables only (no charts)
  def generate_diagnostic_report(csv_file, output_file = nil)
    output_file ||= csv_file.gsub('.csv', '_report.html')
    
    logger.info "Generating HTML diagnostic report for #{csv_file}"
    
    # Load data
    data = CSV.read(csv_file, headers: true)
    headers = data.headers
    
    # Analyze each column
    column_analyses = headers.map do |col|
      analyze_column(data, col)
    end
    
    # Generate HTML
    html = build_html(csv_file, data.size, column_analyses)
    
    # Write to file
    File.write(output_file, html)
    logger.info "HTML report saved to #{output_file}"
    
    output_file
  end

  # Generate HTML report for hyperparameter tracking/results
  def generate_hyperparam_report(csv_file, output_file = nil)
    output_file ||= csv_file.gsub('.csv', '_report.html')
    
    logger.info "Generating hyperparameter report for #{csv_file}"
    
    # Load data
    data = CSV.read(csv_file, headers: true)
    headers = data.headers
    
    # Identify columns
    param_cols = headers.reject { |h| ['experiment_id', 'rmse', 'mae', 'r2', 'notes', 'timestamp'].include?(h) }
    tracking_cols = headers & ['experiment_id', 'rmse', 'mae', 'r2', 'notes', 'timestamp']
    
    # Find best configs (if results exist)
    completed_rows = data.select { |row| !row['rmse'].nil? && !row['rmse'].to_s.strip.empty? }
    best_by_rmse = completed_rows.min_by { |row| row['rmse'].to_f } if completed_rows.any?
    best_by_r2 = completed_rows.max_by { |row| row['r2'].to_f } if completed_rows.any?
    
    # Calculate completion stats
    completed_count = completed_rows.size
    pending_count = data.size - completed_count
    
    # Generate HTML
    html = build_hyperparam_html(
      csv_file, 
      data, 
      param_cols, 
      tracking_cols, 
      best_by_rmse, 
      best_by_r2,
      completed_count,
      pending_count
    )
    
    # Write to file
    File.write(output_file, html)
    logger.info "Hyperparameter report saved to #{output_file}"
    
    output_file
  end

  def build_hyperparam_html(csv_file, data, param_cols, tracking_cols, best_rmse, best_r2, completed, pending)
    model_name = File.basename(csv_file, '.csv')
    
    <<~HTML
      <!DOCTYPE html>
      <html lang="en">
      <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Hyperparameter Report - #{model_name}</title>
        <style>
          * { margin: 0; padding: 0; box-sizing: border-box; }
          body {
            font-family: 'Courier New', monospace;
            background: #1e1e1e;
            color: #d4d4d4;
            padding: 20px;
            line-height: 1.6;
          }
          .container { max-width: 1400px; margin: 0 auto; }
          .header {
            background: #252526;
            padding: 20px;
            margin-bottom: 20px;
            border-left: 4px solid #007acc;
          }
          .header h1 { 
            font-size: 1.8em; 
            margin-bottom: 5px;
            color: #4ec9b0;
          }
          .header p { 
            color: #858585;
            font-size: 0.9em;
          }
          .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
          }
          .stat-card {
            background: #252526;
            padding: 15px;
            border-left: 3px solid #007acc;
          }
          .stat-label {
            color: #858585;
            font-size: 0.85em;
            margin-bottom: 5px;
          }
          .stat-value {
            color: #4ec9b0;
            font-size: 1.5em;
            font-weight: bold;
          }
          table {
            width: 100%;
            border-collapse: collapse;
            background: #252526;
            margin-bottom: 20px;
            font-size: 0.9em;
          }
          th {
            background: #2d2d30;
            color: #4ec9b0;
            padding: 10px 8px;
            text-align: left;
            border-bottom: 2px solid #007acc;
            font-weight: normal;
            position: sticky;
            top: 0;
          }
          td {
            padding: 8px;
            border-bottom: 1px solid #3e3e42;
          }
          tr:hover {
            background: #2a2d2e;
          }
          .section {
            margin-bottom: 30px;
          }
          .section-title {
            color: #4ec9b0;
            font-size: 1.4em;
            margin-bottom: 15px;
            padding-bottom: 8px;
            border-bottom: 2px solid #3e3e42;
          }
          .best-config {
            background: #1e3a1e !important;
            border-left: 3px solid #6a9955;
          }
          .pending {
            color: #858585;
          }
          .completed {
            color: #4ec9b0;
          }
          .badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 3px;
            font-size: 0.8em;
          }
          .badge-best {
            background: #1e3a1e;
            color: #6a9955;
          }
          .badge-pending {
            background: #332a1e;
            color: #d7ba7d;
          }
          .badge-completed {
            background: #1e3a5f;
            color: #4fc3f7;
          }
          .metric-good {
            color: #6a9955;
            font-weight: bold;
          }
          .metric-bad {
            color: #f48771;
          }
          .scrollable {
            max-height: 600px;
            overflow-y: auto;
          }
        </style>
      </head>
      <body>
        <div class="container">
          <div class="header">
            <h1>HYPERPARAMETER TRACKING</h1>
            <p>Model: #{model_name}</p>
            <p>File: #{File.basename(csv_file)}</p>
          </div>

          <div class="stats-grid">
            <div class="stat-card">
              <div class="stat-label">Total Configs</div>
              <div class="stat-value">#{data.size}</div>
            </div>
            <div class="stat-card">
              <div class="stat-label">Completed</div>
              <div class="stat-value completed">#{completed}</div>
            </div>
            <div class="stat-card">
              <div class="stat-label">Pending</div>
              <div class="stat-value pending">#{pending}</div>
            </div>
            <div class="stat-card">
              <div class="stat-label">Progress</div>
              <div class="stat-value">#{((completed.to_f / data.size * 100).round(1))}%</div>
            </div>
          </div>

          #{best_rmse ? generate_best_config_section(best_rmse, best_r2, param_cols) : ''}

          <div class="section">
            <h2 class="section-title">ALL CONFIGURATIONS (#{data.size})</h2>
            <div class="scrollable">
              <table>
                <thead>
                  <tr>
                    <th>ID</th>
                    #{param_cols.map { |col| "<th>#{col}</th>" }.join}
                    <th>RMSE</th>
                    <th>MAE</th>
                    <th>R²</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  #{data.map { |row| generate_config_row(row, param_cols, best_rmse) }.join}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </body>
      </html>
    HTML
  end

  def generate_best_config_section(best_rmse, best_r2, param_cols)
    <<~HTML
      <div class="section">
        <h2 class="section-title">🏆 BEST CONFIGURATION (by RMSE)</h2>
        <table>
          <tr>
            <td><strong>Experiment ID</strong></td>
            <td>#{best_rmse['experiment_id']}</td>
          </tr>
          <tr>
            <td><strong>RMSE</strong></td>
            <td class="metric-good">#{best_rmse['rmse']}</td>
          </tr>
          <tr>
            <td><strong>MAE</strong></td>
            <td>#{best_rmse['mae']}</td>
          </tr>
          <tr>
            <td><strong>R²</strong></td>
            <td>#{best_rmse['r2']}</td>
          </tr>
          #{param_cols.map { |col| "<tr><td><strong>#{col}</strong></td><td>#{best_rmse[col]}</td></tr>" }.join("\n")}
        </table>
      </div>
    HTML
  end

  def generate_config_row(row, param_cols, best_config)
    is_best = best_config && row['experiment_id'] == best_config['experiment_id']
    has_results = !row['rmse'].nil? && !row['rmse'].to_s.strip.empty?
    
    row_class = is_best ? 'best-config' : ''
    status_badge = if is_best
      '<span class="badge badge-best">⭐ BEST</span>'
    elsif has_results
      '<span class="badge badge-completed">✓ Done</span>'
    else
      '<span class="badge badge-pending">⏳ Pending</span>'
    end
    
    <<~HTML
      <tr class="#{row_class}">
        <td>#{row['experiment_id']}</td>
        #{param_cols.map { |col| "<td>#{row[col]}</td>" }.join}
        <td class="#{has_results && row['rmse'].to_f < 2.0 ? 'metric-good' : ''}">#{row['rmse'] || '-'}</td>
        <td>#{row['mae'] || '-'}</td>
        <td>#{row['r2'] || '-'}</td>
        <td>#{status_badge}</td>
      </tr>
    HTML
  end

  private

  def analyze_column(data, column)
    values = data[column].to_a
    non_empty = values.reject { |v| v.nil? || v.to_s.strip.empty? }
    
    analysis = {
      name: column,
      total: values.size,
      missing: values.size - non_empty.size,
      missing_pct: ((values.size - non_empty.size) / values.size.to_f * 100).round(1)
    }
    
    # Try numeric analysis
    numeric_values = non_empty.map { |v| Float(v) rescue nil }.compact
    
    if numeric_values.size > non_empty.size * 0.8 # 80% numeric = numeric column
      analysis[:type] = 'numeric'
      analysis[:min] = numeric_values.min.round(2)
      analysis[:max] = numeric_values.max.round(2)
      analysis[:mean] = (numeric_values.sum / numeric_values.size).round(2)
      analysis[:median] = numeric_values.sort[numeric_values.size / 2].round(2)
      analysis[:unique_count] = numeric_values.uniq.size
      
      # Calculate quartiles
      sorted = numeric_values.sort
      q1 = sorted[sorted.size / 4].round(2)
      q3 = sorted[(sorted.size * 3) / 4].round(2)
      analysis[:q1] = q1
      analysis[:q3] = q3
      
      # Outliers
      iqr = q3 - q1
      lower = q1 - 1.5 * iqr
      upper = q3 + 1.5 * iqr
      outliers = numeric_values.select { |v| v < lower || v > upper }
      analysis[:outliers] = outliers.size
    else
      analysis[:type] = 'text'
      analysis[:unique_count] = non_empty.uniq.size
      analysis[:cardinality] = (analysis[:unique_count] / non_empty.size.to_f * 100).round(1)
      
      # Top values
      value_counts = non_empty.group_by(&:itself).transform_values(&:size)
      analysis[:top_values] = value_counts.sort_by { |k, v| -v }.first(10).to_h
      
      # Text length stats
      lengths = non_empty.map(&:length)
      analysis[:min_length] = lengths.min
      analysis[:max_length] = lengths.max
      analysis[:avg_length] = (lengths.sum / lengths.size.to_f).round(1)
    end
    
    analysis
  end

  def build_html(csv_file, row_count, columns)
    numeric_cols = columns.select { |c| c[:type] == 'numeric' }
    text_cols = columns.select { |c| c[:type] == 'text' }
    
    <<~HTML
      <!DOCTYPE html>
      <html lang="en">
      <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>CSV Diagnostic Report - #{File.basename(csv_file)}</title>
        <style>
          * { margin: 0; padding: 0; box-sizing: border-box; }
          body {
            font-family: 'Courier New', monospace;
            background: #1e1e1e;
            color: #d4d4d4;
            padding: 20px;
            line-height: 1.6;
          }
          .container { max-width: 1200px; margin: 0 auto; }
          .header {
            background: #252526;
            padding: 20px;
            margin-bottom: 20px;
            border-left: 4px solid #007acc;
          }
          .header h1 { 
            font-size: 1.8em; 
            margin-bottom: 5px;
            color: #4ec9b0;
          }
          .header p { 
            color: #858585;
            font-size: 0.9em;
          }
          table {
            width: 100%;
            border-collapse: collapse;
            background: #252526;
            margin-bottom: 20px;
          }
          th {
            background: #2d2d30;
            color: #4ec9b0;
            padding: 12px;
            text-align: left;
            border-bottom: 2px solid #007acc;
            font-weight: normal;
          }
          td {
            padding: 10px 12px;
            border-bottom: 1px solid #3e3e42;
          }
          tr:hover {
            background: #2a2d2e;
          }
          .section {
            margin-bottom: 30px;
          }
          .section-title {
            color: #4ec9b0;
            font-size: 1.4em;
            margin-bottom: 15px;
            padding-bottom: 8px;
            border-bottom: 2px solid #3e3e42;
          }
          .type-badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 3px;
            font-size: 0.85em;
            font-weight: bold;
          }
          .type-numeric {
            background: #1e3a5f;
            color: #4fc3f7;
          }
          .type-text {
            background: #4a2a4f;
            color: #ce93d8;
          }
          .warning {
            background: #332a1e;
            color: #d7ba7d;
            padding: 8px 12px;
            margin-top: 5px;
            border-left: 3px solid #d7ba7d;
          }
          .good {
            color: #6a9955;
          }
          .bad {
            color: #f48771;
          }
          .neutral {
            color: #858585;
          }
          .stats-table {
            margin-bottom: 30px;
          }
          .value-list {
            list-style: none;
            padding: 0;
          }
          .value-list li {
            padding: 4px 0;
            color: #d4d4d4;
          }
          .value-list .count {
            color: #4fc3f7;
            float: right;
          }
        </style>
      </head>
      <body>
        <div class="container">
          <div class="header">
            <h1>CSV DIAGNOSTIC REPORT</h1>
            <p>File: #{File.basename(csv_file)}</p>
            <p>Rows: #{row_count} | Columns: #{columns.size} | Numeric: #{numeric_cols.size} | Text: #{text_cols.size}</p>
          </div>

          <div class="section">
            <h2 class="section-title">SUMMARY STATISTICS</h2>
            <table class="stats-table">
              <thead>
                <tr>
                  <th>Column</th>
                  <th>Type</th>
                  <th>Total</th>
                  <th>Missing</th>
                  <th>Missing %</th>
                  <th>Unique</th>
                  <th>Details</th>
                </tr>
              </thead>
              <tbody>
                #{columns.map { |col| generate_summary_row(col) }.join("\n")}
              </tbody>
            </table>
          </div>

          #{numeric_cols.any? ? generate_numeric_section(numeric_cols) : ''}
          #{text_cols.any? ? generate_text_section(text_cols) : ''}
        </div>
      </body>
      </html>
    HTML
  end

  def generate_summary_row(col)
    type_class = "type-#{col[:type]}"
    
    missing_class = if col[:missing_pct] > 30
      'bad'
    elsif col[:missing_pct] > 10
      'neutral'
    else
      'good'
    end
    
    details = if col[:type] == 'numeric'
      "Min: #{col[:min]} | Max: #{col[:max]} | Mean: #{col[:mean]}"
    else
      "Cardinality: #{col[:cardinality]}%"
    end
    
    warning = col[:missing_pct] > 30 ? "<div class='warning'>HIGH MISSING DATA</div>" : ""
    
    <<~HTML
      <tr>
        <td><strong>#{col[:name]}</strong></td>
        <td><span class="type-badge #{type_class}">#{col[:type].upcase}</span></td>
        <td>#{col[:total]}</td>
        <td class="#{missing_class}">#{col[:missing]}</td>
        <td class="#{missing_class}">#{col[:missing_pct]}%</td>
        <td>#{col[:unique_count]}</td>
        <td>#{details}#{warning}</td>
      </tr>
    HTML
  end

  def generate_numeric_section(numeric_cols)
    <<~HTML
      <div class="section">
        <h2 class="section-title">NUMERIC COLUMNS (#{numeric_cols.size})</h2>
        <table>
          <thead>
            <tr>
              <th>Column</th>
              <th>Min</th>
              <th>Q1</th>
              <th>Median</th>
              <th>Mean</th>
              <th>Q3</th>
              <th>Max</th>
              <th>Outliers</th>
            </tr>
          </thead>
          <tbody>
            #{numeric_cols.map { |col| generate_numeric_row(col) }.join("\n")}
          </tbody>
        </table>
      </div>
    HTML
  end

  def generate_numeric_row(col)
    outlier_class = col[:outliers] > 0 ? 'neutral' : 'good'
    
    <<~HTML
      <tr>
        <td><strong>#{col[:name]}</strong></td>
        <td>#{col[:min]}</td>
        <td>#{col[:q1]}</td>
        <td>#{col[:median]}</td>
        <td>#{col[:mean]}</td>
        <td>#{col[:q3]}</td>
        <td>#{col[:max]}</td>
        <td class="#{outlier_class}">#{col[:outliers]}</td>
      </tr>
    HTML
  end

  def generate_text_section(text_cols)
    html = <<~HTML
      <div class="section">
        <h2 class="section-title">TEXT COLUMNS (#{text_cols.size})</h2>
    HTML
    
    text_cols.each do |col|
      html += <<~HTML
        <div style="margin-bottom: 30px;">
          <table>
            <thead>
              <tr>
                <th colspan="2">#{col[:name]}</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>Unique Values</strong></td>
                <td>#{col[:unique_count]} (#{col[:cardinality]}% cardinality)</td>
              </tr>
              <tr>
                <td><strong>Text Length</strong></td>
                <td>Min: #{col[:min_length]} | Max: #{col[:max_length]} | Avg: #{col[:avg_length]}</td>
              </tr>
      HTML
      
      if col[:top_values]
        html += <<~HTML
              <tr>
                <td><strong>Top Values</strong></td>
                <td>
                  <ul class="value-list">
                    #{col[:top_values].map { |v, c| "<li>#{v} <span class='count'>#{c}</span></li>" }.join("\n")}
                  </ul>
                </td>
              </tr>
        HTML
      end
      
      html += <<~HTML
            </tbody>
          </table>
        </div>
      HTML
    end
    
    html += "</div>"
    html
  end
end
