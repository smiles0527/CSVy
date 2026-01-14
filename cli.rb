#!/usr/bin/env ruby

require 'thor'
require 'dotenv/load'
require_relative 'lib/csv_processor'
require_relative 'lib/data_preprocessor'
require_relative 'lib/data_validator'
require_relative 'lib/dataframe_handler'
require_relative 'lib/database_manager'
require_relative 'lib/csv_diagnostics'
require_relative 'lib/time_series_features'
require_relative 'lib/csv_io_handler'
require_relative 'lib/hyperparameter_manager'
require_relative 'lib/html_reporter'
require_relative 'lib/advanced_features'
require_relative 'lib/model_validator'
require_relative 'lib/ensemble_builder'
require_relative 'lib/model_tracker'
require_relative 'lib/neural_network_wrapper'

class CSVOrganizer < Thor
  desc "report FILE", "Generate HTML report with tables (no fancy charts)"
  option :output, aliases: :o, type: :string, desc: 'Output HTML file (default: FILE_report.html)'
  option :type, aliases: :t, type: :string, default: 'auto', desc: 'Report type: data, hyperparam, or auto'
  def report(file)
    unless File.exist?(file)
      puts "✗ File not found: #{file}"
      exit 1
    end
    
    reporter = HTMLReporter.new
    
    # Auto-detect report type based on file content
    report_type = options[:type]
    if report_type == 'auto'
      # Check if file has hyperparameter tracking columns
      headers = CSV.read(file, headers: true).headers
      has_hyperparam_cols = (headers & ['experiment_id', 'rmse', 'mae', 'r2']).size >= 2
      report_type = has_hyperparam_cols ? 'hyperparam' : 'data'
    end
    
    output_file = if report_type == 'hyperparam'
      reporter.generate_hyperparam_report(file, options[:output])
    else
      reporter.generate_diagnostic_report(file, options[:output])
    end
    
    puts "✓ HTML report generated: #{output_file}"
    puts "  Type: #{report_type}"
    puts "\nOpening in browser..."
    
    # Open in default browser
    if RbConfig::CONFIG['host_os'] =~ /mswin|mingw|cygwin/
      system("start #{output_file}")
    elsif RbConfig::CONFIG['host_os'] =~ /darwin/
      system("open #{output_file}")
    elsif RbConfig::CONFIG['host_os'] =~ /linux|bsd/
      system("xdg-open #{output_file}")
    end
  end

  desc "report-all", "Generate HTML reports for all model hyperparameter CSV files"
  option :pattern, aliases: :p, type: :string, default: 'model*.csv', desc: 'File pattern to match'
  option :open, type: :boolean, default: false, desc: 'Open all reports in browser'
  def report_all
    pattern = options[:pattern]
    files = Dir.glob(pattern)
    
    if files.empty?
      puts "✗ No files found matching pattern: #{pattern}"
      exit 1
    end
    
    puts "Found #{files.size} hyperparameter file(s):\n"
    
    reporter = HTMLReporter.new
    reports = []
    
    files.each do |file|
      puts "  Processing: #{file}"
      
      # Check if it's a hyperparameter file
      headers = CSV.read(file, headers: true).headers
      has_hyperparam_cols = (headers & ['experiment_id', 'rmse', 'mae', 'r2']).size >= 2
      
      if has_hyperparam_cols
        output_file = reporter.generate_hyperparam_report(file, nil)
        reports << output_file
        puts "    ✓ Report: #{output_file}"
      else
        puts "    ⚠ Skipped (not a hyperparameter file)"
      end
    end
    
    puts "\n✓ Generated #{reports.size} report(s)"
    
    if options[:open] && reports.any?
      puts "\nOpening reports in browser..."
      reports.each do |report|
        if RbConfig::CONFIG['host_os'] =~ /mswin|mingw|cygwin/
          system("start #{report}")
        elsif RbConfig::CONFIG['host_os'] =~ /darwin/
          system("open #{report}")
        elsif RbConfig::CONFIG['host_os'] =~ /linux|bsd/
          system("xdg-open #{report}")
        end
      end
    end
  end

  desc "diagnose FILE", "Deep analysis of CSV data quality - detects mixed types, missing values, outliers, distribution issues"
  def diagnose(file)
    unless File.exist?(file)
      puts "✗ File not found: #{file}"
      exit 1
    end
    
    diagnostics = CSVDiagnostics.new(file)
    diagnostics.diagnose
  end

  desc "clean FILE", "Analyze and clean CSV data - adds _cleaned columns with transformations, preserves originals"
  def clean(file)
    puts "Cleaning CSV file: #{file}"
    if CSVProcessor.clean(file)
      puts "✓ File cleaned successfully!"
      puts "\nOriginal columns preserved - view side-by-side comparison:"
      puts "  • Missing values filled (mean for numbers, 'MISSING' for text)"
      puts "  • Outliers replaced using IQR method"
      puts "  • Cleaned columns added with '_cleaned' suffix"
    else
      puts "✗ Failed to clean file. Check the logs for details."
      exit 1
    end
  end

  desc "merge FILE1 FILE2 [OUTPUT]", "Merge two CSV files (default: concatenate)"
  option :output, aliases: :o, type: :string, default: 'merged.csv', desc: 'Output file name'
  option :type, aliases: :t, type: :string, default: 'concat', desc: 'Merge type: concat or join'
  option :key, aliases: :k, type: :string, desc: 'Key column for join operation'
  def merge(file1, file2)
    output_file = options[:output]
    puts "Merging CSV files: #{file1} and #{file2}"
    puts "Output file: #{output_file}"
    
    if CSVProcessor.merge(file1, file2, output_file)
      puts "✓ Files merged successfully!"
    else
      puts "✗ Failed to merge files. Check the logs for details."
      exit 1
    end
  end

  desc "transform FILE", "Transform CSV data with custom operations"
  def transform(file)
    puts "Transforming CSV file: #{file}"
    if CSVProcessor.transform(file)
      puts "✓ File transformed successfully!"
    else
      puts "✗ Failed to transform file. Check the logs for details."
      exit 1
    end
  end

  desc "info FILE", "Display information about a CSV file"
  def info(file)
    unless File.exist?(file)
      puts "✗ File not found: #{file}"
      exit 1
    end

    begin
      data = CSV.read(file, headers: true)
      puts "\n=== CSV File Information ==="
      puts "File: #{file}"
      puts "Rows: #{data.length}"
      puts "Columns: #{data.headers.length}"
      puts "\nColumn Names:"
      data.headers.each_with_index do |header, idx|
        puts "  #{idx + 1}. #{header}"
      end
      puts "\nFirst 3 rows:"
      data.first(3).each_with_index do |row, idx|
        puts "\nRow #{idx + 1}:"
        row.each { |key, value| puts "  #{key}: #{value}" }
      end
    rescue StandardError => e
      puts "✗ Error reading file: #{e.message}"
      exit 1
    end
  end

  desc "version", "Display version information"
  def version
    puts "CSVy Organizer v1.0.0"
    puts "Ruby CSV processing and organization tool"
  end

  desc "encode FILE COLUMN", "One-hot encode a categorical column"
  option :output, aliases: :o, type: :string, desc: 'Output file name'
  option :type, aliases: :t, type: :string, default: 'onehot', desc: 'Encoding type: onehot or label'
  def encode(file, column)
    unless File.exist?(file)
      puts "✗ File not found: #{file}"
      exit 1
    end

    begin
      data = CSV.read(file, headers: true)
      preprocessor = DataPreprocessor.new
      
      encoded = if options[:type] == 'onehot'
        preprocessor.one_hot_encode(data, column)
      else
        preprocessor.label_encode(data, column)
      end
      
      output_file = options[:output] || file.gsub('.csv', "_#{options[:type]}_encoded.csv")
      CSV.open(output_file, 'w', write_headers: true, headers: encoded.headers) do |csv|
        encoded.each { |row| csv << row }
      end
      
      puts "✓ Column '#{column}' encoded successfully!"
      puts "Output: #{output_file}"
    rescue StandardError => e
      puts "✗ Error: #{e.message}"
      exit 1
    end
  end

  desc "normalize FILE COLUMN", "Normalize a numeric column to 0-1 range"
  option :output, aliases: :o, type: :string, desc: 'Output file name'
  option :method, aliases: :m, type: :string, default: 'minmax', desc: 'Method: minmax or zscore'
  def normalize(file, column)
    unless File.exist?(file)
      puts "✗ File not found: #{file}"
      exit 1
    end

    begin
      data = CSV.read(file, headers: true)
      preprocessor = DataPreprocessor.new
      
      normalized = if options[:method] == 'minmax'
        preprocessor.normalize(data, column)
      else
        preprocessor.standardize(data, column)
      end
      
      output_file = options[:output] || file.gsub('.csv', '_normalized.csv')
      CSV.open(output_file, 'w', write_headers: true, headers: normalized.headers) do |csv|
        normalized.each { |row| csv << row }
      end
      
      puts "✓ Column '#{column}' normalized successfully!"
      puts "Output: #{output_file}"
    rescue StandardError => e
      puts "✗ Error: #{e.message}"
      exit 1
    end
  end

  desc "validate FILE", "Validate data quality and generate report"
  def validate(file)
    unless File.exist?(file)
      puts "✗ File not found: #{file}"
      exit 1
    end

    validator = DataValidator.new
    report = validator.validate(file)
    
    puts "\n=== Validation Report ==="
    puts "File: #{report[:file]}"
    puts "Rows: #{report[:total_rows]}"
    puts "Columns: #{report[:total_columns]}"
    puts "\nColumn Types:"
    report[:column_types].each { |col, type| puts "  #{col}: #{type}" }
    
    if report[:issues].empty?
      puts "\n✓ No issues found!"
    else
      puts "\n⚠ Issues Found:"
      report[:issues].each { |issue| puts "  - #{issue}" }
    end
    
    unless report[:warnings].empty?
      puts "\n⚠ Warnings:"
      report[:warnings].each { |warning| puts "  - #{warning}" }
    end
  end

  desc "stats FILE", "Generate statistics for dataset"
  def stats(file)
    unless File.exist?(file)
      puts "✗ File not found: #{file}"
      exit 1
    end

    validator = DataValidator.new
    stats = validator.statistics(file)
    
    puts "\n=== Statistics Report ==="
    stats.each do |column, data|
      puts "\n#{column}:"
      data.each { |key, value| puts "  #{key}: #{value}" }
    end
  end

  desc "profile FILE", "Generate detailed data profile"
  def profile(file)
    unless File.exist?(file)
      puts "✗ File not found: #{file}"
      exit 1
    end

    validator = DataValidator.new
    profile = validator.profile(file)
    
    puts "\n=== Data Profile ==="
    puts "File: #{profile[:file]}"
    puts "Rows: #{profile[:rows]}"
    puts "Columns: #{profile[:columns]}"
    puts "Memory: #{profile[:memory_estimate]}"
    
    puts "\n--- Column Profiles ---"
    profile[:column_profiles].each do |column, data|
      puts "\n#{column}:"
      data.each { |key, value| puts "  #{key}: #{value}" }
    end
  end

  desc "db-import FILE TABLE", "Import CSV to SQLite database"
  option :database, aliases: :d, type: :string, default: 'data/csvs.db', desc: 'Database path'
  def db_import(file, table)
    unless File.exist?(file)
      puts "✗ File not found: #{file}"
      exit 1
    end

    db = DatabaseManager.new(options[:database])
    if db.import_csv(file, table)
      puts "✓ Successfully imported #{file} to table '#{table}'"
    else
      puts "✗ Failed to import data"
      exit 1
    end
    db.disconnect
  end

  desc "db-export TABLE FILE", "Export SQLite table to CSV"
  option :database, aliases: :d, type: :string, default: 'data/csvs.db', desc: 'Database path'
  def db_export(table, file)
    db = DatabaseManager.new(options[:database])
    if db.export_to_csv(table, file)
      puts "✓ Successfully exported table '#{table}' to #{file}"
    else
      puts "✗ Failed to export data"
      exit 1
    end
    db.disconnect
  end

  desc "db-query SQL", "Execute SQL query on database"
  option :database, aliases: :d, type: :string, default: 'data/csvs.db', desc: 'Database path'
  def db_query(sql)
    db = DatabaseManager.new(options[:database])
    results = db.query(sql)
    
    if results.empty?
      puts "No results found"
    else
      puts "\n=== Query Results ==="
      results.each_with_index do |row, idx|
        puts "\nRow #{idx + 1}:"
        row.each { |key, value| puts "  #{key}: #{value}" }
      end
    end
    db.disconnect
  end

  desc "db-tables", "List all tables in database"
  option :database, aliases: :d, type: :string, default: 'data/csvs.db', desc: 'Database path'
  def db_tables
    db = DatabaseManager.new(options[:database])
    tables = db.list_tables
    
    puts "\n=== Database Tables ==="
    if tables.empty?
      puts "No tables found"
    else
      tables.each { |table| puts "  - #{table}" }
    end
    db.disconnect
  end

  # Time Series Feature Engineering Commands
  desc "rolling FILE COLUMN", "Calculate rolling window statistics (moving average, sum, etc.)"
  option :window, aliases: :w, type: :numeric, default: 10, desc: 'Window size'
  option :stat, aliases: :s, type: :string, default: 'mean', desc: 'Statistic: mean, sum, max, min, std'
  option :group, aliases: :g, type: :string, desc: 'Group by column (e.g., team_name)'
  option :output, aliases: :o, type: :string, desc: 'Output file'
  def rolling(file, column)
    data = CSV.read(file, headers: true)
    ts = TimeSeriesFeatures.new
    
    stat = options[:stat].to_sym
    result = ts.rolling_window(data, column, options[:window], stat: stat, group_by: options[:group])
    
    output = options[:output] || file.gsub('.csv', "_rolling_#{stat}_#{options[:window]}.csv")
    CSV.open(output, 'w', write_headers: true, headers: result.headers) do |csv|
      result.each { |row| csv << row }
    end
    
    puts "✓ Rolling #{stat} calculated (window=#{options[:window]})"
    puts "Output: #{output}"
  end

  desc "ewma FILE COLUMN", "Calculate exponentially weighted moving average (recent games weighted more)"
  option :span, aliases: :s, type: :numeric, default: 10, desc: 'Span for EWMA calculation'
  option :group, aliases: :g, type: :string, desc: 'Group by column (e.g., team_name)'
  option :output, aliases: :o, type: :string, desc: 'Output file'
  def ewma(file, column)
    data = CSV.read(file, headers: true)
    ts = TimeSeriesFeatures.new
    
    result = ts.ewma(data, column, options[:span], group_by: options[:group])
    
    output = options[:output] || file.gsub('.csv', "_ewma_#{options[:span]}.csv")
    CSV.open(output, 'w', write_headers: true, headers: result.headers) do |csv|
      result.each { |row| csv << row }
    end
    
    puts "✓ EWMA calculated (span=#{options[:span]})"
    puts "Output: #{output}"
  end

  desc "lag FILE COLUMN PERIODS", "Create lag features (previous game values) - PERIODS is comma-separated (e.g., 1,3,5)"
  option :group, aliases: :g, type: :string, desc: 'Group by column (e.g., team_name)'
  option :output, aliases: :o, type: :string, desc: 'Output file'
  def lag(file, column, periods)
    data = CSV.read(file, headers: true)
    ts = TimeSeriesFeatures.new
    
    period_array = periods.split(',').map(&:to_i)
    result = ts.lag_features(data, column, period_array, group_by: options[:group])
    
    output = options[:output] || file.gsub('.csv', "_lag.csv")
    CSV.open(output, 'w', write_headers: true, headers: result.headers) do |csv|
      result.each { |row| csv << row }
    end
    
    puts "✓ Lag features created (periods: #{period_array.join(', ')})"
    puts "Output: #{output}"
  end

  desc "rate FILE NUMERATOR DENOMINATOR", "Calculate rate statistics (e.g., goals per game)"
  option :output_name, aliases: :n, type: :string, desc: 'Output column name'
  option :output_file, aliases: :o, type: :string, desc: 'Output file'
  def rate(file, numerator, denominator)
    data = CSV.read(file, headers: true)
    ts = TimeSeriesFeatures.new
    
    result = ts.rate_stat(data, numerator, denominator, output_name: options[:output_name])
    
    output = options[:output_file] || file.gsub('.csv', '_rate.csv')
    CSV.open(output, 'w', write_headers: true, headers: result.headers) do |csv|
      result.each { |row| csv << row }
    end
    
    puts "✓ Rate statistic calculated: #{numerator} / #{denominator}"
    puts "Output: #{output}"
  end

  desc "streak FILE COLUMN", "Calculate win/loss streaks from result column"
  option :group, aliases: :g, type: :string, desc: 'Group by column (e.g., team_name)'
  option :output, aliases: :o, type: :string, desc: 'Output file'
  def streak(file, column)
    data = CSV.read(file, headers: true)
    ts = TimeSeriesFeatures.new
    
    result = ts.calculate_streaks(data, column, group_by: options[:group])
    
    output = options[:output] || file.gsub('.csv', '_streak.csv')
    CSV.open(output, 'w', write_headers: true, headers: result.headers) do |csv|
      result.each { |row| csv << row }
    end
    
    puts "✓ Streaks calculated"
    puts "Output: #{output}"
  end

  desc "rest FILE DATE_COLUMN", "Calculate rest days between games"
  option :group, aliases: :g, type: :string, desc: 'Group by column (e.g., team_name)'
  option :output, aliases: :o, type: :string, desc: 'Output file'
  def rest(file, date_column)
    data = CSV.read(file, headers: true)
    ts = TimeSeriesFeatures.new
    
    result = ts.days_between(data, date_column, output_name: 'rest_days', group_by: options[:group])
    
    output = options[:output] || file.gsub('.csv', '_rest.csv')
    CSV.open(output, 'w', write_headers: true, headers: result.headers) do |csv|
      result.each { |row| csv << row }
    end
    
    puts "✓ Rest days calculated"
    puts "Output: #{output}"
  end

  desc "cumulative FILE COLUMN", "Calculate cumulative statistics (running total)"
  option :stat, aliases: :s, type: :string, default: 'sum', desc: 'Statistic: sum, mean, max, min'
  option :group, aliases: :g, type: :string, desc: 'Group by column (e.g., team_name)'
  option :output, aliases: :o, type: :string, desc: 'Output file'
  def cumulative(file, column)
    data = CSV.read(file, headers: true)
    ts = TimeSeriesFeatures.new
    
    stat = options[:stat].to_sym
    result = ts.cumulative(data, column, stat: stat, group_by: options[:group])
    
    output = options[:output] || file.gsub('.csv', "_cumulative_#{stat}.csv")
    CSV.open(output, 'w', write_headers: true, headers: result.headers) do |csv|
      result.each { |row| csv << row }
    end
    
    puts "✓ Cumulative #{stat} calculated"
    puts "Output: #{output}"
  end

  desc "rank FILE COLUMN", "Rank values within dataset or groups"
  option :group, aliases: :g, type: :string, desc: 'Group by column (e.g., team_name)'
  option :ascending, aliases: :a, type: :boolean, default: false, desc: 'Rank ascending (default: descending)'
  option :output, aliases: :o, type: :string, desc: 'Output file'
  def rank(file, column)
    data = CSV.read(file, headers: true)
    ts = TimeSeriesFeatures.new
    
    result = ts.rank_column(data, column, group_by: options[:group], ascending: options[:ascending])
    
    output = options[:output] || file.gsub('.csv', '_ranked.csv')
    CSV.open(output, 'w', write_headers: true, headers: result.headers) do |csv|
      result.each { |row| csv << row }
    end
    
    puts "✓ Column ranked"
    puts "Output: #{output}"
  end

  # IO Operations
  desc "from-clipboard", "Read CSV from clipboard and display info"
  def from_clipboard
    data = CSVIOHandler.from_clipboard
    puts "\n=== CSV from Clipboard ==="
    puts "Rows: #{data.length}"
    puts "Columns: #{data.headers.length}"
    puts "Headers: #{data.headers.join(', ')}"
  rescue => e
    puts "✗ Error reading from clipboard: #{e.message}"
  end

  desc "to-clipboard FILE", "Copy CSV file to clipboard"
  def to_clipboard(file)
    data = CSV.read(file, headers: true)
    CSVIOHandler.to_clipboard(data)
    puts "✓ CSV copied to clipboard (#{data.length} rows)"
  rescue => e
    puts "✗ Error: #{e.message}"
  end

  # Hyperparameter Management
  desc "hyperparam-grid CONFIG_FILE", "Generate hyperparameter grid from YAML config"
  option :output, aliases: :o, type: :string, desc: 'Output CSV file'
  option :sample, aliases: :s, type: :numeric, desc: 'Sample N random configurations instead of full grid'
  def hyperparam_grid(config_file)
    hpm = HyperparameterManager.new
    
    if options[:sample]
      output = hpm.random_search(config_file, options[:sample], options[:output])
    else
      output = hpm.generate_grid(config_file, options[:output], sample_size: options[:sample])
    end
    
    puts "✓ Hyperparameter grid generated: #{output}"
  end

  desc "hyperparam-random CONFIG_FILE N", "Generate N random hyperparameter configurations"
  option :output, aliases: :o, type: :string, desc: 'Output CSV file'
  def hyperparam_random(config_file, n)
    hpm = HyperparameterManager.new
    output = hpm.random_search(config_file, n.to_i, options[:output])
    puts "✓ Random search configurations generated: #{output}"
  end

  desc "hyperparam-bayesian CONFIG_FILE", "Bayesian optimization with Gaussian Process surrogate"
  option :iterations, aliases: :i, type: :numeric, default: 20, desc: 'Total iterations'
  option :initial, type: :numeric, default: 5, desc: 'Initial random samples'
  option :acquisition, aliases: :a, type: :string, default: 'ei', desc: 'Acquisition function (ei, ucb, poi)'
  option :output, aliases: :o, type: :string, desc: 'Output CSV file'
  def hyperparam_bayesian(config_file)
    hpm = HyperparameterManager.new
    output = hpm.bayesian_optimize(
      config_file,
      n_iterations: options[:iterations],
      n_initial: options[:initial],
      acquisition: options[:acquisition],
      output_file: options[:output]
    )
    puts "✓ Bayesian optimization complete: #{output}"
    puts "  Generated #{options[:iterations]} configurations"
    puts "  Next: Train models and use 'add-result' to record metrics"
  end

  desc "hyperparam-genetic CONFIG_FILE", "Genetic algorithm optimization"
  option :population, aliases: :p, type: :numeric, default: 20, desc: 'Population size'
  option :generations, aliases: :g, type: :numeric, default: 10, desc: 'Number of generations'
  option :mutation, aliases: :m, type: :numeric, default: 0.1, desc: 'Mutation rate (0.0-1.0)'
  option :output, aliases: :o, type: :string, desc: 'Output CSV file'
  def hyperparam_genetic(config_file)
    hpm = HyperparameterManager.new
    output = hpm.genetic_algorithm(
      config_file,
      population_size: options[:population],
      generations: options[:generations],
      mutation_rate: options[:mutation],
      output_file: options[:output]
    )
    total_configs = options[:population] + (options[:population] / 2) * options[:generations]
    puts "✓ Genetic algorithm complete: #{output}"
    puts "  Generated ~#{total_configs} configurations"
    puts "  Next: Train models and use 'add-result' to record metrics"
  end

  desc "hyperparam-annealing CONFIG_FILE", "Simulated annealing optimization"
  option :iterations, aliases: :i, type: :numeric, default: 100, desc: 'Number of iterations'
  option :temp, aliases: :t, type: :numeric, default: 1.0, desc: 'Initial temperature'
  option :cooling, aliases: :c, type: :numeric, default: 0.95, desc: 'Cooling rate (0.0-1.0)'
  option :output, aliases: :o, type: :string, desc: 'Output CSV file'
  def hyperparam_annealing(config_file)
    hpm = HyperparameterManager.new
    output = hpm.simulated_annealing(
      config_file,
      n_iterations: options[:iterations],
      initial_temp: options[:temp],
      cooling_rate: options[:cooling],
      output_file: options[:output]
    )
    puts "✓ Simulated annealing complete: #{output}"
    puts "  Generated #{options[:iterations] + 1} configurations"
    puts "  Next: Train models and use 'add-result' to record metrics"
  end

  desc "add-result TRACKING_FILE EXPERIMENT_ID", "Add experiment results to tracking file"
  option :rmse, type: :numeric, desc: 'RMSE value'
  option :mae, type: :numeric, desc: 'MAE value'
  option :r2, type: :numeric, desc: 'R-squared value'
  option :notes, aliases: :n, type: :string, desc: 'Notes about experiment'
  def add_result(tracking_file, experiment_id)
    metrics = {}
    metrics[:rmse] = options[:rmse] if options[:rmse]
    metrics[:mae] = options[:mae] if options[:mae]
    metrics[:r2] = options[:r2] if options[:r2]
    
    hpm = HyperparameterManager.new
    if hpm.add_result(tracking_file, experiment_id, metrics, notes: options[:notes])
      puts "✓ Results added for experiment #{experiment_id}"
    else
      puts "✗ Failed to add results"
    end
  end

  desc "best-params TRACKING_FILE", "Find best hyperparameters based on metric"
  option :metric, aliases: :m, type: :string, default: 'rmse', desc: 'Metric to optimize (rmse, mae, r2)'
  option :ascending, aliases: :a, type: :boolean, default: true, desc: 'Lower is better (true for rmse/mae, false for r2)'
  def best_params(tracking_file)
    hpm = HyperparameterManager.new
    best = hpm.find_best(tracking_file, metric: options[:metric], ascending: options[:ascending])
    
    if best
      puts "\n=== Best Hyperparameters (optimizing #{options[:metric]}) ==="
      best.each { |param, value| puts "  #{param}: #{value}" }
    else
      puts "No completed experiments found"
    end
  end

  desc "export-params CONFIG_FILE", "Export hyperparameters in different formats"
  option :format, aliases: :f, type: :string, default: 'python', desc: 'Format: python, json, yaml, ruby'
  option :output, aliases: :o, type: :string, desc: 'Output file'
  def export_params(config_file)
    hpm = HyperparameterManager.new
    hpm.export_params(config_file, format: options[:format], output_file: options[:output])
  end

  desc "compare-experiments TRACKING_FILE IDS", "Compare multiple experiments (comma-separated IDs)"
  def compare_experiments(tracking_file, ids)
    experiment_ids = ids.split(',').map(&:to_i)
    hpm = HyperparameterManager.new
    hpm.compare_experiments(tracking_file, experiment_ids)
  end

  # ===== COMPETITIVE EDGE COMMANDS =====
  
  desc "advanced-features FILE", "Create advanced competition-winning features"
  option :output, aliases: :o, type: :string, desc: 'Output file'
  option :team_col, type: :string, default: 'Team', desc: 'Team column name'
  option :wins_col, type: :string, default: 'W', desc: 'Wins column'
  option :losses_col, type: :string, default: 'L', desc: 'Losses column'
  option :diff_col, type: :string, default: 'DIFF', desc: 'Goal differential column'
  option :gf_col, type: :string, default: 'GF', desc: 'Goals for column'
  option :ga_col, type: :string, default: 'GA', desc: 'Goals against column'
  option :games_col, type: :string, default: 'GP', desc: 'Games played column'
  def advanced_features(file)
    unless File.exist?(file)
      puts "✗ File not found: #{file}"
      exit 1
    end
    
    puts "🚀 Creating advanced features..."
    
    data = CSV.read(file, headers: true).map(&:to_h)
    af = AdvancedFeatures.new
    
    # Team strength index
    data = af.calculate_team_strength_index(data, options[:team_col], options[:wins_col], 
                                            options[:losses_col], options[:diff_col])
    
    # Pythagorean expectation
    data = af.calculate_pythagorean_wins(data, options[:gf_col], options[:ga_col], options[:games_col])
    
    # Interaction features
    data = af.create_interaction_features(data, options[:gf_col], options[:wins_col], 'offense_power')
    data = af.create_interaction_features(data, options[:ga_col], options[:losses_col], 'defense_weakness')
    
    # Polynomial features
    data = af.create_polynomial_features(data, options[:diff_col], degree: 2)
    
    # Save output
    output_file = options[:output] || file.gsub('.csv', '_advanced.csv')
    CSV.open(output_file, 'w') do |csv|
      csv << data.first.keys
      data.each { |row| csv << data.first.keys.map { |k| row[k] } }
    end
    
    puts "✓ Advanced features saved to: #{output_file}"
    puts "  Added: team_strength_index, pythagorean_wins, offense_power, defense_weakness, #{options[:diff_col]}_pow2"
  end

  desc "validate-model PREDICTIONS_CSV", "Validate model predictions with advanced metrics"
  option :actual_col, type: :string, default: 'actual', desc: 'Actual values column'
  option :pred_col, type: :string, default: 'predicted', desc: 'Predicted values column'
  option :bootstrap, type: :boolean, default: false, desc: 'Run bootstrap confidence intervals'
  option :calibration, type: :boolean, default: false, desc: 'Check prediction calibration'
  def validate_model(predictions_csv)
    unless File.exist?(predictions_csv)
      puts "✗ File not found: #{predictions_csv}"
      exit 1
    end
    
    puts "🔍 Validating model predictions..."
    
    data = CSV.read(predictions_csv, headers: true)
    actuals = data[options[:actual_col]].map(&:to_f)
    predictions = data[options[:pred_col]].map(&:to_f)
    
    validator = ModelValidator.new
    
    # Basic metrics
    rmse = Math.sqrt(predictions.zip(actuals).map { |p, a| (p - a) ** 2 }.sum / actuals.size)
    mae = predictions.zip(actuals).map { |p, a| (p - a).abs }.sum / actuals.size
    
    puts "\n=== Validation Metrics ==="
    puts "RMSE: #{rmse.round(4)}"
    puts "MAE: #{mae.round(4)}"
    
    # Bootstrap if requested
    if options[:bootstrap]
      puts "\n=== Bootstrap Confidence Intervals (1000 iterations) ==="
      bootstrap_results = validator.bootstrap_metric(predictions, actuals, metric: :rmse, n_iterations: 1000)
      puts "RMSE Mean: #{bootstrap_results[:mean].round(4)}"
      puts "RMSE 95% CI: [#{bootstrap_results[:ci_95_lower].round(4)}, #{bootstrap_results[:ci_95_upper].round(4)}]"
    end
    
    # Calibration check
    if options[:calibration]
      puts "\n=== Calibration Analysis ==="
      calibration = validator.check_calibration(predictions, actuals, n_bins: 10)
      puts "Mean Calibration Error: #{calibration[:mean_calibration_error].round(4)}"
    end
  end

  desc "ensemble-optimize PREDICTIONS_DIR", "Optimize ensemble weights from multiple model predictions"
  option :actuals, type: :string, required: true, desc: 'CSV file with actual values'
  option :actual_col, type: :string, default: 'actual', desc: 'Actual values column'
  option :output, aliases: :o, type: :string, desc: 'Output file for optimal weights'
  option :method, aliases: :m, type: :string, default: 'inverse_rmse', desc: 'Weight method: inverse_rmse, softmax, equal, grid_search'
  def ensemble_optimize(predictions_dir)
    unless Dir.exist?(predictions_dir)
      puts "✗ Directory not found: #{predictions_dir}"
      exit 1
    end
    
    unless File.exist?(options[:actuals])
      puts "✗ Actuals file not found: #{options[:actuals]}"
      exit 1
    end
    
    puts "🎯 Optimizing ensemble weights..."
    
    # Load actual values
    actuals_data = CSV.read(options[:actuals], headers: true)
    actuals = actuals_data[options[:actual_col]].map(&:to_f)
    
    # Load predictions from multiple models
    model_predictions = {}
    Dir.glob(File.join(predictions_dir, '*.csv')).each do |pred_file|
      model_name = File.basename(pred_file, '.csv')
      pred_data = CSV.read(pred_file, headers: true)
      
      pred_col = pred_data.headers.find { |h| h =~ /pred/i } || pred_data.headers[1]
      predictions = pred_data[pred_col].map(&:to_f)
      
      model_predictions[model_name] = predictions
    end
    
    puts "Found #{model_predictions.size} models: #{model_predictions.keys.join(', ')}"
    
    ensemble = EnsembleOptimizer.new
    
    # Calculate optimal weights
    method = options[:method].to_sym rescue :inverse_rmse
    result = ensemble.optimize_ensemble_weights(model_predictions.values, actuals, method: method)
    
    puts "\n=== Optimal Ensemble Weights ==="
    model_predictions.keys.each_with_index do |model, idx|
      puts "  #{model}: #{result[:optimal_weights][idx].round(4)}"
    end
    puts "\nEnsemble RMSE: #{result[:best_rmse].round(4)}"
    puts "Best individual RMSE: #{result[:baseline_rmse].round(4)}"
    puts "Improvement: #{result[:improvement].round(4)} (#{result[:improvement_pct].round(2)}%)"
    
    # Save weights if output specified
    if options[:output]
      CSV.open(options[:output], 'w') do |csv|
        csv << ['model', 'weight']
        model_predictions.keys.each_with_index do |model, idx|
          csv << [model, result[:optimal_weights][idx]]
        end
      end
      puts "✓ Weights saved to: #{options[:output]}"
    end
  end

  desc "competitive-pipeline FILE", "Run full competitive preprocessing pipeline"
  option :output_dir, aliases: :o, type: :string, default: 'data/processed', desc: 'Output directory'
  def competitive_pipeline(file)
    unless File.exist?(file)
      puts "✗ File not found: #{file}"
      exit 1
    end
    
    require_relative 'scripts/competitive_pipeline'
    
    preprocessor = CompetitivePreprocessor.new(file, options[:output_dir])
    output_file = preprocessor.run_full_pipeline
    
    puts "\n✓ Competitive features ready!"
    puts "Next steps:"
    puts "  1. Push to GitHub: git add . && git commit -m 'Add competitive features' && git push"
    puts "  2. Pull in DeepNote and train models"
    puts "  3. Track results: ruby cli.rb add-result <tracking_file> <experiment_id> --rmse X --mae Y"
    puts "  4. Find best params: ruby cli.rb best-params <tracking_file> --metric rmse"
  end

  desc "feature-correlation FILE TARGET_COL", "Analyze feature correlations with target column"
  option :output, aliases: :o, type: :string, desc: 'Output CSV file for correlations'
  def feature_correlation(file, target_col)
    unless File.exist?(file)
      puts "✗ File not found: #{file}"
      exit 1
    end
    
    puts "📊 Analyzing feature correlations..."
    
    data = CSV.read(file, headers: true).map(&:to_h)
    af = AdvancedFeatures.new
    
    # Get all numeric feature columns
    feature_cols = data.first.keys.reject { |k| k == target_col || k.match?(/^(experiment_id|timestamp|notes)$/i) }
    
    correlations = af.analyze_feature_correlations(data, target_col, feature_cols)
    
    # Save to CSV if output specified
    if options[:output]
      CSV.open(options[:output], 'w') do |csv|
        csv << ['feature', 'correlation', 'abs_correlation']
        correlations.sort_by { |k, v| -v.abs }.each do |feature, corr|
          csv << [feature, corr, corr.abs]
        end
      end
      puts "✓ Correlations saved to: #{options[:output]}"
    end
  end

  desc "model-track TRACKING_FILE", "Track and analyze model performance"
  option :output, aliases: :o, type: :string, desc: 'Output JSON file'
  def model_track(tracking_file)
    unless File.exist?(tracking_file)
      puts "✗ File not found: #{tracking_file}"
      exit 1
    end
    
    tracker = ModelTracker.new
    stats = tracker.track_model_performance(tracking_file, options[:output])
    
    if stats
      puts "\n=== Model Performance Summary ==="
      puts "Completed: #{stats[:completed_experiments]}/#{stats[:total_experiments]} (#{stats[:completion_rate]}%)"
      puts "Best RMSE: #{stats[:rmse][:best].round(4)}"
      puts "Mean RMSE: #{stats[:rmse][:mean].round(4)}"
      puts "Top 10 experiments: #{stats[:top_10_experiments].join(', ')}"
    end
  end

  desc "model-compare TRACKING_FILES", "Compare multiple model tracking files"
  option :output, aliases: :o, type: :string, default: 'model_comparison.json', desc: 'Output JSON file'
  def model_compare(*tracking_files)
    if tracking_files.empty?
      puts "✗ Please provide at least one tracking file"
      exit 1
    end
    
    missing = tracking_files.reject { |f| File.exist?(f) }
    unless missing.empty?
      puts "✗ Files not found: #{missing.join(', ')}"
      exit 1
    end
    
    tracker = ModelTracker.new
    comparison = tracker.compare_models(tracking_files, options[:output])
    
    puts "\n=== Model Comparison ==="
    comparison[:models].each do |model, stats|
      puts "#{model}:"
      puts "  Best RMSE: #{stats[:best_rmse].round(4)}"
      puts "  Mean RMSE: #{stats[:mean_rmse].round(4)}"
      puts "  Completed: #{stats[:completed]}/#{stats[:total_configs]}"
    end
    puts "\n🏆 Best Model: #{comparison[:best_model]} (RMSE: #{comparison[:best_rmse].round(4)})"
  end

  desc "model-select TRACKING_FILES", "Automatically select best model based on validation metrics"
  option :metric, aliases: :m, type: :string, default: 'rmse', desc: 'Metric: rmse, mae, r2'
  option :criteria, aliases: :c, type: :string, default: 'best', desc: 'Criteria: best, mean, median'
  def model_select(*tracking_files)
    if tracking_files.empty?
      puts "✗ Please provide at least one tracking file"
      exit 1
    end
    
    missing = tracking_files.reject { |f| File.exist?(f) }
    unless missing.empty?
      puts "✗ Files not found: #{missing.join(', ')}"
      exit 1
    end
    
    tracker = ModelTracker.new
    selection = tracker.select_best_model(tracking_files, metric: options[:metric].to_sym, criteria: options[:criteria].to_sym)
    
    if selection
      puts "\n=== Model Selection ==="
      puts "Criteria: #{selection[:selection_criteria]}"
      puts "\nAll Models:"
      selection[:all_scores].sort_by { |_, score| score }.each do |model, score|
        marker = model == selection[:best_model] ? "🏆" : "  "
        puts "#{marker} #{model}: #{score.round(4)}"
      end
      puts "\nSelected: #{selection[:best_model]} (#{selection[:best_score].round(4)})"
    else
      puts "✗ No completed experiments found"
    end
  end

  desc "model-report TRACKING_FILE", "Generate detailed performance report"
  option :output, aliases: :o, type: :string, desc: 'Output text file'
  def model_report(tracking_file)
    unless File.exist?(tracking_file)
      puts "✗ File not found: #{tracking_file}"
      exit 1
    end
    
    tracker = ModelTracker.new
    report_file = tracker.generate_performance_report(tracking_file, options[:output])
    
    if report_file
      puts "✓ Performance report generated: #{report_file}"
      puts "\nPreview:"
      puts File.read(report_file).lines.first(15).join
    end
  end

  desc "export-predictions PREDICTIONS_CSV", "Export predictions in formats for Python/DeepNote"
  option :format, aliases: :f, type: :string, default: 'numpy', desc: 'Format: numpy, pandas, json'
  option :output, aliases: :o, type: :string, desc: 'Output file'
  option :pred_col, type: :string, default: 'predicted', desc: 'Predictions column'
  def export_predictions(predictions_csv)
    unless File.exist?(predictions_csv)
      puts "✗ File not found: #{predictions_csv}"
      exit 1
    end
    
    data = CSV.read(predictions_csv, headers: true)
    
    # Guard against missing prediction column
    unless data.headers.include?(options[:pred_col])
      puts "✗ Column '#{options[:pred_col]}' not found in CSV"
      puts "Available columns: #{data.headers.join(', ')}"
      exit 1
    end
    
    predictions = data[options[:pred_col]].map(&:to_f)
    
    output_file = options[:output] || predictions_csv.gsub('.csv', "_#{options[:format]}.txt")
    
    case options[:format].downcase
    when 'numpy'
      File.write(output_file, "import numpy as np\npredictions = np.array([#{predictions.join(', ')}])\n")
      
    when 'pandas'
      File.write(output_file, "import pandas as pd\npredictions = pd.Series([#{predictions.join(', ')}])\n")
      
    when 'json'
      require 'json'
      File.write(output_file, JSON.pretty_generate({ predictions: predictions }))
      
    else
      puts "✗ Unknown format: #{options[:format]}"
      exit 1
    end
    
    puts "✓ Predictions exported to #{output_file} (#{options[:format]} format)"
    puts "  #{predictions.size} predictions exported"
  end

  desc "diversity-analysis PREDICTIONS_DIR ACTUALS", "Analyze ensemble model diversity"
  option :actual_col, type: :string, default: 'actual', desc: 'Actual values column'
  def diversity_analysis(predictions_dir, actuals_file)
    unless Dir.exist?(predictions_dir)
      puts "✗ Directory not found: #{predictions_dir}"
      exit 1
    end
    
    unless File.exist?(actuals_file)
      puts "✗ Actuals file not found: #{actuals_file}"
      exit 1
    end
    
    puts "📊 Analyzing model diversity..."
    
    # Load actual values
    actuals_data = CSV.read(actuals_file, headers: true)
    actuals = actuals_data[options[:actual_col]].map(&:to_f)
    
    # Load predictions
    model_predictions = []
    model_names = []
    
    Dir.glob(File.join(predictions_dir, '*.csv')).each do |pred_file|
      model_names << File.basename(pred_file, '.csv')
      pred_data = CSV.read(pred_file, headers: true)
      pred_col = pred_data.headers.find { |h| h =~ /pred/i } || pred_data.headers[1]
      model_predictions << pred_data[pred_col].map(&:to_f)
    end
    
    ensemble = EnsembleOptimizer.new
    diversity = ensemble.analyze_model_diversity(model_predictions, actuals)
    
    puts "\n=== Model Diversity Analysis ==="
    puts "Average error correlation: #{diversity[:avg_correlation].round(4)}"
    puts "Diversity score: #{diversity[:diversity_score].round(4)} (higher = more diverse)"
    
    puts "\n=== Pairwise Correlations ==="
    model_names.each_with_index do |model_i, i|
      model_names.each_with_index do |model_j, j|
        next if i >= j
        corr = diversity[:correlations][i][j]
        puts "  #{model_i} <-> #{model_j}: #{corr.round(3)}"
      end
    end
    
    if diversity[:diversity_score] > 0.5
      puts "\n✓ Good diversity! Models are complementary (ensemble will perform well)"
    else
      puts "\n⚠ Low diversity. Models may be too similar (consider different algorithms)"
    end
  end
  
  desc "train-neural-network DATA_FILE", "Train neural network model with hyperparameter search"
  option :config, aliases: :c, type: :string, desc: 'Hyperparameter config YAML (default: model6_neural_network.yaml)'
  option :iterations, aliases: :i, type: :numeric, default: 50, desc: 'Number of random search iterations'
  option :target, aliases: :t, type: :string, default: 'goals', desc: 'Target column name'
  option :output, aliases: :o, type: :string, default: 'model6_neural_network_results.csv', desc: 'Output CSV'
  def train_neural_network(data_file)
    unless File.exist?(data_file)
      puts "✗ Data file not found: #{data_file}"
      exit 1
    end
    
    puts "🧠 Training neural network model..."
    puts "  Data: #{data_file}"
    puts "  Iterations: #{options[:iterations]}"
    puts "  Target: #{options[:target]}"
    
    nn = NeuralNetworkWrapper.new
    
    # Check dependencies first
    unless nn.check_dependencies
      puts "\n✗ Missing Python dependencies. Install with:"
      puts "  pip install tensorflow scikit-learn pandas numpy pyyaml"
      exit 1
    end
    
    # Train model
    begin
      results = nn.train(
        data_file,
        config_file: options[:config],
        iterations: options[:iterations].to_i,
        target: options[:target],
        output_csv: options[:output]
      )
      
      puts "\n✓ Neural network training complete!"
      puts "\n=== Best Configuration ==="
      best = results[:best]
      puts "  Architecture: #{best['layer1_units']}-#{best['layer2_units']}-#{best['layer3_units']}"
      puts "  Dropout: #{best['dropout_rate']}"
      puts "  Learning rate: #{best['learning_rate']}"
      puts "  Batch size: #{best['batch_size']}"
      
      puts "\n=== Performance ==="
      puts "  RMSE: #{results[:best_rmse].round(4)}"
      puts "  R²: #{results[:best_r2].round(4)}"
      puts "  MAE: #{best['mae'].to_f.round(4)}"
      
      puts "\n=== Output Files ==="
      puts "  Results: #{options[:output]}"
      puts "  Model: models/best_nn_model.keras"
      puts "  Scaler: models/scaler.pkl"
      
      puts "\n💡 Use this model in ensemble with:"
      puts "  ruby cli.rb ensemble-with-nn #{data_file} --actuals your_actuals.csv"
      
    rescue => e
      puts "\n✗ Training failed: #{e.message}"
      puts e.backtrace.first(5).join("\n")
      exit 1
    end
  end
  
  desc "predict-neural-network DATA_FILE", "Make predictions with trained neural network"
  option :target, aliases: :t, type: :string, default: 'goals', desc: 'Target column name'
  option :output, aliases: :o, type: :string, desc: 'Output CSV for predictions'
  def predict_neural_network(data_file)
    unless File.exist?(data_file)
      puts "✗ Data file not found: #{data_file}"
      exit 1
    end
    
    puts "🔮 Making predictions with neural network..."
    
    nn = NeuralNetworkWrapper.new
    model_info = nn.model_info
    
    unless model_info[:trained]
      puts "✗ No trained model found. Train first with:"
      puts "  ruby cli.rb train-neural-network #{data_file}"
      exit 1
    end
    
    begin
      result = nn.predict(data_file, target: options[:target])
      predictions = result['predictions']
      
      puts "✓ Generated #{predictions.size} predictions"
      
      if result['metrics']
        puts "\n=== Test Metrics ==="
        puts "  RMSE: #{result['metrics']['rmse'].round(4)}"
        puts "  R²: #{result['metrics']['r2'].round(4)}"
        puts "  MAE: #{result['metrics']['mae'].round(4)}"
      end
      
      # Save predictions if output specified
      if options[:output]
        CSV.open(options[:output], 'w') do |csv|
          csv << ['prediction']
          predictions.each { |p| csv << [p] }
        end
        puts "\n✓ Predictions saved to: #{options[:output]}"
      else
        puts "\nFirst 5 predictions:"
        predictions.first(5).each_with_index { |p, i| puts "  [#{i}] #{p.round(4)}" }
      end
      
    rescue => e
      puts "\n✗ Prediction failed: #{e.message}"
      exit 1
    end
  end
  
  desc "ensemble-with-nn DATA_FILE", "Build full 6-model ensemble including neural network"
  option :actuals, type: :string, required: true, desc: 'CSV file with actual values'
  option :actual_col, type: :string, default: 'goals', desc: 'Actual values column'
  option :models, type: :array, default: ['rf', 'xgb', 'elo', 'linear', 'nn'], desc: 'Models to include'
  option :analyze, type: :boolean, default: false, desc: 'Analyze NN contribution separately'
  option :output, aliases: :o, type: :string, desc: 'Output file for ensemble predictions'
  def ensemble_with_nn(data_file)
    unless File.exist?(data_file)
      puts "✗ Data file not found: #{data_file}"
      exit 1
    end
    
    unless File.exist?(options[:actuals])
      puts "✗ Actuals file not found: #{options[:actuals]}"
      exit 1
    end
    
    puts "🎯 Building ensemble with neural network..."
    puts "  Models: #{options[:models].join(', ')}"
    
    # Load actuals
    actuals_data = CSV.read(options[:actuals], headers: true)
    
    unless actuals_data.headers.include?(options[:actual_col])
      puts "✗ Column '#{options[:actual_col]}' not found in #{options[:actuals]}"
      puts "  Available columns: #{actuals_data.headers.join(', ')}"
      exit 1
    end
    
    actuals = actuals_data[options[:actual_col]].map(&:to_f)
    
    ensemble = EnsembleOptimizer.new
    
    # Analyze NN contribution if requested
    if options[:analyze]
      puts "\n=== Neural Network Contribution Analysis ==="
      base_models = options[:models].reject { |m| m =~ /nn|neural/i }.map(&:to_sym)
      
      analysis = ensemble.analyze_nn_contribution(
        data_file,
        actuals,
        base_models: base_models
      )
      
      puts "Without NN: RMSE = #{analysis[:rmse_without_nn].round(4)}"
      puts "With NN:    RMSE = #{analysis[:rmse_with_nn].round(4)}"
      puts "Improvement: #{analysis[:improvement].round(4)} (#{analysis[:improvement_pct].round(2)}%)"
      puts "NN Weight: #{analysis[:nn_weight].round(4)}"
    end
    
    # Build full ensemble
    puts "\n=== Building Full Ensemble ==="
    result = ensemble.build_full_ensemble(
      data_file,
      actuals,
      models: options[:models].map(&:to_sym)
    )
    
    if result
      puts "\n✓ Ensemble complete!"
      puts "\n=== Model Weights ==="
      result[:weights].each do |model, weight|
        puts "  #{model}: #{weight.round(4)}"
      end
      
      puts "\n=== Performance ==="
      puts "  Ensemble RMSE: #{result[:rmse].round(4)}"
      puts "  Improvement: #{result[:optimization][:improvement].round(4)} (#{result[:optimization][:improvement_pct].round(2)}%)"
      
      # Save predictions if output specified
      if options[:output]
        CSV.open(options[:output], 'w') do |csv|
          csv << ['ensemble_prediction']
          result[:predictions].each { |p| csv << [p] }
        end
        puts "\n✓ Predictions saved to: #{options[:output]}"
      end
    else
      puts "\n✗ Failed to build ensemble"
      exit 1
    end
  end
  
  desc "nn-status", "Check neural network model status and dependencies"
  def nn_status
    puts "🧠 Neural Network Status\n\n"
    
    nn = NeuralNetworkWrapper.new
    
    puts "=== Python Dependencies ==="
    if nn.check_dependencies
      puts "✓ All dependencies installed"
    else
      puts "✗ Missing dependencies - see errors above"
    end
    
    puts "\n=== Trained Model ==="
    info = nn.model_info
    
    if info[:trained]
      puts "✓ Model trained and ready"
      puts "  Path: #{info[:model_path]}"
      puts "  Size: #{(info[:model_size] / 1024.0).round(2)} KB"
      puts "  Modified: #{info[:modified]}"
    else
      puts "✗ No trained model found"
      puts "  Train with: ruby cli.rb train-neural-network DATA_FILE"
    end
    
    puts "\n=== Integration ==="
    puts "Available commands:"
    puts "  ruby cli.rb train-neural-network DATA_FILE --iterations 100"
    puts "  ruby cli.rb predict-neural-network DATA_FILE"
    puts "  ruby cli.rb ensemble-with-nn DATA_FILE --actuals ACTUALS_FILE"
  end
end

# Run the CLI
CSVOrganizer.start(ARGV)
