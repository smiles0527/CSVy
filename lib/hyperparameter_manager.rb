require 'csv'
require 'json'
require 'logger'

class HyperparameterManager
  attr_accessor :params, :results

  def initialize(param_config = {})
    @params = param_config
    @results = []
  end

  # Add a hyperparameter with its search space
  def add_param(name, config)
    @params[name] = config
  end

  # Perform random search over the hyperparameter space
  # Supports both:
  # - New style: random_search(iterations = 10) { |sample, i| ... }
  # - Old style: random_search(config_file, n_samples, output_file = nil)
  def random_search(arg1 = 10, arg2 = nil, arg3 = nil)
    # Detect old-style call: random_search(config_file, n_samples, output_file)
    if arg1.is_a?(String) && (arg2.is_a?(Integer) || arg2.is_a?(String))
      config_file = arg1
      n_samples = arg2.to_i
      output_file = arg3
      
      # Load config if @params is empty
      if @params.empty? && File.exist?(config_file)
        require 'yaml'
        @params = YAML.safe_load_file(config_file, permitted_classes: [Symbol], aliases: true)
      end
      
      results = []
      n_samples.times do |i|
        sample = {}
        @params.each do |param_name, config|
          sample[param_name] = sample_param(config)
        end
        results << sample
      end
      
      # Save to file if specified
      if output_file && !results.empty?
        CSV.open(output_file, 'w') do |csv|
          csv << results.first.keys
          results.each { |r| csv << r.values }
        end
        return output_file
      end
      
      return results
    else
      # New-style call: random_search(iterations) { |sample, i| ... }
      iterations = arg1
      iterations.times do |i|
        sample = {}
        @params.each do |param_name, config|
          sample[param_name] = sample_param(config)
        end
        yield(sample, i) if block_given?
      end
    end
  end

  # Sample a single parameter value based on its configuration
  private

  def sample_param(config)
    # Handle range format [min, max, 'range'] for continuous parameters
    if config.is_a?(Array) && config.length == 3 && config[2] == 'range'
      min_val = config[0]
      max_val = config[1]
      # Generate random float between min and max
      rand(min_val..max_val)
    # Handle discrete choice arrays
    elsif config.is_a?(Array)
      config.sample
    # Handle hash configurations with 'type' key
    elsif config.is_a?(Hash)
      case config['type']
      when 'range'
        min_val = config['min']
        max_val = config['max']
        rand(min_val..max_val)
      when 'choice'
        config['values'].sample
      when 'int'
        min_val = config['min'] || 0
        max_val = config['max'] || 100
        rand(min_val..max_val).to_i
      else
        config.values.sample
      end
    else
      config
    end
  end

  public

  # Add a result from a hyperparameter trial
  # Supports both:
  # - New style: add_result(hyperparams, metrics)
  # - Old style: add_result(tracking_file, experiment_id, metrics = {}, notes: nil)
  def add_result(arg1, arg2 = nil, arg3 = nil, notes: nil)
    # Detect new-style call: both args are hashes
    if arg1.is_a?(Hash) && arg2.is_a?(Hash) && arg3.nil?
      hyperparams = arg1
      metrics = arg2
      @results << { params: hyperparams, metrics: metrics }
      return true
    else
      # Old-style call: add_result(tracking_file, experiment_id, metrics, notes: notes)
      tracking_file = arg1
      experiment_id = arg2
      metrics = arg3 || {}
      
      # For backward compatibility, store in tracking file format
      # Read existing data if file exists
      data = []
      if File.exist?(tracking_file)
        require 'yaml'
        data = YAML.safe_load_file(tracking_file, permitted_classes: [Symbol, Time], aliases: true) || []
      end
      
      # Add new result
      result = {
        'experiment_id' => experiment_id,
        'timestamp' => Time.now.iso8601,
        'metrics' => metrics,
        'notes' => notes
      }
      data << result
      
      # Save back to file
      File.write(tracking_file, data.to_yaml)
      
      # Also add to internal results for consistency
      @results << {
        params: { experiment_id: experiment_id },
        metrics: metrics
      }
      
      return true
    end
  rescue Errno::ENOENT, Errno::EACCES, IOError => e
    # File system errors (permission denied, file not found after check, etc.)
    puts "File error adding result: #{e.message}"
    return false
  rescue Psych::SyntaxError, Psych::BadAlias => e
    # YAML parsing/writing errors
    puts "YAML error adding result: #{e.message}"
    return false
  end

  # Save results to CSV file
  def save_results(filename = 'hyperparameter_results.csv')
    return if @results.empty?

    CSV.open(filename, 'w') do |csv|
      # Extract all unique parameter and metric keys (normalized to strings)
      all_param_keys = @results.map { |r| r[:params].keys.map(&:to_s) }.flatten.uniq.sort
      all_metric_keys = @results.map { |r| r[:metrics].keys.map(&:to_s) }.flatten.uniq.sort

      # Write headers
      headers = all_param_keys + all_metric_keys
      csv << headers

      # Write data rows
      @results.each do |result|
        row = all_param_keys.map { |key| result[:params][key.to_sym] || result[:params][key] } +
              all_metric_keys.map { |key| result[:metrics][key.to_sym] || result[:metrics][key] }
        csv << row
      end
    end
  end

  # Load results from CSV file
  def load_results(filename = 'hyperparameter_results.csv')
    @results = []
    return unless File.exist?(filename)

    data = CSV.read(filename, headers: true)
    return if data.empty?

    # Common metric column names to identify metrics vs params
    known_metrics = ['rmse', 'mae', 'r2', 'mse', 'accuracy', 'precision', 'recall', 
                     'f1', 'auc', 'loss', 'error', 'score']
    
    data.each do |row|
      hash = row.to_h
      # Parse JSON if values look like JSON
      hash.each { |k, v| hash[k] = JSON.parse(v) rescue v }
      
      # Separate params and metrics based on column names
      params_hash = {}
      metrics_hash = {}
      
      hash.each do |key, value|
        # Check if this looks like a metric (case-insensitive)
        if known_metrics.any? { |m| key.to_s.downcase.include?(m) }
          metrics_hash[key.to_sym] = value
        else
          params_hash[key.to_sym] = value
        end
      end
      
      @results << { params: params_hash, metrics: metrics_hash }
    end
  end

  # Get the best result based on a metric
  def best_result(metric, mode = :max)
    return nil if @results.empty?

    if mode == :max
      @results.max_by { |r| r[:metrics][metric.to_sym] }
    else
      @results.min_by { |r| r[:metrics][metric.to_sym] }
    end
  end

  # Get summary statistics of results
  def summary
    return {} if @results.empty?

    summary_stats = {}
    all_metric_keys = @results.map { |r| r[:metrics].keys }.flatten.uniq

    all_metric_keys.each do |key|
      values = @results.map { |r| r[:metrics][key] }.compact
      next if values.empty?

      summary_stats[key] = {
        mean: values.sum.to_f / values.length,
        min: values.min,
        max: values.max,
        std_dev: calculate_std_dev(values)
      }
    end

    summary_stats
  end

  # Calculate standard deviation
  private

  def calculate_std_dev(values)
    return 0 if values.length <= 1

    mean = values.sum.to_f / values.length
    variance = values.map { |v| (v - mean) ** 2 }.sum / (values.length - 1)
    Math.sqrt(variance)
  end
end
