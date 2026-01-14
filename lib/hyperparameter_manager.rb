require 'csv'
require 'json'

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
  def random_search(iterations = 10)
    iterations.times do |i|
      sample = {}
      @params.each do |param_name, config|
        sample[param_name] = sample_param(config)
      end
      yield(sample, i) if block_given?
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
        config['min'] = config['min'] || 0
        config['max'] = config['max'] || 100
        rand(config['min']..config['max']).to_i
      else
        config.values.sample
      end
    else
      config
    end
  end

  public

  # Add a result from a hyperparameter trial
  def add_result(hyperparams, metrics)
    @results << { params: hyperparams, metrics: metrics }
  end

  # Save results to CSV file
  def save_results(filename = 'hyperparameter_results.csv')
    return if @results.empty?

    CSV.open(filename, 'w') do |csv|
      # Extract all unique parameter and metric keys
      all_param_keys = @results.map { |r| r[:params].keys }.flatten.uniq.sort
      all_metric_keys = @results.map { |r| r[:metrics].keys }.flatten.uniq.sort

      # Write headers
      headers = all_param_keys + all_metric_keys
      csv << headers

      # Write data rows
      @results.each do |result|
        row = all_param_keys.map { |key| result[:params][key] } +
              all_metric_keys.map { |key| result[:metrics][key] }
        csv << row
      end
    end
  end

  # Load results from CSV file
  def load_results(filename = 'hyperparameter_results.csv')
    @results = []
    return unless File.exist?(filename)

    CSV.read(filename, headers: true).each do |row|
      hash = row.to_h
      # Parse JSON if values look like JSON
      hash.each { |k, v| hash[k] = JSON.parse(v) rescue v }
      @results << hash
    end
  end

  # Get the best result based on a metric
  def best_result(metric, mode = :max)
    return nil if @results.empty?

    best = @results.max_by { |r| r[:metrics][metric.to_sym] }
    best if mode == :max

    best = @results.min_by { |r| r[:metrics][metric.to_sym] }
    best if mode == :min
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
