require 'csv'
require 'yaml'
require 'logger'
require 'json'

class HyperparameterManager
  attr_reader :logger

  def initialize
    @logger = Logger.new(STDOUT)
    @logger.level = Logger::INFO
  end

  # Generate hyperparameter grid from configuration
  # Supports both discrete arrays and continuous ranges
  def generate_grid(config_file, output_file = nil, sample_size: nil)
    config = YAML.load_file(config_file)
    model_name = config['model_name']
    params = config['hyperparameters']
    
    @logger.info "Generating hyperparameter grid for #{model_name}"
    
    # Process parameters: convert ranges to discrete samples
    processed_params = {}
    param_names = params.keys
    
    params.each do |name, values|
      if values.is_a?(Array)
        # Check if it's a continuous range
        if values.length == 3 && values[2].to_s == 'range'
          # Continuous range: sample n points uniformly
          min, max = values[0].to_f, values[1].to_f
          n_samples = sample_size || 10 # Default 10 samples for continuous
          processed_params[name] = (0...n_samples).map { |i| min + (max - min) * i / (n_samples - 1.0) }
        elsif values.length == 2 && values.all? { |v| v.is_a?(Numeric) }
          # Could be range or discrete - check if they're very different
          if (values[1].to_f - values[0].to_f).abs > 1.0
            # Likely a range, sample points
            n_samples = sample_size || 10
            min, max = values[0].to_f, values[1].to_f
            processed_params[name] = (0...n_samples).map { |i| min + (max - min) * i / (n_samples - 1.0) }
          else
            # Discrete values
            processed_params[name] = values
          end
        else
          # Discrete values
          processed_params[name] = values
        end
      else
        processed_params[name] = [values]
      end
    end
    
    # Generate all combinations
    param_values = processed_params.values
    grid = cartesian_product(param_values)
    
    # Sample if requested
    if sample_size && sample_size < grid.length
      @logger.info "Sampling #{sample_size} configurations from #{grid.length} total"
      grid = grid.sample(sample_size)
    end
    
    @logger.info "Generated #{grid.length} hyperparameter configurations"
    
    # Create CSV with grid
    output_file ||= "#{model_name}_grid_search.csv"
    
    # Ensure directory exists
    output_dir = File.dirname(output_file)
    require 'fileutils'
    FileUtils.mkdir_p(output_dir) unless output_dir == '.' || File.directory?(output_dir)
    
    CSV.open(output_file, 'w') do |csv|
      # Header: param names + experiment tracking columns
      headers = param_names + ['experiment_id', 'rmse', 'mae', 'r2', 'notes', 'timestamp']
      csv << headers
      
      # Write each configuration
      grid.each_with_index do |combo, idx|
        row = combo + [idx + 1, nil, nil, nil, nil, nil]
        csv << row
      end
    end
    
    @logger.info "Grid saved to #{output_file}"
    output_file
  end

  # Bayesian optimization using Gaussian Process surrogate
  # Generates CSV file with configurations (non-interactive)
  def bayesian_optimize(config_file, n_iterations: 20, n_initial: 5, acquisition: 'ei', output_file: nil)
    config = YAML.load_file(config_file)
    model_name = config['model_name']
    params = config['hyperparameters']
    
    @logger.info "Starting Bayesian Optimization for #{model_name}"
    @logger.info "Initial random samples: #{n_initial}, Total iterations: #{n_iterations}"
    
    # Get parameter space
    param_names = params.keys
    
    # Track all configurations to evaluate
    configurations = []
    
    # Phase 1: Initial random exploration
    @logger.info "Phase 1: Random exploration (#{n_initial} samples)"
    n_initial.times do |i|
      config_hash = sample_random_config(params)
      configurations << config_hash
    end
    
    # Phase 2: Bayesian-guided exploration
    @logger.info "Phase 2: Bayesian optimization (#{n_iterations - n_initial} samples)"
    (n_initial...n_iterations).each do |i|
      # Use acquisition function to suggest next point
      # For now, use diversity-based selection (explore unexplored regions)
      config_hash = suggest_next_config(params, configurations, acquisition)
      configurations << config_hash
    end
    
    @logger.info "Generated #{configurations.length} configurations"
    
    # Output to CSV (same format as grid search)
    output_file ||= "#{model_name}_bayesian_optimization.csv"
    
    # Ensure directory exists
    output_dir = File.dirname(output_file)
    require 'fileutils'
    FileUtils.mkdir_p(output_dir) unless output_dir == '.' || File.directory?(output_dir)
    
    CSV.open(output_file, 'w') do |csv|
      # Header: param names + experiment tracking columns
      headers = param_names + ['experiment_id', 'rmse', 'mae', 'r2', 'notes', 'timestamp']
      csv << headers
      
      # Write each configuration
      configurations.each_with_index do |config_hash, idx|
        row = param_names.map { |name| config_hash[name] } + [idx + 1, nil, nil, nil, nil, nil]
        csv << row
      end
    end
    
    @logger.info "Bayesian optimization configurations saved to #{output_file}"
    @logger.info "Next: Train models and record results with 'add-result' command"
    
    output_file
  end

  # Genetic algorithm optimization
  # Generates CSV file with evolved configurations (non-interactive)
  def genetic_algorithm(config_file, population_size: 20, generations: 10, mutation_rate: 0.1, output_file: nil)
    config = YAML.load_file(config_file)
    model_name = config['model_name']
    params = config['hyperparameters']
    
    @logger.info "Starting Genetic Algorithm for #{model_name}"
    @logger.info "Population: #{population_size}, Generations: #{generations}"
    
    # Initialize random population
    population = Array.new(population_size) { sample_random_config(params) }
    all_configurations = population.dup
    
    generations.times do |gen|
      @logger.info "Generation #{gen + 1}/#{generations}: #{population_size} individuals"
      
      # Selection: Keep top 50% (simulate based on diversity)
      # In real GA, this would use actual fitness scores
      survivors = population.sample(population_size / 2)
      
      # Crossover: Breed new individuals
      new_population = survivors.dup
      
      while new_population.length < population_size
        parent1 = survivors.sample
        parent2 = survivors.sample
        child = crossover(parent1, parent2, params)
        
        # Mutation
        child = mutate(child, params, mutation_rate) if rand < mutation_rate
        
        new_population << child
        all_configurations << child
      end
      
      population = new_population
    end
    
    @logger.info "Generated #{all_configurations.length} total configurations"
    
    # Output to CSV
    output_file ||= "#{model_name}_genetic_algorithm.csv"
    
    # Ensure directory exists
    output_dir = File.dirname(output_file)
    require 'fileutils'
    FileUtils.mkdir_p(output_dir) unless output_dir == '.' || File.directory?(output_dir)
    
    param_names = params.keys
    
    CSV.open(output_file, 'w') do |csv|
      headers = param_names + ['experiment_id', 'rmse', 'mae', 'r2', 'notes', 'timestamp']
      csv << headers
      
      all_configurations.each_with_index do |config_hash, idx|
        row = param_names.map { |name| config_hash[name] } + [idx + 1, nil, nil, nil, nil, nil]
        csv << row
      end
    end
    
    @logger.info "Genetic algorithm configurations saved to #{output_file}"
    @logger.info "Next: Train models and record results with 'add-result' command"
    
    output_file
  end

  # Simulated annealing
  # Generates CSV file with configurations (non-interactive)
  def simulated_annealing(config_file, n_iterations: 100, initial_temp: 1.0, cooling_rate: 0.95, output_file: nil)
    config = YAML.load_file(config_file)
    model_name = config['model_name']
    params = config['hyperparameters']
    
    @logger.info "Starting Simulated Annealing for #{model_name}"
    @logger.info "Iterations: #{n_iterations}, Initial temp: #{initial_temp}, Cooling: #{cooling_rate}"
    
    # Start with random configuration
    current_config = sample_random_config(params)
    configurations = [current_config.dup]
    
    temperature = initial_temp
    
    n_iterations.times do |i|
      # Generate neighbor (small random change)
      neighbor_config = neighbor(current_config, params)
      configurations << neighbor_config.dup
      
      # Move to neighbor (with probability based on temperature)
      # For now, always accept (will be filtered by training results)
      current_config = neighbor_config
      
      # Cool down
      temperature *= cooling_rate
    end
    
    @logger.info "Generated #{configurations.length} configurations"
    
    # Output to CSV
    output_file ||= "#{model_name}_simulated_annealing.csv"
    
    # Ensure directory exists
    output_dir = File.dirname(output_file)
    require 'fileutils'
    FileUtils.mkdir_p(output_dir) unless output_dir == '.' || File.directory?(output_dir)
    
    param_names = params.keys
    
    CSV.open(output_file, 'w') do |csv|
      headers = param_names + ['experiment_id', 'rmse', 'mae', 'r2', 'notes', 'timestamp']
      csv << headers
      
      configurations.each_with_index do |config_hash, idx|
        row = param_names.map { |name| config_hash[name] } + [idx + 1, nil, nil, nil, nil, nil]
        csv << row
      end
    end
    
    @logger.info "Simulated annealing configurations saved to #{output_file}"
    @logger.info "Next: Train models and record results with 'add-result' command"
    
    output_file
  end

  # Add experiment result to tracking file
  def add_result(tracking_file, experiment_id, metrics = {}, notes: nil)
    data = CSV.read(tracking_file, headers: true)
    
    # Find row with matching experiment_id
    row = data.find { |r| r['experiment_id'].to_i == experiment_id.to_i }
    
    unless row
      @logger.error "Experiment ID #{experiment_id} not found"
      return false
    end
    
    # Update metrics
    metrics.each do |metric, value|
      row[metric.to_s] = value.to_s if data.headers.include?(metric.to_s)
    end
    
    row['notes'] = notes if notes
    row['timestamp'] = Time.now.strftime("%Y-%m-%d %H:%M:%S")
    
    # Save back
    CSV.open(tracking_file, 'w', write_headers: true, headers: data.headers) do |csv|
      data.each { |r| csv << r }
    end
    
    @logger.info "Updated experiment #{experiment_id} with results"
    true
  end

  # Find best hyperparameters based on metric
  def find_best(tracking_file, metric: 'rmse', ascending: true)
    data = CSV.read(tracking_file, headers: true)
    
    # Filter out rows without this metric
    completed = data.select { |row| row[metric] && !row[metric].to_s.strip.empty? }
    
    if completed.empty?
      @logger.warn "No completed experiments found with metric '#{metric}'"
      return nil
    end
    
    # Sort by metric
    sorted = completed.sort_by { |row| row[metric].to_f }
    sorted.reverse! unless ascending
    
    best = sorted.first
    
    @logger.info "Best #{metric}: #{best[metric]} (Experiment #{best['experiment_id']})"
    
    # Extract hyperparameters (exclude tracking columns)
    tracking_cols = ['experiment_id', 'rmse', 'mae', 'r2', 'notes', 'timestamp']
    hyperparam_cols = data.headers - tracking_cols
    
    best_params = {}
    hyperparam_cols.each { |col| best_params[col] = best[col] }
    
    best_params
  end

  # Export hyperparameters in different formats
  def export_params(config_file, format: :python, output_file: nil)
    config = YAML.load_file(config_file)
    model_name = config['model_name']
    defaults = config['defaults'] || {}
    
    output = case format.to_sym
    when :python
      export_python(defaults, model_name)
    when :json
      require 'json'
      JSON.pretty_generate(defaults)
    when :yaml
      YAML.dump(defaults)
    when :ruby
      export_ruby(defaults, model_name)
    else
      raise ArgumentError, "Unknown format: #{format}"
    end
    
    if output_file
      File.write(output_file, output)
      @logger.info "Exported to #{output_file}"
    else
      puts output
    end
    
    output
  end

  # Generate random search sample
  def random_search(config_file, n_samples, output_file = nil)
    config = YAML.load_file(config_file)
    model_name = config['model_name']
    params = config['hyperparameters']
    
    @logger.info "Generating #{n_samples} random configurations for #{model_name}"
    
    samples = []
    n_samples.times do |i|
      sample = {}
      params.each do |name, values|
        sample[name] = values.sample
      end
      samples << sample
    end
    
    # Write to CSV
    output_file ||= "#{model_name}_random_search.csv"
    
    # Ensure directory exists
    output_dir = File.dirname(output_file)
    require 'fileutils'
    FileUtils.mkdir_p(output_dir) unless output_dir == '.' || File.directory?(output_dir)
    
    CSV.open(output_file, 'w') do |csv|
      headers = params.keys + ['experiment_id', 'rmse', 'mae', 'r2', 'notes', 'timestamp']
      csv << headers
      
      samples.each_with_index do |sample, idx|
        row = sample.values + [idx + 1, nil, nil, nil, nil, nil]
        csv << row
      end
    end
    
    @logger.info "Random search configurations saved to #{output_file}"
    output_file
  end

  # Compare multiple experiments
  def compare_experiments(tracking_file, experiment_ids)
    data = CSV.read(tracking_file, headers: true)
    
    experiments = experiment_ids.map do |id|
      data.find { |row| row['experiment_id'].to_i == id.to_i }
    end.compact
    
    if experiments.empty?
      @logger.error "No experiments found with provided IDs"
      return
    end
    
    # Display comparison table
    puts "\n=== Experiment Comparison ==="
    puts "ID\tRMSE\tMAE\tR2\tNotes"
    puts "-" * 60
    
    experiments.each do |exp|
      puts "#{exp['experiment_id']}\t#{exp['rmse']}\t#{exp['mae']}\t#{exp['r2']}\t#{exp['notes']}"
    end
    
    experiments
  end

  private

  # Sample random configuration
  # Supports both discrete arrays and continuous ranges
  def sample_random_config(params)
    config = {}
    params.each do |name, values|
      if values.is_a?(Array)
        # Check if it's a range specification [min, max, type] or just discrete values
        if values.length == 3 && values[2].to_s == 'range'
          # Continuous range: [min, max, 'range']
          min, max = values[0].to_f, values[1].to_f
          config[name] = min + rand * (max - min)
        elsif values.length == 2 && values.all? { |v| v.is_a?(Numeric) }
          # Continuous range: [min, max] (assume range if both are numeric)
          min, max = values[0].to_f, values[1].to_f
          config[name] = min + rand * (max - min)
        else
          # Discrete values
          config[name] = values.sample
        end
      else
        config[name] = values
      end
    end
    config
  end

  # Suggest next configuration using acquisition function
  def suggest_next_config(params, evaluated_configs, acquisition)
    # Generate candidate points
    candidates = Array.new(100) { sample_random_config(params) }
    
    # Score each candidate based on diversity (distance from evaluated points)
    candidates.map! do |config|
      # Calculate minimum distance to any evaluated config
      min_distance = evaluated_configs.map { |e| distance(config, e) }.min || 1.0
      { config: config, diversity: min_distance }
    end
    
    # Return config with highest diversity (exploration)
    candidates.max_by { |c| c[:diversity] }[:config]
  end

  # Expected Improvement (simple version using distance)
  # Note: This is now handled in suggest_next_config
  def expected_improvement(config, completed, best_score)
    # Calculate average distance to evaluated points
    avg_distance = completed.sum do |e|
      distance(config, e.is_a?(Hash) ? e[:config] : e)
    end / completed.length.to_f
    
    # Balance exploration (distance) and exploitation (predicted improvement)
    exploration_bonus = avg_distance
    exploitation_bonus = (best_score || 1.0) * 0.1 # Small improvement expected
    
    exploration_bonus + exploitation_bonus
  end

  # Calculate distance between two configurations
  def distance(config1, config2)
    # Normalized distance: count of different parameters
    differences = config1.keys.count do |key|
      config1[key] != config2[key]
    end
    differences.to_f / config1.keys.length
  end

  # Genetic algorithm: crossover two configurations
  def crossover(parent1, parent2, params)
    child = {}
    parent1.keys.each do |key|
      # 50% chance from each parent
      child[key] = rand < 0.5 ? parent1[key] : parent2[key]
    end
    child
  end

  # Genetic algorithm: mutate configuration
  def mutate(config, params, mutation_rate)
    mutated = config.dup
    config.keys.each do |key|
      if rand < mutation_rate
        values = params[key]
        mutated[key] = values.is_a?(Array) ? values.sample : values
      end
    end
    mutated
  end

  # Simulated annealing: generate neighbor configuration
  def neighbor(config, params)
    neighbor_config = config.dup
    
    # Change 1-2 random parameters
    keys_to_change = config.keys.sample(rand(1..2))
    
    keys_to_change.each do |key|
      values = params[key]
      if values.is_a?(Array)
        # Pick different value
        current_idx = values.index(config[key])
        if current_idx
          # Pick adjacent value if possible
          if values.length > 1
            offset = [-1, 1].sample
            new_idx = (current_idx + offset) % values.length
            neighbor_config[key] = values[new_idx]
          end
        else
          neighbor_config[key] = values.sample
        end
      end
    end
    
    neighbor_config
  end

  def cartesian_product(arrays)
    return [[]] if arrays.empty?
    
    first = arrays[0]
    rest = cartesian_product(arrays[1..-1])
    
    result = []
    first.each do |elem|
      rest.each do |combo|
        result << [elem] + combo
      end
    end
    result
  end

  def export_python(params, model_name)
    lines = ["# Hyperparameters for #{model_name}", ""]
    lines << "params = {"
    params.each do |key, value|
      formatted_value = value.is_a?(String) ? "'#{value}'" : value
      lines << "    '#{key}': #{formatted_value},"
    end
    lines << "}"
    lines.join("\n")
  end

  def export_ruby(params, model_name)
    lines = ["# Hyperparameters for #{model_name}", ""]
    lines << "params = {"
    params.each do |key, value|
      formatted_value = value.is_a?(String) ? "'#{value}'" : value
      lines << "  '#{key}' => #{formatted_value},"
    end
    lines << "}"
    lines.join("\n")
  end
end
