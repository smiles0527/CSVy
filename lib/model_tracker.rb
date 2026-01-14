require 'csv'
require 'logger'
require 'json'

class ModelTracker
  attr_reader :logger

  def initialize(logger = Logger.new(STDOUT))
    @logger = logger
  end

  # Track model performance across experiments
  def track_model_performance(experiment_file, output_file = nil)
    logger.info "Tracking model performance from #{experiment_file}"
    
    data = CSV.read(experiment_file, headers: true)
    
    # Filter completed experiments
    completed = data.select { |row| row['rmse'] && !row['rmse'].to_s.strip.empty? }
    
    if completed.empty?
      logger.warn "No completed experiments found"
      return nil
    end
    
    # Calculate statistics
    rmses = completed.map { |r| r['rmse'].to_f }
    maes = completed.map { |r| r['mae'].to_f }.compact
    r2s = completed.map { |r| r['r2'].to_f }.compact
    
    stats = {
      total_experiments: data.size,
      completed_experiments: completed.size,
      completion_rate: (completed.size.to_f / data.size * 100).round(2),
      rmse: {
        best: rmses.min,
        worst: rmses.max,
        mean: rmses.sum / rmses.size,
        median: rmses.sort[rmses.size / 2],
        std_dev: calculate_std_dev(rmses)
      }
    }
    
    stats[:mae] = {
      best: maes.min,
      worst: maes.max,
      mean: maes.sum / maes.size,
      median: maes.sort[maes.size / 2]
    } if maes.any?
    
    stats[:r2] = {
      best: r2s.max,
      worst: r2s.min,
      mean: r2s.sum / r2s.size,
      median: r2s.sort[r2s.size / 2]
    } if r2s.any?
    
    # Find top performers
    top_10 = completed.sort_by { |r| r['rmse'].to_f }.first(10)
    stats[:top_10_experiments] = top_10.map { |r| r['experiment_id'].to_i }
    
    # Save to JSON
    output_file ||= experiment_file.gsub('.csv', '_performance.json')
    File.write(output_file, JSON.pretty_generate(stats))
    
    logger.info "Performance tracking saved to #{output_file}"
    logger.info "Best RMSE: #{stats[:rmse][:best].round(4)}"
    logger.info "Mean RMSE: #{stats[:rmse][:mean].round(4)}"
    
    stats
  end

  # Compare multiple model tracking files
  def compare_models(tracking_files, output_file = nil)
    logger.info "Comparing #{tracking_files.size} models"
    
    comparisons = {}
    
    tracking_files.each do |file|
      model_name = File.basename(file, '.csv')
      data = CSV.read(file, headers: true)
      completed = data.select { |r| r['rmse'] && !r['rmse'].to_s.strip.empty? }
      
      next if completed.empty?
      
      rmses = completed.map { |r| r['rmse'].to_f }
      best = completed.min_by { |r| r['rmse'].to_f }
      
      comparisons[model_name] = {
        total_configs: data.size,
        completed: completed.size,
        best_rmse: rmses.min,
        mean_rmse: rmses.sum / rmses.size,
        best_experiment_id: best['experiment_id'].to_i,
        best_params: best.to_h.reject { |k, v| ['experiment_id', 'rmse', 'mae', 'r2', 'notes', 'timestamp'].include?(k) }
      }
    end
    
    # Find best overall model
    best_model = comparisons.min_by { |_, stats| stats[:best_rmse] }
    
    comparison_report = {
      models: comparisons,
      best_model: best_model[0],
      best_rmse: best_model[1][:best_rmse],
      comparison_date: Time.now.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save report
    output_file ||= 'model_comparison.json'
    File.write(output_file, JSON.pretty_generate(comparison_report))
    
    logger.info "Model comparison saved to #{output_file}"
    logger.info "Best model: #{best_model[0]} (RMSE: #{best_model[1][:best_rmse].round(4)})"
    
    comparison_report
  end

  # Generate performance report
  def generate_performance_report(tracking_file, output_file = nil)
    logger.info "Generating performance report"
    
    data = CSV.read(tracking_file, headers: true)
    completed = data.select { |r| r['rmse'] && !r['rmse'].to_s.strip.empty? }
    
    return nil if completed.empty?
    
    # Sort by RMSE
    sorted = completed.sort_by { |r| r['rmse'].to_f }
    
    output_file ||= tracking_file.gsub('.csv', '_report.txt')
    
    File.open(output_file, 'w') do |f|
      f.puts "=" * 60
      f.puts "MODEL PERFORMANCE REPORT"
      f.puts "=" * 60
      f.puts "Generated: #{Time.now.strftime("%Y-%m-%d %H:%M:%S")}"
      f.puts "Total experiments: #{data.size}"
      f.puts "Completed: #{completed.size} (#{(completed.size.to_f / data.size * 100).round(1)}%)"
      f.puts ""
      
      # Top 10 experiments
      f.puts "TOP 10 EXPERIMENTS:"
      f.puts "-" * 60
      sorted.first(10).each_with_index do |exp, idx|
        f.puts "#{idx + 1}. Experiment #{exp['experiment_id']}"
        f.puts "   RMSE: #{exp['rmse']}"
        f.puts "   MAE: #{exp['mae']}" if exp['mae']
        f.puts "   R²: #{exp['r2']}" if exp['r2']
        f.puts ""
      end
      
      # Statistics
      rmses = completed.map { |r| r['rmse'].to_f }
      f.puts "RMSE STATISTICS:"
      f.puts "-" * 60
      f.puts "Best: #{rmses.min.round(4)}"
      f.puts "Worst: #{rmses.max.round(4)}"
      f.puts "Mean: #{(rmses.sum / rmses.size).round(4)}"
      f.puts "Median: #{rmses.sort[rmses.size / 2].round(4)}"
      f.puts "Std Dev: #{calculate_std_dev(rmses).round(4)}"
    end
    
    logger.info "Performance report saved to #{output_file}"
    output_file
  end

  private

  # Automated model selection based on validation metrics
  def select_best_model(tracking_files, metric: :rmse, criteria: :best)
    logger.info "Selecting best model based on #{metric} (#{criteria} criteria)"
    
    model_scores = {}
    
    tracking_files.each do |file|
      model_name = File.basename(file, '.csv')
      data = CSV.read(file, headers: true)
      completed = data.select { |r| r[metric.to_s] && !r[metric.to_s].to_s.strip.empty? }
      
      next if completed.empty?
      
      scores = completed.map { |r| r[metric.to_s].to_f }
      
      model_scores[model_name] = case criteria
      when :best
        scores.min # Lower is better for RMSE/MAE
      when :mean
        scores.sum / scores.size
      when :median
        scores.sort[scores.size / 2]
      else
        scores.min
      end
    end
    
    return nil if model_scores.empty?
    
    # Select best (lowest for RMSE/MAE, highest for R2)
    best_model = if metric.to_s == 'r2'
      model_scores.max_by { |_, score| score }
    else
      model_scores.min_by { |_, score| score }
    end
    
    {
      best_model: best_model[0],
      best_score: best_model[1],
      all_scores: model_scores,
      selection_criteria: "#{criteria} #{metric}"
    }
  end

  def calculate_std_dev(values)
    mean = values.sum / values.size.to_f
    variance = values.map { |v| (v - mean) ** 2 }.sum / values.size
    Math.sqrt(variance)
  end
end
