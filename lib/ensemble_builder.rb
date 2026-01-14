require 'csv'
require 'logger'
require_relative 'neural_network_wrapper'

class EnsembleOptimizer
  attr_reader :logger, :nn_wrapper

  def initialize(logger = Logger.new(STDOUT))
    @logger = logger
    @nn_wrapper = nil  # Lazy initialization
  end
  
  # Get or create neural network wrapper
  def neural_network
    @nn_wrapper ||= NeuralNetworkWrapper.new(logger: logger)
  end

  # Stacked generalization (train meta-model on base model predictions)
  def prepare_stacking_features(model_predictions, original_features: [])
    logger.info "Preparing stacked features for meta-learner"
    
    stacked_data = []
    
    # Each row has predictions from multiple base models
    model_preds = model_predictions.first.keys.select { |k| k.to_s.start_with?('model') }
    
    data.each_with_index do |row, idx|
      stacked_row = {}
      
      # Include predictions from all base models
      model_preds.each_with_index do |(model_name, preds), i|
        stacked_row["model_#{i + 1}_pred"] = preds[idx]
      end
      
      # Optionally include original features
      if use_base_features
        row.each { |k, v| stacked_row[k] = v unless k == target_col }
      end
      
      stacked_data << row
    end
    
    logger.info "Stacked features: #{n_models} model predictions + #{data.first.keys.size} original features"
    data
  end

  # Blending (simple averaging with weights)
  def blend_predictions(predictions_hash, weights: nil, method: :inverse_rmse)
    logger.info "Blending predictions using #{method}"
    
    models = predictions_hash.keys
    n_samples = predictions_hash.values.first.size
    
    if weights.nil?
      # Equal weights
      weights = Hash[models.map { |m| [m, 1.0 / models.size] }]
    end
    
    blended_preds = []
    
    n_samples.times do |i|
      weighted_sum = 0.0
      
      models.each do |model|
        weight = weights[model] || (1.0 / models.size)
        pred = predictions_hash[model][i]
        weighted_sum += pred * weight
      end
      
      blended_preds << weighted_sum
    end
    
    blended_preds
  end

  # Stacking (meta-model)
  def stack_predictions(base_predictions, actuals, meta_learner: :ridge, alpha: 0.1)
    logger.info "Training stacking meta-learner (#{meta_learner})"
    
    # Create meta-features (base model predictions)
    n_models = base_predictions.size
    n_samples = base_predictions.first.size
    
    meta_features = Array.new(n_samples) { |i|
      base_predictions.map { |preds| preds[i] }
    }
    
    # User will train meta-learner in Python/DeepNote
    # This generates the structure for stacking
    
    {
      n_base_models: base_predictions.size,
      train_size: base_predictions.first.size,
      meta_features_ready: true
    }
  end

  # Optimize ensemble weights using multiple methods
  def optimize_ensemble_weights(predictions_arrays, actuals, method: :inverse_rmse)
    logger.info "Optimizing ensemble weights using #{method} method"
    
    n_models = predictions_arrays.size
    n_samples = predictions_arrays.first.size
    
    # Calculate individual model RMSEs
    model_rmses = predictions_arrays.map do |preds|
      Math.sqrt(preds.zip(actuals).map { |p, a| (p - a) ** 2 }.sum / n_samples)
    end
    
    logger.info "Individual model RMSEs: #{model_rmses.map { |r| r.round(4) }}"
    
    case method.to_sym
    when :inverse_rmse
      # Weight inversely proportional to RMSE (better models get higher weight)
      weights = model_rmses.map { |rmse| 1.0 / (rmse + 1e-6) }
      normalized_weights = weights.map { |w| w / weights.sum }
      
    when :softmax
      # Softmax weighting (exponential of negative RMSE)
      weights = model_rmses.map { |rmse| Math.exp(-rmse) }
      normalized_weights = weights.map { |w| w / weights.sum }
      
    when :equal
      # Equal weights
      normalized_weights = Array.new(n_models, 1.0 / n_models)
      
    when :grid_search
      # Grid search over weight space (coarse grid for speed)
      best_weights = nil
      best_rmse = Float::INFINITY
      
      # Generate weight combinations (normalized to sum to 1)
      # Note: For n_models > 4, this can be very slow (11^n combinations)
      # Use random_search or inverse_rmse for large ensembles
      step = 0.1
      grid_values = (0..10).to_a
      
      if n_models > 4
        logger.warn "Grid search with #{n_models} models = #{11**n_models} combinations. Consider :inverse_rmse or :softmax instead."
      end
      
      grid_values.repeated_permutation(n_models).each do |weights|
        next if weights.sum == 0
        normalized = weights.map { |w| w / weights.sum.to_f }
        
        # Calculate ensemble predictions
        ensemble_preds = n_samples.times.map do |i|
          predictions_arrays.each_with_index.map { |preds, j| preds[i] * normalized[j] }.sum
        end
        
        # Calculate RMSE
        rmse = Math.sqrt(ensemble_preds.zip(actuals).map { |p, a| (p - a) ** 2 }.sum / n_samples)
        
        if rmse < best_rmse
          best_rmse = rmse
          best_weights = normalized.dup
        end
      end
      
      normalized_weights = best_weights
      
    else
      # Default to inverse RMSE
      weights = model_rmses.map { |rmse| 1.0 / (rmse + 1e-6) }
      normalized_weights = weights.map { |w| w / weights.sum }
    end
    
    # Calculate ensemble RMSE with optimal weights
    ensemble_preds = n_samples.times.map do |i|
      predictions_arrays.each_with_index.map { |preds, j| preds[i] * normalized_weights[j] }.sum
    end
    
    ensemble_rmse = Math.sqrt(ensemble_preds.zip(actuals).map { |p, a| (p - a) ** 2 }.sum / n_samples)
    baseline_rmse = model_rmses.min
    
    logger.info "Optimal weights: #{normalized_weights.map { |w| w.round(4) }}"
    logger.info "Ensemble RMSE: #{ensemble_rmse.round(4)}"
    logger.info "Best individual RMSE: #{baseline_rmse.round(4)}"
    logger.info "Improvement: #{(baseline_rmse - ensemble_rmse).round(4)} (#{((baseline_rmse - ensemble_rmse) / baseline_rmse * 100).round(2)}%)"
    
    {
      optimal_weights: normalized_weights,
      best_rmse: ensemble_rmse,
      baseline_rmse: baseline_rmse,
      improvement: baseline_rmse - ensemble_rmse,
      improvement_pct: (baseline_rmse - ensemble_rmse) / baseline_rmse * 100
    }
  end

  # Voting ensemble (for classification)
  def create_voting_ensemble(predictions_array, weights: nil, method: :soft)
    logger.info "Creating voting ensemble (#{method} voting)"
    
    n_models = predictions_array.size
    n_samples = predictions_array.first.size
    
    weights ||= Array.new(n_models, 1.0 / n_models)
    weight_sum = weights.sum
    normalized_weights = weights.map { |w| w / weight_sum }
    
    if method == :hard
      # Majority vote
      ensemble_preds = []
      n_samples.times do |i|
        votes = Hash.new(0)
        predictions_array.each_with_index do |preds, model_idx|
          vote = preds[i].round # Round to nearest integer for classification
          votes[vote] += normalized_weights[model_idx]
        end
        ensemble_preds << votes.max_by { |_, count| count }.first
      end
    else
      # Weighted average (soft voting)
      ensemble_preds = []
      n_samples.times do |i|
        weighted_sum = predictions_array.each_with_index.map { |preds, idx| preds[i] * normalized_weights[idx] }.sum
        ensemble_preds << weighted_sum
      end
    end
    
    ensemble_preds
  end

  # Stacking ensemble (meta-learner)
  def stack_predictions(base_predictions, meta_learner_params = {})
    logger.info "Stacking predictions from #{base_predictions.size} models"
    
    # This would integrate with sklearn in DeepNote
    # For now, save predictions for stacking
    stacked_data = []
    
    base_predictions.first.size.times do |i|
      row = base_predictions.map { |pred_array| pred_array[i] }
      stacked_data << row
    end
    
    {
      meta_features: stacked_data,
      shape: [stacked_data.size, stacked_data.first&.size || 0]
    }
  end

  # Weighted average ensemble
  def weighted_ensemble(predictions_array, weights)
    logger.info "Creating weighted ensemble (#{predictions_array.first.size} predictions)"
    
    n_samples = predictions_array.first.size
    weight_sum = weights.sum
    normalized_weights = weights.map { |w| w / weight_sum }
    
    ensemble_preds = []
    
    n_samples.times do |i|
      weighted_sum = predictions_array.each_with_index.map { |preds, model_idx| preds[i] * normalized_weights[model_idx] }.sum
      ensemble_preds << weighted_sum
    end
    
    ensemble_preds
  end

  # Stacked generalization (meta-model)
  def prepare_stacking_features(base_predictions)
    logger.info "Preparing stacking features from #{base_predictions.size} base models"
    
    # Combine predictions from multiple models as features
    stacked_features = []
    
    base_predictions.first.size.times do |i|
      row_features = base_predictions.map { |model_preds| model_preds[i] }
      stacked_features << row_features
    end
    
    logger.info "Generated #{stacked_features.size} stacked feature vectors"
    stacked_features
  end

  # Weighted voting ensemble
  def weighted_vote_predictions(model_predictions, weights = nil)
    logger.info "Creating weighted ensemble predictions"
    
    n_models = model_predictions.size
    weights = weights || Array.new(n_models, 1.0 / n_models) # Equal weights by default
    
    # Normalize weights
    weight_sum = weights.sum
    normalized_weights = weights.map { |w| w / weight_sum }
    
    ensemble_predictions = []
    
    model_predictions.first.size.times do |i|
      weighted_pred = 0
      model_predictions.each_with_index do |preds, model_idx|
        weighted_pred += preds[i] * normalized_weights[model_idx]
      end
      ensemble_predictions << weighted_pred
    end
    
    logger.info "Created ensemble predictions from #{model_predictions.size} models"
    ensemble_predictions
  end

  # Stacking meta-learner
  def prepare_stacking_data(model_predictions, actuals)
    logger.info "Preparing stacking data from #{model_predictions.size} base models"
    
    stacking_features = []
    
    model_predictions.first.size.times do |i|
      features = model_predictions.map { |preds| preds[i] }
      stacking_features << features
    end
    
    {
      X: stacking_features,
      y: actuals,
      n_features: model_predictions.size
    }
  end

  # Dynamic weight adjustment based on recent performance
  def calculate_dynamic_weights(model_predictions, actuals, window: 10)
    logger.info "Calculating dynamic weights (window=#{window})"
    
    n_models = model_predictions.size
    weights = Array.new(n_models, 0.0)
    
    # Calculate recent performance for each model
    model_predictions.each_with_index do |preds, model_idx|
      recent_preds = preds.last(window)
      recent_actuals = actuals.last(window)
      
      # Calculate inverse RMSE (better models get higher weight)
      rmse = Math.sqrt(recent_preds.zip(recent_actuals).map { |p, a| (p - a) ** 2 }.sum / recent_preds.size)
      weights[model_idx] = 1.0 / (rmse + 1e-6) # Add small epsilon to avoid division by zero
      
      # Implement guassian function to smooth weights after
    end
    
    # Normalize weights
    total = weights.sum
    normalized_weights = weights.map { |w| w / total }
    
    logger.info "Dynamic weights: #{normalized_weights.map { |w| w.round(4) }}"
    normalized_weights
  end

  # Blending (holdout set for meta-model)
  def create_blending_split(data, blend_ratio: 0.2)
    logger.info "Creating blending split (#{blend_ratio * 100}% for blending)"
    
    split_idx = (data.size * (1 - blend_ratio)).to_i
    
    {
      train: data[0...split_idx],
      blend: data[split_idx..-1]
    }
  end

  # Rank averaging (converts predictions to ranks)
  def rank_average_ensemble(model_predictions)
    logger.info "Creating rank-averaged ensemble"
    
    n_samples = model_predictions.first.size
    
    # Convert each model's predictions to ranks
    ranked_predictions = model_predictions.map do |preds|
      preds.each_with_index.sort_by { |val, _| val }.map.with_index { |(_, orig_idx), rank| [orig_idx, rank] }.sort.map { |_, rank| rank }
    end
    
    # Average ranks
    ensemble_ranks = []
    n_samples.times do |i|
      avg_rank = ranked_predictions.map { |ranks| ranks[i] }.sum / ranked_predictions.size.to_f
      ensemble_ranks << avg_rank
    end
    
    logger.info "Created rank-averaged ensemble"
    ensemble_ranks
  end

  # Diversity analysis (check if models are complementary)
  def analyze_model_diversity(model_predictions, actuals)
    logger.info "Analyzing model diversity"
    
    n_models = model_predictions.size
    correlations = Array.new(n_models) { Array.new(n_models, 0.0) }
    
    # Calculate pairwise error correlations
    n_models.times do |i|
      n_models.times do |j|
        errors_i = model_predictions[i].zip(actuals).map { |p, a| p - a }
        errors_j = model_predictions[j].zip(actuals).map { |p, a| p - a }
        
        # Pearson correlation
        mean_i = errors_i.sum / errors_i.size.to_f
        mean_j = errors_j.sum / errors_j.size.to_f
        
        numerator = errors_i.zip(errors_j).map { |ei, ej| (ei - mean_i) * (ej - mean_j) }.sum
        denom_i = Math.sqrt(errors_i.map { |ei| (ei - mean_i) ** 2 }.sum)
        denom_j = Math.sqrt(errors_j.map { |ej| (ej - mean_j) ** 2 }.sum)
        
        correlations[i][j] = numerator / (denom_i * denom_j + 1e-6)
      end
    end
    
    # Average off-diagonal correlations (diversity measure)
    off_diagonal = []
    n_models.times do |i|
      n_models.times do |j|
        off_diagonal << correlations[i][j] if i != j
      end
    end
    
    avg_correlation = off_diagonal.sum / off_diagonal.size
    
    logger.info "Average error correlation: #{avg_correlation.round(4)} (lower = more diverse)"
    
    {
      correlations: correlations,
      avg_correlation: avg_correlation,
      diversity_score: 1 - avg_correlation.abs
    }
  end

  # Export ensemble configuration
  def export_ensemble_config(models, weights, output_file)
    logger.info "Exporting ensemble configuration to #{output_file}"
    
    CSV.open(output_file, 'w') do |csv|
      csv << ['model_name', 'model_type', 'weight', 'notes']
      
      models.each_with_index do |model, idx|
        csv << [
          model[:name],
          model[:type],
          weights[idx].round(4),
          model[:notes] || ''
        ]
      end
    end
    
    logger.info "✓ Ensemble config saved"
  end
  
  # Train neural network and add to ensemble
  def train_neural_network(data_file, config_file: nil, iterations: 50, target: 'goals')
    logger.info "Training neural network model for ensemble"
    
    # Check dependencies first
    unless neural_network.check_dependencies
      logger.error "Cannot train neural network - missing Python dependencies"
      return nil
    end
    
    # Train model
    results = neural_network.train(
      data_file,
      config_file: config_file,
      iterations: iterations,
      target: target
    )
    
    logger.info "Neural network training complete"
    logger.info "  Best RMSE: #{results[:best_rmse].round(4)}"
    logger.info "  Best R²: #{results[:best_r2].round(4)}"
    
    results
  end
  
  # Get predictions from all models including neural network
  def get_all_predictions(data_file, models: [:rf, :xgb, :elo, :linear, :nn], target: 'goals')
    logger.info "Gathering predictions from #{models.size} models"
    
    predictions_hash = {}
    
    models.each do |model_type|
      case model_type.to_sym
      when :nn, :neural_network
        # Get neural network predictions
        if neural_network.model_info[:trained]
          logger.info "  Loading neural network predictions..."
          nn_preds = neural_network.predict_array(data_file, target: target)
          predictions_hash[:neural_network] = nn_preds
        else
          logger.warn "  Neural network not trained - skipping"
        end
        
      when :rf, :random_forest
        logger.error "  Random Forest model loading not yet implemented"
        logger.error "  Use predict_array parameter or implement load_rf_predictions method"
        raise NotImplementedError, "Random Forest prediction loading not implemented. Pass predictions directly or implement loader."
        
      when :xgb, :xgboost
        logger.error "  XGBoost model loading not yet implemented"
        raise NotImplementedError, "XGBoost prediction loading not implemented. Pass predictions directly or implement loader."
        
      when :elo
        logger.error "  Elo model loading not yet implemented"
        raise NotImplementedError, "Elo prediction loading not implemented. Pass predictions directly or implement loader."
        
      when :linear
        logger.error "  Linear regression model loading not yet implemented"
        raise NotImplementedError, "Linear regression prediction loading not implemented. Pass predictions directly or implement loader."
        
      else
        logger.error "  Unknown model type: #{model_type}"
        raise ArgumentError, "Unknown model type: #{model_type}. Supported: :nn (neural_network)"
      end
    end
    
    predictions_hash
  end
  
  # Create full 6-model ensemble including neural network
  def build_full_ensemble(data_file, actuals, models: [:rf, :xgb, :elo, :linear, :nn])
    logger.info "Building full ensemble with #{models.size} models"
    
    # Get all predictions
    predictions_hash = get_all_predictions(data_file, models: models)
    
    if predictions_hash.empty?
      logger.error "No predictions available - cannot build ensemble"
      return nil
    end
    
    # Convert to arrays for optimization
    model_names = predictions_hash.keys
    predictions_arrays = predictions_hash.values
    
    # Optimize weights
    optimization_result = optimize_ensemble_weights(
      predictions_arrays,
      actuals,
      method: :inverse_rmse
    )
    
    # Create weighted ensemble
    ensemble_predictions = weighted_ensemble(
      predictions_arrays,
      optimization_result[:optimal_weights]
    )
    
    # Calculate ensemble metrics
    ensemble_rmse = Math.sqrt(
      ensemble_predictions.zip(actuals).map { |p, a| (p - a) ** 2 }.sum / actuals.size.to_f
    )
    
    logger.info "✓ Full ensemble complete"
    logger.info "  Ensemble RMSE: #{ensemble_rmse.round(4)}"
    logger.info "  Models: #{model_names.join(', ')}"
    
    {
      predictions: ensemble_predictions,
      weights: Hash[model_names.zip(optimization_result[:optimal_weights])],
      rmse: ensemble_rmse,
      models: model_names,
      optimization: optimization_result
    }
  end
  
  # Ensemble with neural network diversity analysis
  def analyze_nn_contribution(data_file, actuals, base_models: [:rf, :xgb, :elo, :linear])
    logger.info "Analyzing neural network contribution to ensemble"
    
    # Ensemble without NN
    without_nn = get_all_predictions(data_file, models: base_models)
    preds_without = without_nn.values
    
    opt_without = optimize_ensemble_weights(preds_without, actuals)
    ensemble_without = weighted_ensemble(preds_without, opt_without[:optimal_weights])
    rmse_without = Math.sqrt(
      ensemble_without.zip(actuals).map { |p, a| (p - a) ** 2 }.sum / actuals.size.to_f
    )
    
    # Ensemble with NN
    with_nn = get_all_predictions(data_file, models: base_models + [:nn])
    preds_with = with_nn.values
    
    opt_with = optimize_ensemble_weights(preds_with, actuals)
    ensemble_with = weighted_ensemble(preds_with, opt_with[:optimal_weights])
    rmse_with = Math.sqrt(
      ensemble_with.zip(actuals).map { |p, a| (p - a) ** 2 }.sum / actuals.size.to_f
    )
    
    improvement = rmse_without - rmse_with
    improvement_pct = (improvement / rmse_without) * 100
    
    logger.info "Neural network contribution analysis:"
    logger.info "  Ensemble without NN: RMSE = #{rmse_without.round(4)}"
    logger.info "  Ensemble with NN: RMSE = #{rmse_with.round(4)}"
    logger.info "  Improvement: #{improvement.round(4)} (#{improvement_pct.round(2)}%)"
    
    {
      rmse_without_nn: rmse_without,
      rmse_with_nn: rmse_with,
      improvement: improvement,
      improvement_pct: improvement_pct,
      nn_weight: opt_with[:optimal_weights].last
    }
  end
end
