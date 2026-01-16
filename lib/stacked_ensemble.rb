require 'csv'
require 'logger'
require 'json'
require 'tempfile'

# STACKED ENSEMBLE (Meta-Learner)
# 
# Instead of fixed weights (inverse RMSE), train a model to learn
# WHEN to trust each base model. This captures:
# - Model 1 good at low-scoring games
# - Model 2 good at home games
# - Model 3 good at back-to-back situations
# 
# Expected improvement: 5-10% RMSE over simple averaging
class StackedEnsemble
  attr_reader :logger, :meta_model_type
  
  def initialize(logger: Logger.new($stdout), meta_model: 'ridge')
    @logger = logger
    @meta_model_type = meta_model # ridge, lasso, xgboost, neural_net
    @meta_model_path = nil
    @base_model_names = []
  end
  
  def train_meta_model(predictions_dir, actual_values_file, target_col: 'goals')
    """
    Train the meta-model (Level 2) on base model predictions (Level 1)
    
    Flow:
    1. Load all base model predictions (6 models × N games)
    2. Stack into feature matrix [model1_pred, model2_pred, ..., model6_pred]
    3. Train meta-model: actual_goals = f(all_base_predictions)
    4. Meta-model learns which model to trust in each situation
    
    Args:
      predictions_dir: Directory with CSV files (model1_predictions.csv, model2_predictions.csv, etc.)
      actual_values_file: CSV with actual target values
      target_col: Name of target column
    """
    logger.info "=" * 70
    logger.info "TRAINING STACKED ENSEMBLE META-MODEL"
    logger.info "=" * 70
    
    # Load actual values
    actual_data = CSV.read(actual_values_file, headers: true)
    actuals = actual_data.map { |row| row[target_col].to_f }
    
    # Load all base model predictions
    prediction_files = Dir.glob(File.join(predictions_dir, "*_predictions.csv"))
    
    if prediction_files.empty?
      raise "No prediction files found in #{predictions_dir}"
    end
    
    logger.info "Found #{prediction_files.size} base models:"
    
    # Stack predictions into feature matrix
    stacked_features = []
    @base_model_names = []
    
    prediction_files.each do |pred_file|
      model_name = File.basename(pred_file, '_predictions.csv')
      @base_model_names << model_name
      
      preds = CSV.read(pred_file, headers: true).map { |row| row['prediction'].to_f }
      
      if preds.size != actuals.size
        raise "Prediction count mismatch: #{model_name} has #{preds.size}, expected #{actuals.size}"
      end
      
      # Add as column in stacked features
      if stacked_features.empty?
        preds.each_with_index { |p, i| stacked_features[i] = [p] }
      else
        preds.each_with_index { |p, i| stacked_features[i] << p }
      end
      
      # Calculate base model RMSE
      rmse = Math.sqrt(preds.zip(actuals).sum { |p, a| (p - a) ** 2 } / preds.size)
      logger.info "  #{model_name}: RMSE #{rmse.round(3)}"
    end
    
    logger.info ""
    logger.info "Stacked feature matrix: #{stacked_features.size} samples × #{@base_model_names.size} models"
    
    # Write stacked features to temp CSV for Python
    temp_train = Tempfile.new(['stacked_train', '.csv'])
    CSV.open(temp_train.path, 'w') do |csv|
      csv << @base_model_names + ['actual']
      stacked_features.zip(actuals).each do |features, actual|
        csv << features + [actual]
      end
    end
    
    # Train meta-model in Python
    logger.info "Training meta-model (#{@meta_model_type})..."
    
    @meta_model_path = "models/meta_model_#{@meta_model_type}.pkl"
    
    python_script = create_meta_model_training_script(
      temp_train.path,
      @meta_model_path,
      @base_model_names
    )
    
    script_file = Tempfile.new(['train_meta', '.py'])
    File.write(script_file.path, python_script)
    
    # Execute Python script
    require 'open3'
    python_exe = ENV['PYTHON_PATH'] || 'python'
    stdout, stderr, status = Open3.capture3(python_exe, script_file.path)
    output = stdout + stderr
    
    unless status.success?
      logger.error "Meta-model training failed:"
      logger.error output
      raise "Meta-model training failed"
    end
    
    logger.info output
    
    # Parse results
    if output =~ /Meta-model RMSE: ([\d.]+)/
      meta_rmse = $1.to_f
      logger.info ""
      logger.info "=" * 70
      logger.info "STACKING COMPLETE"
      logger.info "  Meta-model RMSE: #{meta_rmse.round(3)}"
      logger.info "  Meta-model saved: #{@meta_model_path}"
      logger.info "=" * 70
      
      return {
        meta_rmse: meta_rmse,
        model_path: @meta_model_path,
        base_models: @base_model_names
      }
    else
      raise "Could not parse meta-model results"
    end
  ensure
    temp_train&.close
    temp_train&.unlink
    script_file&.close
    script_file&.unlink
  end
  
  def predict_with_meta_model(predictions_dir, output_file: 'stacked_predictions.csv')
    """
    Use trained meta-model to generate ensemble predictions
    
    Args:
      predictions_dir: Directory with base model predictions
      output_file: Where to save stacked predictions
    """
    unless @meta_model_path && File.exist?(@meta_model_path)
      raise "Meta-model not trained yet. Call train_meta_model first."
    end
    
    logger.info "Generating stacked ensemble predictions..."
    
    # Load base model predictions
    prediction_files = Dir.glob(File.join(predictions_dir, "*_predictions.csv"))
    
    stacked_features = []
    prediction_files.each do |pred_file|
      preds = CSV.read(pred_file, headers: true).map { |row| row['prediction'].to_f }
      
      if stacked_features.empty?
        preds.each_with_index { |p, i| stacked_features[i] = [p] }
      else
        preds.each_with_index { |p, i| stacked_features[i] << p }
      end
    end
    
    # Write to temp CSV
    temp_pred = Tempfile.new(['stacked_pred', '.csv'])
    
    # Load base_model_names from pickle if not set
    if @base_model_names.nil? || @base_model_names.empty?
      require 'open3'
      script = Tempfile.new(['load_names', '.py'])
      script.write(<<~PYTHON)
        import pickle
        with open(#{@meta_model_path.to_json}, 'rb') as f:
            meta = pickle.load(f)
        print(','.join(meta['base_models']))
      PYTHON
      script.close
      
      python_exe = ENV['PYTHON_PATH'] || 'python'
      stdout, stderr, status = Open3.capture3(python_exe, script.path)
      if status.success?
        @base_model_names = stdout.strip.split(',')
      else
        raise "Failed to load base_model_names: #{stderr}"
      end
      script.unlink
    end
    
    CSV.open(temp_pred.path, 'w') do |csv|
      csv << @base_model_names
      stacked_features.each { |features| csv << features }
    end
    
    # Create prediction script
    python_script = create_meta_model_prediction_script(
      temp_pred.path,
      @meta_model_path,
      output_file
    )
    
    script_file = Tempfile.new(['predict_meta', '.py'])
    File.write(script_file.path, python_script)
    
    # Execute
    require 'open3'
    python_exe = ENV['PYTHON_PATH'] || 'python'
    stdout, stderr, status = Open3.capture3(python_exe, script_file.path)
    output = stdout + stderr
    
    unless status.success?
      logger.error "Meta-model prediction failed:"
      logger.error output
      raise "Meta-model prediction failed"
    end
    
    logger.info "✓ Stacked predictions saved: #{output_file}"
    
    output_file
  ensure
    temp_pred&.close
    temp_pred&.unlink
    script_file&.close
    script_file&.unlink
  end
  
  def analyze_meta_model_weights
    """
    Analyze which base models the meta-model trusts most
    
    For linear meta-models (Ridge/Lasso), shows learned coefficients
    For tree-based (XGBoost), shows feature importance
    """
    unless @meta_model_path && File.exist?(@meta_model_path)
      raise "Meta-model not trained yet"
    end
    
    logger.info "Analyzing meta-model learned weights..."
    
    python_script = create_weight_analysis_script(@meta_model_path, @base_model_names)
    
    script_file = Tempfile.new(['analyze_weights', '.py'])
    File.write(script_file.path, python_script)
    
    require 'open3'
    python_exe = ENV['PYTHON_PATH'] || 'python'
    stdout, stderr, status = Open3.capture3(python_exe, script_file.path)
    output = stdout + stderr
    
    logger.info output
    
    output
  ensure
    script_file&.close
    script_file&.unlink
  end
  
  private
  
  def create_meta_model_training_script(train_csv, model_output, base_model_names)
    """
    Generate Python script to train meta-model
    
    Meta-model types:
    - ridge: Ridge regression (L2 regularization, linear combination)
    - lasso: Lasso regression (L1 regularization, feature selection)
    - xgboost: XGBoost (non-linear, learns interactions)
    - neural_net: Small NN (most flexible, may overfit)
    """
    
    base_models_json = base_model_names.to_json
    
    case @meta_model_type
    when 'ridge', 'lasso'
      create_linear_meta_script(train_csv, model_output, base_models_json)
    when 'xgboost'
      create_xgboost_meta_script(train_csv, model_output, base_models_json)
    when 'neural_net'
      create_nn_meta_script(train_csv, model_output, base_models_json)
    else
      raise "Unknown meta-model type: #{@meta_model_type}"
    end
  end
  
  def create_linear_meta_script(train_csv, model_output, base_models_json)
    <<~PYTHON
      import pandas as pd
      import numpy as np
      import pickle
      from sklearn.linear_model import Ridge, Lasso
      from sklearn.model_selection import cross_val_score
      from sklearn.metrics import mean_squared_error
      
      # Load stacked training data
      df = pd.read_csv(#{train_csv.to_json})
      
      base_models = #{base_models_json}
      X = df[base_models].values
      y = df['actual'].values
      
      print(f"Training meta-model on {len(X)} samples, {X.shape[1]} base models")
      
      # Train meta-model with cross-validation to find best alpha
      best_alpha = None
      best_score = float('inf')
      
      for alpha in [0.001, 0.01, 0.1, 1.0, 10.0]:
          if #{@meta_model_type.to_json} == 'ridge':
              model = Ridge(alpha=alpha)
          else:
              model = Lasso(alpha=alpha)
          
          scores = cross_val_score(model, X, y, cv=5, 
                                   scoring='neg_mean_squared_error')
          rmse = np.sqrt(-scores.mean())
          
          if rmse < best_score:
              best_score = rmse
              best_alpha = alpha
      
      print(f"Best alpha: {best_alpha}, CV RMSE: {best_score:.3f}")
      
      # Train final model on all data
      if #{@meta_model_type.to_json} == 'ridge':
          meta_model = Ridge(alpha=best_alpha)
      else:
          meta_model = Lasso(alpha=best_alpha)
      
      meta_model.fit(X, y)
      
      # Evaluate
      y_pred = meta_model.predict(X)
      rmse = np.sqrt(mean_squared_error(y, y_pred))
      
      print(f"\\nMeta-model RMSE: {rmse:.3f}")
      print(f"\\nLearned weights:")
      for name, coef in zip(base_models, meta_model.coef_):
          print(f"  {name}: {coef:.3f}")
      
      # Save model
      with open(#{model_output.to_json}, 'wb') as f:
          pickle.dump({
              'model': meta_model,
              'base_models': base_models,
              'type': #{@meta_model_type.to_json}
          }, f)
      
      print(f"\\nMeta-model saved to {#{model_output.to_json}}")
    PYTHON
  end
  
  def create_xgboost_meta_script(train_csv, model_output, base_models_json)
    <<~PYTHON
      import pandas as pd
      import numpy as np
      import pickle
      from xgboost import XGBRegressor
      from sklearn.model_selection import cross_val_score
      from sklearn.metrics import mean_squared_error
      
      df = pd.read_csv(#{train_csv.to_json})
      
      base_models = #{base_models_json}
      X = df[base_models].values
      y = df['actual'].values
      
      print(f"Training XGBoost meta-model on {len(X)} samples")
      
      # XGBoost with limited depth to prevent overfitting
      meta_model = XGBRegressor(
          n_estimators=100,
          max_depth=3,
          learning_rate=0.1,
          subsample=0.8,
          random_state=42
      )
      
      meta_model.fit(X, y)
      
      y_pred = meta_model.predict(X)
      rmse = np.sqrt(mean_squared_error(y, y_pred))
      
      print(f"\\nMeta-model RMSE: {rmse:.3f}")
      print(f"\\nFeature importance:")
      for name, importance in zip(base_models, meta_model.feature_importances_):
          print(f"  {name}: {importance:.3f}")
      
      with open(#{model_output.to_json}, 'wb') as f:
          pickle.dump({
              'model': meta_model,
              'base_models': base_models,
              'type': 'xgboost'
          }, f)
      
      print(f"\\nMeta-model saved to {#{model_output.to_json}}")
    PYTHON
  end
  
  def create_nn_meta_script(train_csv, model_output, base_models_json)
    <<~PYTHON
      import pandas as pd
      import numpy as np
      import pickle
      from sklearn.preprocessing import StandardScaler
      from sklearn.metrics import mean_squared_error
      import tensorflow as tf
      from tensorflow import keras
      from tensorflow.keras import layers
      
      df = pd.read_csv(#{train_csv.to_json})
      
      base_models = #{base_models_json}
      X = df[base_models].values
      y = df['actual'].values
      
      print(f"Training Neural Network meta-model on {len(X)} samples")
      
      # Scale inputs
      scaler = StandardScaler()
      X_scaled = scaler.fit_transform(X)
      
      # Small NN to prevent overfitting
      model = keras.Sequential([
          layers.Dense(8, activation='relu', input_shape=(X.shape[1],)),
          layers.Dropout(0.3),
          layers.Dense(4, activation='relu'),
          layers.Dense(1, activation='linear')
      ])
      
      model.compile(optimizer='adam', loss='mse')
      
      early_stop = keras.callbacks.EarlyStopping(
          monitor='loss', patience=20, restore_best_weights=True
      )
      
      model.fit(X_scaled, y, epochs=200, batch_size=16, 
                callbacks=[early_stop], verbose=0)
      
      y_pred = model.predict(X_scaled, verbose=0).flatten()
      rmse = np.sqrt(mean_squared_error(y, y_pred))
      
      print(f"\\nMeta-model RMSE: {rmse:.3f}")
      
      # Save Keras model separately, pickle the rest
      model_file = #{model_output.to_json}.replace('.pkl', '_model.keras')
      model.save(model_file)
      
      with open(#{model_output.to_json}, 'wb') as f:
          pickle.dump({
              'model_file': model_file,
              'scaler': scaler,
              'base_models': base_models,
              'type': 'neural_net'
          }, f)
      
      print(f"\\nMeta-model saved to {#{model_output.to_json}}")
      print(f"Keras model saved to {model_file}")
    PYTHON
  end
  
  def create_meta_model_prediction_script(input_csv, model_path, output_csv)
    <<~PYTHON
      import pandas as pd
      import numpy as np
      import pickle
      
      # Load meta-model
      with open(#{model_path.to_json}, 'rb') as f:
          meta = pickle.load(f)
      
      base_models = meta['base_models']
      model_type = meta['type']
      
      # Load model based on type
      if model_type == 'neural_net':
          from tensorflow import keras
          model = keras.models.load_model(meta['model_file'])
      else:
          model = meta['model']
      
      # Load predictions
      df = pd.read_csv(#{input_csv.to_json})
      X = df[base_models].values
      
      # Predict
      if model_type == 'neural_net':
          scaler = meta['scaler']
          X_scaled = scaler.transform(X)
          predictions = model.predict(X_scaled, verbose=0).flatten()
      else:
          predictions = model.predict(X)
      
      # Save
      output_df = pd.DataFrame({
          'prediction': predictions
      })
      output_df.to_csv(#{output_csv.to_json}, index=False)
      
      print(f"Stacked predictions saved: {#{output_csv.to_json}}")
    PYTHON
  end
  
  def create_weight_analysis_script(model_path, base_models)
    <<~PYTHON
      import pickle
      import numpy as np
      
      with open(#{model_path.to_json}, 'rb') as f:
          meta = pickle.load(f)
      
      base_models = meta['base_models']
      model_type = meta['type']
      
      # Load model based on type
      if model_type == 'neural_net':
          from tensorflow import keras
          model = keras.models.load_model(meta['model_file'])
      else:
          model = meta['model']
      
      print("=" * 60)
      print("META-MODEL ANALYSIS")
      print("=" * 60)
      print(f"Type: {model_type}")
      print(f"Base models: {len(base_models)}")
      print()
      
      if model_type in ['ridge', 'lasso']:
          print("Learned Coefficients (how much to trust each model):")
          coeffs = model.coef_
          total = np.sum(np.abs(coeffs))
          
          for name, coef in sorted(zip(base_models, coeffs), key=lambda x: abs(x[1]), reverse=True):
              pct = abs(coef) / total * 100
              print(f"  {name:20s}: {coef:+.3f} ({pct:.1f}%)")
          
          print(f"\\nIntercept: {model.intercept_:.3f}")
      
      elif model_type == 'xgboost':
          print("Feature Importance (which models matter most):")
          importances = model.feature_importances_
          
          for name, importance in sorted(zip(base_models, importances), key=lambda x: x[1], reverse=True):
              pct = importance * 100
              print(f"  {name:20s}: {importance:.3f} ({pct:.1f}%)")
      
      else:
          print("Neural network meta-model (weights are complex)")
          print("Use model.summary() to see architecture")
      
      print("=" * 60)
    PYTHON
  end
end
