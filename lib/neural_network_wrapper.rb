require 'json'
require 'logger'
require 'tempfile'
require 'csv'

class NeuralNetworkWrapper
  attr_reader :logger, :python_script, :model_path, :scaler_path
  
  def initialize(logger: Logger.new(STDOUT), python_exe: 'python')
    @logger = logger
    @python_exe = python_exe
    @python_script = File.join(Dir.pwd, 'scripts', 'train_neural_network.py')
    @model_path = File.join(Dir.pwd, 'models', 'best_nn_model.keras')
    @scaler_path = File.join(Dir.pwd, 'models', 'scaler.pkl')
    
    # Check if Python script exists
    unless File.exist?(@python_script)
      raise "Neural network training script not found: #{@python_script}"
    end
  end
  
  # Train neural network with hyperparameter search
  def train(data_file, config_file: nil, iterations: 50, target: 'goals', output_csv: nil)
    logger.info "Starting neural network training"
    logger.info "  Data file: #{data_file}"
    logger.info "  Iterations: #{iterations}"
    logger.info "  Target column: #{target}"
    
    unless File.exist?(data_file)
      raise "Data file not found: #{data_file}"
    end
    
    # Set default config if not specified
    config_file ||= File.join(Dir.pwd, 'config', 'hyperparams', 'model6_neural_network.yaml')
    output_csv ||= 'model6_neural_network_results.csv'
    
    # Build command
    cmd = [
      @python_exe,
      @python_script,
      data_file,
      '--target', target,
      '--config', config_file,
      '--search', iterations.to_s,
      '--output', output_csv
    ].join(' ')
    
    logger.info "Executing: #{cmd}"
    
    # Run training (this may take a while)
    success = system(cmd)
    
    unless success
      raise "Neural network training failed. Check Python dependencies: pip install tensorflow scikit-learn pandas numpy pyyaml"
    end
    
    logger.info "✓ Neural network training complete"
    logger.info "  Results saved to: #{output_csv}"
    logger.info "  Best model saved to: #{@model_path}"
    
    # Load and return results
    load_results(output_csv)
  end
  
  # Make predictions using trained model
  def predict(data_file, target: 'goals')
    logger.info "Making predictions with neural network"
    
    unless File.exist?(data_file)
      raise "Data file not found: #{data_file}"
    end
    
    unless File.exist?(@model_path)
      raise "Trained model not found: #{@model_path}. Run train() first."
    end
    
    # Create temporary Python script for prediction
    predict_script = create_prediction_script
    
    begin
      logger.info "Executing prediction: #{@python_exe} #{predict_script} #{data_file} #{target}"
      
      # Use IO.popen to avoid shell injection
      output = IO.popen([@python_exe, predict_script, data_file, target], err: [:child, :out], &:read)
      
      if $?.success?
        predictions = JSON.parse(output)
        logger.info "✓ Predictions complete: #{predictions['predictions'].size} samples"
        predictions
      else
        raise "Prediction failed: #{output}"
      end
    ensure
      File.delete(predict_script) if File.exist?(predict_script)
    end
  end
  
  # Get predictions as array (for ensemble integration)
  def predict_array(data_file, target: 'goals')
    result = predict(data_file, target: target)
    result['predictions']
  end
  
  # Load hyperparameter search results
  def load_results(csv_file)
    unless File.exist?(csv_file)
      raise "Results file not found: #{csv_file}"
    end
    
    results = CSV.read(csv_file, headers: true).map(&:to_h)
    
    logger.info "Loaded #{results.size} experiment results"
    
    # Find best result
    best = results.min_by { |r| r['rmse'].to_f }
    
    logger.info "Best result:"
    logger.info "  RMSE: #{best['rmse']}"
    logger.info "  R²: #{best['r2']}"
    logger.info "  Architecture: #{best['layer1_units']}-#{best['layer2_units']}-#{best['layer3_units']}"
    
    {
      results: results,
      best: best,
      best_rmse: best['rmse'].to_f,
      best_r2: best['r2'].to_f
    }
  end
  
  # Check if Python dependencies are installed
  def check_dependencies
    logger.info "Checking Python dependencies..."
    
    packages = ['tensorflow', 'sklearn', 'pandas', 'numpy', 'yaml']
    missing = []
    
    packages.each do |pkg|
      cmd = "#{@python_exe} -c \"import #{pkg}\""
      unless system(cmd, out: File::NULL, err: File::NULL)
        missing << pkg
      end
    end
    
    if missing.empty?
      logger.info "✓ All Python dependencies installed"
      true
    else
      logger.error "✗ Missing Python packages: #{missing.join(', ')}"
      logger.error "  Install with: pip install #{missing.join(' ')}"
      false
    end
  end
  
  # Get model metrics
  def model_info
    unless File.exist?(@model_path)
      return { trained: false, message: "No trained model found" }
    end
    
    {
      trained: true,
      model_path: @model_path,
      scaler_path: @scaler_path,
      model_size: File.size(@model_path),
      modified: File.mtime(@model_path)
    }
  end
  
  private
  
  # Create temporary prediction script
  def create_prediction_script
    script_content = <<~PYTHON
      import sys
      import json
      import pickle
      import pandas as pd
      import numpy as np
      import tensorflow as tf
      from tensorflow import keras
      
      def predict(data_file, target_col):
          # Load model and scaler
          model = keras.models.load_model('#{@model_path}')
          with open('#{@scaler_path}', 'rb') as f:
              scaler = pickle.load(f)
          
          # Load data
          df = pd.read_csv(data_file)
          if target_col in df.columns:
              X = df.drop(columns=[target_col])
              y = df[target_col]
          else:
              X = df
              y = None
          
          # Make predictions
          X_scaled = scaler.transform(X)
          predictions = model.predict(X_scaled, verbose=0).flatten()
          
          # Calculate metrics if target available
          result = {'predictions': predictions.tolist()}
          
          if y is not None:
              from sklearn.metrics import mean_squared_error, r2_score
              rmse = np.sqrt(mean_squared_error(y, predictions))
              r2 = r2_score(y, predictions)
              mae = np.mean(np.abs(y - predictions))
              
              result['metrics'] = {
                  'rmse': float(rmse),
                  'r2': float(r2),
                  'mae': float(mae)
              }
          
          return result
      
      if __name__ == '__main__':
          data_file = sys.argv[1]
          target_col = sys.argv[2] if len(sys.argv) > 2 else 'goals'
          result = predict(data_file, target_col)
          print(json.dumps(result))
    PYTHON
    
    temp_file = Tempfile.new(['predict_nn', '.py'])
    temp_file.write(script_content)
    temp_file.close
    temp_file.path
  end
end
