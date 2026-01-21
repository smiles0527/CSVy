require 'rspec'
require_relative '../lib/model_explainer'
require 'csv'
require 'tempfile'
require 'fileutils'

RSpec.describe ModelExplainer do
  let(:explainer) { ModelExplainer.new }
  let(:temp_dir) { Dir.mktmpdir }
  
  after(:each) do
    FileUtils.rm_rf(temp_dir) if File.exist?(temp_dir)
  end

  describe '#initialize' do
    it 'creates an instance with a logger' do
      expect(explainer).to be_a(ModelExplainer)
      expect(explainer.logger).not_to be_nil
    end
  end

  describe '#debug_features' do
    let(:test_data_path) { File.join(temp_dir, 'test_data.csv') }
    
    before do
      # Create sample CSV data
      CSV.open(test_data_path, 'w') do |csv|
        csv << ['feature1', 'feature2', 'feature3', 'feature4']
        20.times do |i|
          csv << [i, i * 2, 100, i % 2 == 0 ? i : nil]
        end
      end
    end

    it 'detects missing values' do
      results = explainer.debug_features(
        data_path: test_data_path,
        output_dir: temp_dir,
        threshold: 3.0
      )
      
      expect(results[:missing_values]).to be_an(Array)
      expect(results[:missing_values].any? { |mv| mv[:feature] == 'feature4' }).to be true
    end

    it 'detects constant features' do
      results = explainer.debug_features(
        data_path: test_data_path,
        output_dir: temp_dir,
        threshold: 3.0
      )
      
      expect(results[:constant_features]).to be_an(Array)
      constant_feature = results[:constant_features].find { |cf| cf[:feature] == 'feature3' }
      expect(constant_feature).not_to be_nil
    end

    it 'calculates feature quality scores' do
      results = explainer.debug_features(
        data_path: test_data_path,
        output_dir: temp_dir,
        threshold: 3.0
      )
      
      expect(results[:feature_quality_score]).to be_an(Array)
      expect(results[:feature_quality_score].first).to include(:feature, :quality_score, :completeness, :uniqueness)
    end

    it 'generates HTML report' do
      explainer.debug_features(
        data_path: test_data_path,
        output_dir: temp_dir,
        threshold: 3.0
      )
      
      report_path = File.join(temp_dir, 'feature_debug_report.html')
      expect(File.exist?(report_path)).to be true
      
      html_content = File.read(report_path)
      expect(html_content).to include('Feature Debug Report')
    end

    it 'generates CSV output' do
      explainer.debug_features(
        data_path: test_data_path,
        output_dir: temp_dir,
        threshold: 3.0
      )
      
      csv_path = File.join(temp_dir, 'feature_debug.csv')
      expect(File.exist?(csv_path)).to be true
      
      data = CSV.read(csv_path, headers: true)
      expect(data.headers).to include('Feature', 'Quality Score', 'Completeness %', 'Uniqueness %')
    end
  end

  describe '#analyze_errors' do
    let(:predictions_path) { File.join(temp_dir, 'predictions.csv') }
    let(:actuals_path) { File.join(temp_dir, 'actuals.csv') }
    let(:features_path) { File.join(temp_dir, 'features.csv') }
    let(:output_path) { File.join(temp_dir, 'error_analysis.csv') }

    before do
      # Create sample prediction data
      CSV.open(predictions_path, 'w') do |csv|
        csv << ['prediction']
        20.times { |i| csv << [0.5 + (i * 0.05)] }
      end

      # Create sample actual data
      CSV.open(actuals_path, 'w') do |csv|
        csv << ['actual']
        20.times { |i| csv << [0.45 + (i * 0.05)] }
      end

      # Create sample features data
      CSV.open(features_path, 'w') do |csv|
        csv << ['feature1', 'feature2']
        20.times { |i| csv << [i, i * 2] }
      end
    end

    it 'calculates overall statistics' do
      analysis = explainer.analyze_errors(
        predictions_path: predictions_path,
        actuals_path: actuals_path,
        features_path: features_path,
        output_path: output_path
      )

      expect(analysis[:overall]).to include(:mae, :rmse, :mean_error, :count)
      expect(analysis[:overall][:mae]).to be > 0
      expect(analysis[:overall][:count]).to eq(20)
    end

    it 'groups errors by magnitude' do
      analysis = explainer.analyze_errors(
        predictions_path: predictions_path,
        actuals_path: actuals_path,
        features_path: features_path,
        output_path: output_path
      )

      expect(analysis[:by_magnitude]).to be_an(Array)
      expect(analysis[:by_magnitude].size).to eq(5) # 5 bins
      
      bin = analysis[:by_magnitude].first
      expect(bin).to include(:name, :count, :percentage, :avg_error)
    end

    it 'detects systematic bias' do
      analysis = explainer.analyze_errors(
        predictions_path: predictions_path,
        actuals_path: actuals_path,
        features_path: features_path,
        output_path: output_path
      )

      bias = analysis[:systematic_bias]
      expect(bias).to include(:overall_bias, :overestimation_rate, :underestimation_rate, :significant_bias)
      expect(bias[:overestimation_rate] + bias[:underestimation_rate]).to eq(100.0)
    end

    it 'identifies worst predictions' do
      analysis = explainer.analyze_errors(
        predictions_path: predictions_path,
        actuals_path: actuals_path,
        features_path: features_path,
        output_path: output_path
      )

      worst = analysis[:worst_predictions]
      expect(worst).to be_an(Array)
      expect(worst.size).to be <= 20
      expect(worst.first).to include(:index, :predicted, :actual, :error, :absolute_error)
    end

    it 'generates error report HTML' do
      explainer.analyze_errors(
        predictions_path: predictions_path,
        actuals_path: actuals_path,
        features_path: features_path,
        output_path: output_path
      )

      report_path = output_path.sub('.csv', '_report.html')
      expect(File.exist?(report_path)).to be true
      
      html_content = File.read(report_path)
      expect(html_content).to include('Error Analysis Report')
      expect(html_content).to include('Overall Statistics')
    end
  end

  describe 'statistical helpers' do
    it 'calculates standard deviation correctly' do
      values = [1, 2, 3, 4, 5]
      std = explainer.send(:standard_deviation, values)
      
      # Expected std for [1,2,3,4,5] is ~1.414
      expect(std).to be_within(0.01).of(1.414)
    end

    it 'calculates correlation correctly' do
      x = [1, 2, 3, 4, 5]
      y = [2, 4, 6, 8, 10]
      
      corr = explainer.send(:calculate_correlation, x, y)
      expect(corr).to be_within(0.01).of(1.0) # Perfect positive correlation
    end

    it 'handles negative correlation' do
      x = [1, 2, 3, 4, 5]
      y = [5, 4, 3, 2, 1]
      
      corr = explainer.send(:calculate_correlation, x, y)
      expect(corr).to be < 0 # Negative correlation (r ≈ -1.0)
    end
  end

  describe 'validation' do
    it 'raises error for missing model file' do
      expect {
        explainer.send(:validate_inputs, 'nonexistent.pkl', File.join(temp_dir, 'data.csv'))
      }.to raise_error(/Model file not found/)
    end

    it 'raises error for missing data file' do
      model_path = File.join(temp_dir, 'model.pkl')
      File.write(model_path, 'dummy')
      
      expect {
        explainer.send(:validate_inputs, model_path, 'nonexistent.csv')
      }.to raise_error(/Data file not found/)
    end
  end
end
