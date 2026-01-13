require 'rspec'
require_relative '../lib/data_preprocessor'
require 'tempfile'
require 'csv'

RSpec.describe DataPreprocessor do
  let(:preprocessor) { DataPreprocessor.new }
  let(:temp_file) { Tempfile.new(['test', '.csv']) }

  after(:each) do
    temp_file.close
    temp_file.unlink
  end

  describe '#one_hot_encode' do
    before do
      CSV.open(temp_file.path, 'w', write_headers: true, headers: ['name', 'category', 'value']) do |csv|
        csv << ['Alice', 'A', '10']
        csv << ['Bob', 'B', '20']
        csv << ['Charlie', 'A', '30']
      end
    end

    it 'creates binary columns for each unique value' do
      data = CSV.read(temp_file.path, headers: true)
      encoded = preprocessor.one_hot_encode(data, 'category')
      
      expect(encoded.headers).to include('category_a', 'category_b')
      expect(encoded.headers).not_to include('category')
    end

    it 'sets correct binary values' do
      data = CSV.read(temp_file.path, headers: true)
      encoded = preprocessor.one_hot_encode(data, 'category')
      
      expect(encoded[0]['category_a']).to eq('1')
      expect(encoded[0]['category_b']).to eq('0')
    end
  end

  describe '#normalize' do
    before do
      CSV.open(temp_file.path, 'w', write_headers: true, headers: ['name', 'score']) do |csv|
        csv << ['Alice', '10']
        csv << ['Bob', '20']
        csv << ['Charlie', '30']
      end
    end

    it 'normalizes values to 0-1 range' do
      data = CSV.read(temp_file.path, headers: true)
      normalized = preprocessor.normalize(data, 'score')
      
      expect(normalized[0]['score'].to_f).to be_within(0.01).of(0.0)
      expect(normalized[1]['score'].to_f).to be_within(0.01).of(0.5)
      expect(normalized[2]['score'].to_f).to be_within(0.01).of(1.0)
    end
  end

  describe '#handle_missing' do
    before do
      CSV.open(temp_file.path, 'w', write_headers: true, headers: ['name', 'score']) do |csv|
        csv << ['Alice', '10']
        csv << ['Bob', '']
        csv << ['Charlie', '30']
      end
    end

    it 'fills missing values with mean' do
      data = CSV.read(temp_file.path, headers: true)
      filled = preprocessor.handle_missing(data, 'score', strategy: :mean)
      
      expect(filled[1]['score'].to_f).to be_within(0.1).of(20.0)
    end
  end
end
