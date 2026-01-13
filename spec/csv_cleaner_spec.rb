require 'rspec'
require_relative '../lib/csv_cleaner'
require 'tempfile'
require 'csv'

RSpec.describe CSVCleaner do
  let(:temp_file) { Tempfile.new(['test', '.csv']) }

  after(:each) do
    temp_file.close
    temp_file.unlink
  end

  describe '#clean_data' do
    context 'with a CSV containing empty rows' do
      before do
        CSV.open(temp_file.path, 'w', write_headers: true, headers: ['name', 'age', 'city']) do |csv|
          csv << ['Alice', '30', 'New York']
          csv << ['', '', ''] # empty row
          csv << ['Bob', '25', 'Los Angeles']
        end
      end

      it 'removes empty rows' do
        cleaner = CSVCleaner.new(temp_file.path)
        cleaned = cleaner.clean_data
        expect(cleaned.length).to eq(2)
      end
    end

    context 'with a CSV containing duplicates' do
      before do
        CSV.open(temp_file.path, 'w', write_headers: true, headers: ['name', 'age']) do |csv|
          csv << ['Alice', '30']
          csv << ['Bob', '25']
          csv << ['Alice', '30'] # duplicate
        end
      end

      it 'removes duplicate rows' do
        cleaner = CSVCleaner.new(temp_file.path)
        cleaned = cleaner.clean_data
        expect(cleaned.length).to eq(2)
      end
    end

    context 'with whitespace in fields' do
      before do
        CSV.open(temp_file.path, 'w', write_headers: true, headers: ['name', 'age']) do |csv|
          csv << ['  Alice  ', '  30  ']
          csv << ['Bob  ', '25']
        end
      end

      it 'trims whitespace from fields' do
        cleaner = CSVCleaner.new(temp_file.path)
        cleaned = cleaner.clean_data
        expect(cleaned[0]['name']).to eq('Alice')
        expect(cleaned[0]['age']).to eq('30')
      end
    end
  end

  describe '#normalize_column' do
    before do
      CSV.open(temp_file.path, 'w', write_headers: true, headers: ['name', 'score']) do |csv|
        csv << ['Alice', '10']
        csv << ['Bob', '20']
        csv << ['Charlie', '30']
      end
    end

    it 'normalizes numeric column to 0-1 range' do
      cleaner = CSVCleaner.new(temp_file.path)
      normalized = cleaner.normalize_column(cleaner.data, 'score')
      
      expect(normalized[0]['score'].to_f).to be_within(0.01).of(0.0)
      expect(normalized[1]['score'].to_f).to be_within(0.01).of(0.5)
      expect(normalized[2]['score'].to_f).to be_within(0.01).of(1.0)
    end
  end
end
