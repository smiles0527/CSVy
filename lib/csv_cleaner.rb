require 'csv'
require 'logger'
require 'set'

class CSVCleaner
  attr_reader :file_path, :data, :logger

  def initialize(file_path)
    @file_path = file_path
    @data = []
    @logger = Logger.new(STDOUT)
    @logger.level = Logger::INFO
    load_data
  end

  def load_data
    @data = CSV.read(@file_path, headers: true)
    logger.info "Loaded #{@data.length} rows from #{@file_path}"
  end

  def clean_data
    logger.info "Starting data cleaning process"
    
    # Remove empty rows
    non_empty_data = @data.reject do |row|
      row.fields.all? { |field| field.nil? || field.to_s.strip.empty? }
    end
    
    # Trim whitespace
    non_empty_data.each do |row|
      @data.headers.each do |header|
        if row[header]
          row[header] = row[header].to_s.strip
        end
      end
    end
    
    # Remove duplicates
    seen = Set.new
    result_rows = []
    
    non_empty_data.each do |row|
      row_key = row.fields.join('|')
      unless seen.include?(row_key)
        seen.add(row_key)
        result_rows << row
      end
    end
    
    result_data = CSV::Table.new(result_rows)
    logger.info "Cleaning complete: removed empty rows, trimmed whitespace, and removed duplicates"
    result_data
  end
  
  def identify_numeric_columns
    return [] if @data.empty?
    
    @data.headers.select do |header|
      sample_values = @data.map { |row| row[header] }.compact.reject { |v| v.to_s.strip.empty? }.first(10)
      sample_values.all? { |v| numeric?(v) }
    end
  end
  
  def calculate_column_mean(column_name)
    values = @data.map { |row| row[column_name] }.compact.reject { |v| v.to_s.strip.empty? }.map(&:to_f)
    return 0 if values.empty?
    (values.sum / values.size).round(2).to_s
  end
  
  def handle_outlier(column_name, value)
    values = @data.map { |row| row[column_name] }.compact.reject { |v| v.to_s.strip.empty? }.map(&:to_f).sort
    return value.round(2).to_s if values.size < 4
    
    # IQR method for outlier detection
    q1 = values[values.size / 4]
    q3 = values[(values.size * 3) / 4]
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    if value < lower_bound || value > upper_bound
      median = values[values.size / 2]
      median.round(2).to_s
    else
      value.round(2).to_s
    end
  end

  def remove_empty_rows(data)
    data.reject do |row|
      row.fields.all?(&:nil?) || row.fields.all? { |f| f.to_s.strip.empty? }
    end
  end

  def handle_missing_values(data, strategy: :remove)
    case strategy
    when :remove
      # Remove rows with any missing values
      data.reject { |row| row.fields.any?(&:nil?) }
    when :fill_zero
      # Fill missing numeric values with 0
      data.each do |row|
        row.headers.each do |header|
          row[header] = '0' if row[header].nil? && numeric?(row[header])
        end
      end
      data
    else
      data
    end
  end

  def remove_duplicates(data)
    seen = Set.new
    data.select do |row|
      row_key = row.fields.join('|')
      if seen.include?(row_key)
        false
      else
        seen.add(row_key)
        true
      end
    end
  end

  def trim_whitespace(data)
    data.each do |row|
      row.headers.each do |header|
        row[header] = row[header].to_s.strip unless row[header].nil?
      end
    end
    data
  end

  def normalize_column(data, column_name)
    # Normalize numeric column to 0-1 range
    values = data.map { |row| row[column_name].to_f }
    min_val = values.min
    max_val = values.max
    range = max_val - min_val
    
    return data if range.zero?
    
    data.each do |row|
      normalized = (row[column_name].to_f - min_val) / range
      row[column_name] = normalized.to_s
    end
    
    data
  end

  def save_to_csv(data, output_file)
    CSV.open(output_file, 'w', write_headers: true, headers: data.headers) do |csv|
      data.each { |row| csv << row }
    end
    logger.info "Data saved to #{output_file}"
  end

  private

  def numeric?(value)
    true if Float(value) rescue false
  end
end
