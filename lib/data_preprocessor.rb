require 'csv'
require 'logger'

class DataPreprocessor
  attr_reader :logger

  def initialize
    @logger = Logger.new(STDOUT)
    @logger.level = Logger::INFO
  end

  # One-hot encoding for categorical variables (removes original column)
  def one_hot_encode(data, column_name)
    logger.info "One-hot encoding column: #{column_name}"
    
    # Get unique values
    unique_values = data.map { |row| row[column_name] }.compact.uniq.sort
    logger.info "Found #{unique_values.length} unique values: #{unique_values.join(', ')}"
    
    # Create new columns for each unique value
    unique_values.each do |value|
      new_column_name = "#{column_name}_#{sanitize_column_name(value).downcase}"
      data.each do |row|
        row[new_column_name] = row[column_name] == value ? '1' : '0'
      end
    end
    
    # Remove original column
    data.each { |row| row.delete(column_name) }
    
    logger.info "One-hot encoding complete. Removed original column and added #{unique_values.length} encoded columns"
    data
  end

  # Label encoding for categorical variables (adds new column, preserves original)
  def label_encode(data, column_name)
    logger.info "Label encoding column: #{column_name}"
    
    unique_values = data.map { |row| row[column_name] }.compact.uniq.sort
    encoding_map = unique_values.each_with_index.to_h
    
    new_column_name = "#{column_name}_label_encoded"
    data.each do |row|
      row[new_column_name] = encoding_map[row[column_name]].to_s if row[column_name]
    end
    
    logger.info "Label encoding complete. Original column preserved, encoded column added"
    logger.info "Encoding map: #{encoding_map}"
    data
  end

  # Normalize: modifies column in place
  def normalize(data, column_name)
    logger.info "Normalizing column: #{column_name}"
    
    values = data.map { |row| row[column_name].to_f }.reject(&:nan?)
    min_val = values.min
    max_val = values.max
    range = max_val - min_val
    
    if range.zero?
      logger.warn "Range is zero, cannot normalize"
      return data
    end
    
    data.each do |row|
      if row[column_name] && !row[column_name].to_s.strip.empty?
        normalized = (row[column_name].to_f - min_val) / range
        row[column_name] = normalized.round(6).to_s
      end
    end
    
    logger.info "Normalization complete. Original column preserved, normalized column added"
    logger.info "Range: min=#{min_val}, max=#{max_val}"
    data
  end

  # Standardize: adds new column, preserves original
  def standardize(data, column_name)
    logger.info "Standardizing column: #{column_name}"
    
    values = data.map { |row| row[column_name].to_f }.reject(&:nan?)
    mean = values.sum / values.size.to_f
    variance = values.map { |v| (v - mean)**2 }.sum / values.size.to_f
    std_dev = Math.sqrt(variance)
    
    if std_dev.zero?
      logger.warn "Standard deviation is zero, cannot standardize"
      return data
    end
    
    new_column_name = "#{column_name}_standardized"
    data.each do |row|
      if row[column_name] && !row[column_name].to_s.strip.empty?
        standardized = (row[column_name].to_f - mean) / std_dev
        row[new_column_name] = standardized.round(6).to_s
      end
    end
    
    logger.info "Standardization complete. Original column preserved, standardized column added"
    logger.info "Stats: mean=#{mean.round(3)}, std=#{std_dev.round(3)}"
    data
  end

  # Bin continuous variables into categories
  def bin_column(data, column_name, bins, labels = nil)
    logger.info "Binning column: #{column_name} into #{bins.length - 1} bins"
    
    labels ||= (0...bins.length - 1).map { |i| "bin_#{i}" }
    
    data.each do |row|
      value = row[column_name].to_f
      bin_index = bins.each_cons(2).find_index { |min, max| value >= min && value < max }
      bin_index ||= bins.length - 2 if value >= bins[-2]
      row[column_name] = labels[bin_index] if bin_index
    end
    
    logger.info "Binning complete"
    data
  end

  # Handle missing values with various strategies
  def handle_missing(data, column_name, strategy: :mean)
    logger.info "Handling missing values in column: #{column_name}, strategy: #{strategy}"
    
    case strategy
    when :mean
      fill_with_mean(data, column_name)
    when :median
      fill_with_median(data, column_name)
    when :mode
      fill_with_mode(data, column_name)
    when :forward_fill
      forward_fill(data, column_name)
    when :backward_fill
      backward_fill(data, column_name)
    when :constant
      fill_with_constant(data, column_name, 0)
    else
      logger.error "Unknown strategy: #{strategy}"
      data
    end
  end

  # Detect and remove outliers using IQR method
  def remove_outliers(data, column_name, method: :iqr)
    logger.info "Removing outliers from column: #{column_name}, method: #{method}"
    
    values = data.map { |row| row[column_name].to_f }.sort
    
    case method
    when :iqr
      q1 = percentile(values, 25)
      q3 = percentile(values, 75)
      iqr = q3 - q1
      lower_bound = q1 - (1.5 * iqr)
      upper_bound = q3 + (1.5 * iqr)
      
      filtered = data.reject do |row|
        val = row[column_name].to_f
        val < lower_bound || val > upper_bound
      end
      
      logger.info "Removed #{data.length - filtered.length} outliers"
      filtered
    when :zscore
      mean = values.sum / values.size.to_f
      std = Math.sqrt(values.map { |v| (v - mean)**2 }.sum / values.size.to_f)
      
      filtered = data.reject do |row|
        zscore = ((row[column_name].to_f - mean) / std).abs
        zscore > 3
      end
      
      logger.info "Removed #{data.length - filtered.length} outliers"
      filtered
    else
      logger.error "Unknown method: #{method}"
      data
    end
  end

  private

  def sanitize_column_name(value)
    value.to_s.downcase.gsub(/[^a-z0-9]+/, '_')
  end

  def fill_with_mean(data, column_name)
    numeric_values = data.map { |row| row[column_name] }
                        .reject { |v| v.nil? || v.to_s.strip.empty? }
                        .map(&:to_f)
    mean = numeric_values.sum / numeric_values.size.to_f
    
    data.each do |row|
      row[column_name] = mean.to_s if row[column_name].nil? || row[column_name].to_s.strip.empty?
    end
    data
  end

  def fill_with_median(data, column_name)
    numeric_values = data.map { |row| row[column_name] }
                        .reject { |v| v.nil? || v.to_s.strip.empty? }
                        .map(&:to_f)
                        .sort
    median = percentile(numeric_values, 50)
    
    data.each do |row|
      row[column_name] = median.to_s if row[column_name].nil? || row[column_name].to_s.strip.empty?
    end
    data
  end

  def fill_with_mode(data, column_name)
    values = data.map { |row| row[column_name] }
                .reject { |v| v.nil? || v.to_s.strip.empty? }
    frequency = values.each_with_object(Hash.new(0)) { |val, hash| hash[val] += 1 }
    mode = frequency.max_by { |_, count| count }[0]
    
    data.each do |row|
      row[column_name] = mode if row[column_name].nil? || row[column_name].to_s.strip.empty?
    end
    data
  end

  def forward_fill(data, column_name)
    last_valid = nil
    data.each do |row|
      if row[column_name].nil? || row[column_name].to_s.strip.empty?
        row[column_name] = last_valid
      else
        last_valid = row[column_name]
      end
    end
    data
  end

  def backward_fill(data, column_name)
    data.reverse.each_cons(2) do |current, previous|
      if previous[column_name].nil? || previous[column_name].to_s.strip.empty?
        previous[column_name] = current[column_name]
      end
    end
    data
  end

  def fill_with_constant(data, column_name, constant)
    data.each do |row|
      row[column_name] = constant.to_s if row[column_name].nil? || row[column_name].to_s.strip.empty?
    end
    data
  end

  def percentile(sorted_array, percentile)
    index = (percentile / 100.0) * (sorted_array.length - 1)
    lower = sorted_array[index.floor]
    upper = sorted_array[index.ceil]
    lower + (upper - lower) * (index - index.floor)
  end
end
