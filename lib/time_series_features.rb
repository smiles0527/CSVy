require 'csv'
require 'logger'

class TimeSeriesFeatures
  attr_reader :logger

  def initialize
    @logger = Logger.new(STDOUT)
    @logger.level = Logger::INFO
  end

  # Calculate rolling window statistics (moving average, sum, etc.)
  def rolling_window(data, column_name, window_size, stat: :mean, group_by: nil)
    logger.info "Calculating rolling #{stat} for '#{column_name}' (window=#{window_size})"
    
    new_column = "#{column_name}_rolling_#{stat}_#{window_size}"
    
    if group_by
      # Calculate per group (e.g., per team)
      groups = data.group_by { |row| row[group_by] }
      
      groups.each do |group_value, group_rows|
        calculate_rolling_for_rows(group_rows, column_name, window_size, stat, new_column)
      end
    else
      # Calculate for entire dataset
      calculate_rolling_for_rows(data, column_name, window_size, stat, new_column)
    end
    
    logger.info "Added column '#{new_column}'"
    data
  end

  # Calculate exponentially weighted moving average (EWMA)
  def ewma(data, column_name, span, group_by: nil)
    logger.info "Calculating EWMA for '#{column_name}' (span=#{span})"
    
    new_column = "#{column_name}_ewma_#{span}"
    alpha = 2.0 / (span + 1)
    
    if group_by
      groups = data.group_by { |row| row[group_by] }
      
      groups.each do |group_value, group_rows|
        calculate_ewma_for_rows(group_rows, column_name, alpha, new_column)
      end
    else
      calculate_ewma_for_rows(data, column_name, alpha, new_column)
    end
    
    logger.info "Added column '#{new_column}' (alpha=#{alpha.round(3)})"
    data
  end

  # Calculate lag features (previous values)
  def lag_features(data, column_name, periods, group_by: nil)
    logger.info "Creating lag features for '#{column_name}' (periods=#{periods.join(', ')})"
    
    periods.each do |lag|
      new_column = "#{column_name}_lag_#{lag}"
      
      if group_by
        groups = data.group_by { |row| row[group_by] }
        
        groups.each do |group_value, group_rows|
          group_rows.each_with_index do |row, idx|
            row[new_column] = idx >= lag ? group_rows[idx - lag][column_name] : nil
          end
        end
      else
        data.each_with_index do |row, idx|
          row[new_column] = idx >= lag ? data[idx - lag][column_name] : nil
        end
      end
      
      logger.info "Added column '#{new_column}'"
    end
    
    data
  end

  # Calculate rate statistics (e.g., goals per game)
  def rate_stat(data, numerator_col, denominator_col, output_name: nil)
    output_name ||= "#{numerator_col}_per_#{denominator_col}"
    logger.info "Calculating rate: #{numerator_col} / #{denominator_col}"
    
    data.each do |row|
      num = row[numerator_col].to_f
      denom = row[denominator_col].to_f
      
      row[output_name] = denom.zero? ? 0 : (num / denom).round(4).to_s
    end
    
    logger.info "Added column '#{output_name}'"
    data
  end

  # Calculate win/loss streaks
  def calculate_streaks(data, result_column, group_by: nil)
    logger.info "Calculating streaks from '#{result_column}'"
    
    new_column = "#{result_column}_streak"
    
    if group_by
      groups = data.group_by { |row| row[group_by] }
      
      groups.each do |group_value, group_rows|
        calculate_streak_for_rows(group_rows, result_column, new_column)
      end
    else
      calculate_streak_for_rows(data, result_column, new_column)
    end
    
    logger.info "Added column '#{new_column}'"
    data
  end

  # Calculate days between events (e.g., rest days)
  def days_between(data, date_column, output_name: 'days_since_last', group_by: nil)
    logger.info "Calculating days between events from '#{date_column}'"
    
    if group_by
      groups = data.group_by { |row| row[group_by] }
      
      groups.each do |group_value, group_rows|
        calculate_days_for_rows(group_rows, date_column, output_name)
      end
    else
      calculate_days_for_rows(data, date_column, output_name)
    end
    
    logger.info "Added column '#{output_name}'"
    data
  end

  # Calculate cumulative statistics
  def cumulative(data, column_name, stat: :sum, group_by: nil)
    logger.info "Calculating cumulative #{stat} for '#{column_name}'"
    
    new_column = "#{column_name}_cumulative_#{stat}"
    
    if group_by
      groups = data.group_by { |row| row[group_by] }
      
      groups.each do |group_value, group_rows|
        calculate_cumulative_for_rows(group_rows, column_name, stat, new_column)
      end
    else
      calculate_cumulative_for_rows(data, column_name, stat, new_column)
    end
    
    logger.info "Added column '#{new_column}'"
    data
  end

  # Calculate rank within group
  def rank_column(data, column_name, group_by: nil, ascending: false)
    logger.info "Ranking column '#{column_name}'"
    
    new_column = "#{column_name}_rank"
    
    if group_by
      groups = data.group_by { |row| row[group_by] }
      
      groups.each do |group_value, group_rows|
        sorted = group_rows.sort_by { |row| row[column_name].to_f }
        sorted.reverse! unless ascending
        sorted.each_with_index { |row, idx| row[new_column] = (idx + 1).to_s }
      end
    else
      sorted = data.sort_by { |row| row[column_name].to_f }
      sorted.reverse! unless ascending
      sorted.each_with_index { |row, idx| row[new_column] = (idx + 1).to_s }
    end
    
    logger.info "Added column '#{new_column}'"
    data
  end

  private

  def calculate_rolling_for_rows(rows, column, window, stat, new_column)
    rows.each_with_index do |row, idx|
      start_idx = [0, idx - window + 1].max
      window_values = rows[start_idx..idx].map { |r| r[column].to_f }
      
      row[new_column] = case stat
      when :mean
        (window_values.sum / window_values.size.to_f).round(4)
      when :sum
        window_values.sum.round(4)
      when :max
        window_values.max.round(4)
      when :min
        window_values.min.round(4)
      when :std
        mean = window_values.sum / window_values.size.to_f
        variance = window_values.map { |v| (v - mean)**2 }.sum / window_values.size.to_f
        Math.sqrt(variance).round(4)
      else
        0
      end.to_s
    end
  end

  def calculate_ewma_for_rows(rows, column, alpha, new_column)
    ewma_value = nil
    
    rows.each do |row|
      value = row[column].to_f
      
      if ewma_value.nil?
        ewma_value = value
      else
        ewma_value = alpha * value + (1 - alpha) * ewma_value
      end
      
      row[new_column] = ewma_value.round(4).to_s
    end
  end

  def calculate_streak_for_rows(rows, result_col, new_column)
    current_streak = 0
    last_result = nil
    
    rows.each do |row|
      result = row[result_col].to_s.strip.upcase
      
      if result == last_result && !result.empty?
        current_streak += 1
      else
        current_streak = 1
        last_result = result
      end
      
      # Positive for wins, negative for losses
      streak_value = result.start_with?('W') ? current_streak : -current_streak
      row[new_column] = streak_value.to_s
    end
  end

  def calculate_days_for_rows(rows, date_col, output_col)
    require 'date'
    
    rows.each_with_index do |row, idx|
      if idx == 0
        row[output_col] = nil
      else
        begin
          current_date = Date.parse(row[date_col].to_s)
          prev_date = Date.parse(rows[idx - 1][date_col].to_s)
          row[output_col] = (current_date - prev_date).to_i.to_s
        rescue
          row[output_col] = nil
        end
      end
    end
  end

  def calculate_cumulative_for_rows(rows, column, stat, new_column)
    cumulative_value = 0
    values = []
    
    rows.each do |row|
      value = row[column].to_f
      
      case stat
      when :sum
        cumulative_value += value
        row[new_column] = cumulative_value.round(4).to_s
      when :mean
        values << value
        row[new_column] = (values.sum / values.size.to_f).round(4).to_s
      when :max
        values << value
        row[new_column] = values.max.round(4).to_s
      when :min
        values << value
        row[new_column] = values.min.round(4).to_s
      end
    end
  end
end
