require 'daru'
require 'logger'

class DataframeHandler
  attr_reader :df, :logger

  def initialize(file_path = nil)
    @logger = Logger.new(STDOUT)
    @logger.level = Logger::INFO
    @df = file_path ? load_from_csv(file_path) : nil
  end

  def load_from_csv(file_path)
    logger.info "Loading CSV into Daru DataFrame: #{file_path}"
    @df = Daru::DataFrame.from_csv(file_path)
    logger.info "Loaded DataFrame with #{@df.nrows} rows and #{@df.ncols} columns"
    @df
  end

  def filter(conditions)
    logger.info "Applying filter conditions"
    filtered = @df.filter(:row) do |row|
      conditions.all? do |column, value|
        if value.is_a?(Range)
          value.include?(row[column])
        elsif value.is_a?(Proc)
          value.call(row[column])
        else
          row[column] == value
        end
      end
    end
    logger.info "Filter result: #{filtered.nrows} rows"
    filtered
  end

  def sort_by(column, order: :ascending)
    logger.info "Sorting by column: #{column}, order: #{order}"
    @df.sort([column], ascending: order == :ascending ? [true] : [false])
  end

  def select_columns(*columns)
    logger.info "Selecting columns: #{columns.join(', ')}"
    @df[*columns]
  end

  def drop_columns(*columns)
    logger.info "Dropping columns: #{columns.join(', ')}"
    result = @df.dup
    columns.each { |col| result.delete_vector(col) }
    result
  end

  def group_by(column)
    logger.info "Grouping by column: #{column}"
    @df.group_by([column])
  end

  def aggregate(column, operation: :mean)
    logger.info "Aggregating column: #{column}, operation: #{operation}"
    case operation
    when :mean
      @df[column].mean
    when :sum
      @df[column].sum
    when :min
      @df[column].min
    when :max
      @df[column].max
    when :count
      @df[column].size
    when :std
      @df[column].sd
    else
      logger.error "Unknown aggregation operation: #{operation}"
      nil
    end
  end

  def describe
    logger.info "Generating descriptive statistics"
    {
      rows: @df.nrows,
      columns: @df.ncols,
      column_names: @df.vectors.to_a,
      dtypes: @df.vectors.to_a.map { |v| [v, @df[v].type] }.to_h,
      missing_values: @df.vectors.to_a.map { |v| [v, @df[v].missing_values.size] }.to_h
    }
  end

  def head(n = 5)
    @df.head(n)
  end

  def tail(n = 5)
    @df.tail(n)
  end

  def save_to_csv(output_file)
    logger.info "Saving DataFrame to CSV: #{output_file}"
    @df.write_csv(output_file)
    logger.info "Successfully saved to #{output_file}"
  end

  def reshape(operation, options = {})
    case operation
    when :pivot
      pivot_table(options[:index], options[:columns], options[:values])
    when :melt
      melt_dataframe(options[:id_vars], options[:value_vars])
    else
      logger.error "Unknown reshape operation: #{operation}"
      nil
    end
  end

  def apply_function(column, &block)
    logger.info "Applying custom function to column: #{column}"
    @df[column].map(&block)
  end

  private

  def pivot_table(index, columns, values)
    logger.info "Creating pivot table"
    # Daru's pivot_table implementation
    @df.pivot_table(index: index, columns: columns, values: values)
  rescue StandardError => e
    logger.error "Error creating pivot table: #{e.message}"
    nil
  end

  def melt_dataframe(id_vars, value_vars)
    logger.info "Melting dataframe"
    # Custom melt implementation
    result = []
    @df.each_row_with_index do |row, idx|
      id_values = id_vars.map { |v| row[v] }
      value_vars.each do |var|
        result << (id_values + [var, row[var]])
      end
    end
    headers = id_vars + ['variable', 'value']
    Daru::DataFrame.new(result.transpose, order: headers)
  rescue StandardError => e
    logger.error "Error melting dataframe: #{e.message}"
    nil
  end
end
