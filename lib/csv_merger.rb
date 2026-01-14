require 'csv'
require 'logger'

class CSVMerger
  attr_reader :logger

  def initialize
    @logger = Logger.new(STDOUT)
    @logger.level = Logger::INFO
  end

  def merge_files(file1, file2, merge_type: :concat)
    case merge_type
    when :concat
      concatenate(file1, file2)
    when :join
      join_on_column(file1, file2)
    else
      concatenate(file1, file2)
    end
  end

  def concatenate(file1, file2)
    logger.info "Concatenating files"
    
    data1 = CSV.read(file1, headers: true)
    data2 = CSV.read(file2, headers: true)
    
    # Check if headers match
    if data1.headers != data2.headers
      logger.warn "Headers don't match. Using headers from first file."
    end
    
    # Combine the data
    merged_rows = []
    data1.each { |row| merged_rows << row }
    data2.each { |row| merged_rows << row }
    merged = CSV::Table.new(merged_rows)
    
    logger.info "Merged #{data1.length} and #{data2.length} rows = #{merged.length} total rows"
    merged
  end

  def join_on_column(file1, file2, key_column: nil)
    logger.info "Joining files on column: #{key_column}"
    
    data1 = CSV.read(file1, headers: true)
    data2 = CSV.read(file2, headers: true)
    
    unless key_column
      logger.error "Key column not specified for join operation"
      return CSV::Table.new([])
    end
    
    # Create a hash for fast lookup from file2
    data2_hash = {}
    data2.each do |row|
      key = row[key_column]
      data2_hash[key] = row
    end
    
    # Join the data
    all_headers = (data1.headers + data2.headers).uniq
    merged_rows = []
    
    data1.each do |row1|
      key = row1[key_column]
      row2 = data2_hash[key]
      
      if row2
        # Merge rows
        merged_row = CSV::Row.new(all_headers, [])
        all_headers.each do |header|
          merged_row[header] = row1[header] || row2[header]
        end
        merged_rows << merged_row
      else
        # Add row1 with nil values for file2 columns
        merged_row = CSV::Row.new(all_headers, [])
        all_headers.each do |header|
          merged_row[header] = row1[header]
        end
        merged_rows << merged_row
      end
    end
    
    merged = CSV::Table.new(merged_rows)
    logger.info "Join complete. #{merged.length} rows in result"
    merged
  end

  def save_to_csv(data, output_file)
    CSV.open(output_file, 'w', write_headers: true, headers: data.headers) do |csv|
      data.each { |row| csv << row }
    end
    logger.info "Merged data saved to #{output_file}"
  end
end
