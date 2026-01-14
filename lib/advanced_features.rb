require 'csv'
require 'date'
require 'logger'

# Advanced features for CSV processing
# This module provides enhanced functionality for working with CSV files

class CSVAdvancedFeatures
  def initialize(logger = nil)
    @logger = logger || Logger.new($stdout)
  end

  # Safely parse dates with improved error handling
  # 
  # @param date_string [String] The date string to parse
  # @param format [String] Optional date format string
  # @return [Date, nil] Parsed Date object or nil if parsing fails
  def parse_date_safe(date_string, format = nil)
    return nil if date_string.nil? || date_string.to_s.strip.empty?

    begin
      if format
        Date.strptime(date_string.to_s.strip, format)
      else
        # Try common date formats in order of likelihood
        Date.iso8601(date_string.to_s.strip)
      end
    rescue ArgumentError => e
      @logger.warn("Failed to parse date '#{date_string}': #{e.message}")
      nil
    rescue StandardError => e
      @logger.error("Unexpected error parsing date '#{date_string}': #{e.message}")
      nil
    end
  end
end
