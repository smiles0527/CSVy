require 'sqlite3'
require 'csv'
require 'logger'

class DatabaseManager
  attr_reader :db, :logger

  def initialize(db_path = 'data/csvs.db')
    @logger = Logger.new(STDOUT)
    @logger.level = Logger::INFO
    @db_path = db_path
    @db = nil
  end

  def connect
    logger.info "Connecting to database: #{@db_path}"
    @db = SQLite3::Database.new(@db_path)
    @db.results_as_hash = true
    logger.info "Database connection established"
  end

  def disconnect
    logger.info "Disconnecting from database"
    @db.close if @db
    @db = nil
  end

  def import_csv(file_path, table_name)
    connect unless @db
    logger.info "Importing CSV to table: #{table_name}"
    
    data = CSV.read(file_path, headers: true)
    
    # Drop table if exists
    @db.execute("DROP TABLE IF EXISTS #{table_name}")
    
    # Create table
    create_table(table_name, data.headers)
    
    # Insert data
    insert_data(table_name, data)
    
    logger.info "Successfully imported #{data.length} rows into #{table_name}"
    true
  rescue StandardError => e
    logger.error "Error importing CSV: #{e.message}"
    false
  end

  def export_to_csv(table_name, output_file)
    connect unless @db
    logger.info "Exporting table #{table_name} to CSV: #{output_file}"
    
    rows = @db.execute("SELECT * FROM #{table_name}")
    columns = @db.execute("PRAGMA table_info(#{table_name})").map { |col| col['name'] }
    
    CSV.open(output_file, 'w', write_headers: true, headers: columns) do |csv|
      rows.each do |row|
        csv << columns.map { |col| row[col] }
      end
    end
    
    logger.info "Successfully exported #{rows.length} rows to #{output_file}"
    true
  rescue StandardError => e
    logger.error "Error exporting to CSV: #{e.message}"
    false
  end

  def query(sql)
    connect unless @db
    logger.info "Executing query: #{sql[0..100]}..."
    
    results = @db.execute(sql)
    logger.info "Query returned #{results.length} rows"
    results
  rescue StandardError => e
    logger.error "Query error: #{e.message}"
    []
  end

  def list_tables
    connect unless @db
    tables = @db.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables.map { |t| t['name'] }
  end

  def table_info(table_name)
    connect unless @db
    info = @db.execute("PRAGMA table_info(#{table_name})")
    count = @db.execute("SELECT COUNT(*) as count FROM #{table_name}")[0]['count']
    
    {
      name: table_name,
      columns: info.map { |col| { name: col['name'], type: col['type'] } },
      row_count: count
    }
  end

  def merge_tables(table1, table2, output_table, join_column = nil)
    connect unless @db
    logger.info "Merging tables: #{table1} and #{table2}"
    
    if join_column
      # Inner join
      sql = <<-SQL
        CREATE TABLE #{output_table} AS
        SELECT * FROM #{table1}
        INNER JOIN #{table2} ON #{table1}.#{join_column} = #{table2}.#{join_column}
      SQL
    else
      # Union
      sql = <<-SQL
        CREATE TABLE #{output_table} AS
        SELECT * FROM #{table1}
        UNION ALL
        SELECT * FROM #{table2}
      SQL
    end
    
    @db.execute("DROP TABLE IF EXISTS #{output_table}")
    @db.execute(sql)
    
    logger.info "Tables merged successfully into #{output_table}"
    true
  rescue StandardError => e
    logger.error "Error merging tables: #{e.message}"
    false
  end

  def filter_data(table_name, conditions)
    connect unless @db
    where_clause = conditions.map { |col, val| "#{col} = '#{val}'" }.join(' AND ')
    sql = "SELECT * FROM #{table_name} WHERE #{where_clause}"
    query(sql)
  end

  def aggregate(table_name, group_by, aggregations)
    connect unless @db
    agg_clauses = aggregations.map { |col, func| "#{func}(#{col}) as #{col}_#{func}" }
    sql = "SELECT #{group_by}, #{agg_clauses.join(', ')} FROM #{table_name} GROUP BY #{group_by}"
    query(sql)
  end

  def backup(backup_path)
    connect unless @db
    logger.info "Creating database backup: #{backup_path}"
    
    backup_db = SQLite3::Database.new(backup_path)
    @db.backup('main', backup_db, 'main')
    backup_db.close
    
    logger.info "Backup created successfully"
    true
  rescue StandardError => e
    logger.error "Backup error: #{e.message}"
    false
  end

  def optimize
    connect unless @db
    logger.info "Optimizing database"
    @db.execute('VACUUM')
    @db.execute('ANALYZE')
    logger.info "Database optimized"
  end

  private

  def create_table(table_name, headers)
    # Sanitize column names
    columns = headers.map { |h| sanitize_column_name(h) }
    
    # Create columns as TEXT type (we can be more sophisticated later)
    column_defs = columns.map { |col| "#{col} TEXT" }.join(', ')
    
    sql = "CREATE TABLE #{table_name} (id INTEGER PRIMARY KEY AUTOINCREMENT, #{column_defs})"
    @db.execute(sql)
    
    logger.info "Created table: #{table_name} with columns: #{columns.join(', ')}"
  end

  def insert_data(table_name, data)
    columns = data.headers.map { |h| sanitize_column_name(h) }
    placeholders = (['?'] * columns.length).join(', ')
    
    sql = "INSERT INTO #{table_name} (#{columns.join(', ')}) VALUES (#{placeholders})"
    
    @db.transaction do
      data.each do |row|
        values = columns.map { |col| row[col] }
        @db.execute(sql, values)
      end
    end
  end

  def sanitize_column_name(name)
    # Replace spaces and special characters with underscores
    name.to_s.downcase.gsub(/[^a-z0-9]+/, '_').gsub(/^_|_$/, '')
  end
end
