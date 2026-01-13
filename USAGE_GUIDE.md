# CSVy Usage Guide

## Complete Feature Reference

### 1. Data Cleaning

#### Basic Cleaning
```bash
ruby cli.rb clean data/sample_students_dirty.csv
```

Features:
- Removes empty rows
- Removes duplicate rows
- Handles missing values
- Trims whitespace

#### Advanced Cleaning with Preprocessing
```ruby
require_relative 'lib/csv_cleaner'
require_relative 'lib/data_preprocessor'

# Load and clean data
cleaner = CSVCleaner.new('data/input.csv')
cleaned_data = cleaner.clean_data

# Handle missing values
preprocessor = DataPreprocessor.new
cleaned_data = preprocessor.handle_missing(cleaned_data, 'age', strategy: :mean)

# Save
cleaner.save_to_csv(cleaned_data, 'output.csv')
```

### 2. Data Encoding

#### One-Hot Encoding
```bash
ruby cli.rb encode data/sample_employees.csv department -t onehot -o encoded.csv
```

Before:
```
name,department
Alice,Engineering
Bob,Marketing
```

After:
```
name,department_engineering,department_marketing
Alice,1,0
Bob,0,1
```

#### Label Encoding
```bash
ruby cli.rb encode data/sample_employees.csv department -t label
```

### 3. Data Normalization

#### Min-Max Normalization (0-1 range)
```bash
ruby cli.rb normalize data/sample_employees.csv salary -m minmax
```

#### Z-Score Standardization
```bash
ruby cli.rb normalize data/sample_employees.csv salary -m zscore
```

### 4. Data Validation

#### Validate Data Quality
```bash
ruby cli.rb validate data/sample_employees.csv
```

Output:
```
=== Validation Report ===
File: data/sample_employees.csv
Rows: 10
Columns: 5

Column Types:
  name: string
  age: integer
  salary: float

Issues Found:
  - Found 2 duplicate rows
  - Column 'phone' has 3 missing values
```

#### Generate Statistics
```bash
ruby cli.rb stats data/sample_employees.csv
```

#### Profile Dataset
```bash
ruby cli.rb profile data/sample_employees.csv
```

### 5. Data Merging

#### Concatenate CSV Files
```bash
ruby cli.rb merge file1.csv file2.csv -o merged.csv
```

#### Join on Column (Programmatic)
```ruby
require_relative 'lib/csv_merger'

merger = CSVMerger.new
merged = merger.join_on_column('employees.csv', 'departments.csv', key_column: 'dept_id')
merger.save_to_csv(merged, 'joined.csv')
```

### 6. Database Operations

#### Import CSV to Database
```bash
ruby cli.rb db-import data/sample_employees.csv employees
```

#### Export Table to CSV
```bash
ruby cli.rb db-export employees output.csv
```

#### Query Database
```bash
ruby cli.rb db-query "SELECT * FROM employees WHERE age > 30"
```

#### List Tables
```bash
ruby cli.rb db-tables
```

### 7. Dataframe Operations (Daru)

```ruby
require_relative 'lib/dataframe_handler'

# Load data as dataframe
df = DataframeHandler.new('data/sample_employees.csv')

# Filter rows
filtered = df.filter({'age' => (25..35), 'department' => 'Engineering'})

# Sort
sorted = df.sort_by('salary', order: :descending)

# Select columns
subset = df.select_columns('name', 'salary', 'department')

# Group and aggregate
df.group_by('department')
avg_salary = df.aggregate('salary', operation: :mean)

# Get statistics
stats = df.describe

# Save
df.save_to_csv('output.csv')
```

### 8. Advanced Preprocessing

#### Remove Outliers
```ruby
require_relative 'lib/data_preprocessor'

preprocessor = DataPreprocessor.new
data = CSV.read('data/input.csv', headers: true)

# Using IQR method
clean_data = preprocessor.remove_outliers(data, 'salary', method: :iqr)

# Using Z-score method
clean_data = preprocessor.remove_outliers(data, 'salary', method: :zscore)
```

#### Bin Continuous Variables
```ruby
# Create age groups
bins = [0, 18, 30, 50, 100]
labels = ['child', 'young_adult', 'adult', 'senior']
binned = preprocessor.bin_column(data, 'age', bins, labels)
```

#### Fill Missing Values (Multiple Strategies)
```ruby
# Mean
data = preprocessor.handle_missing(data, 'salary', strategy: :mean)

# Median
data = preprocessor.handle_missing(data, 'age', strategy: :median)

# Mode
data = preprocessor.handle_missing(data, 'category', strategy: :mode)

# Forward fill
data = preprocessor.handle_missing(data, 'value', strategy: :forward_fill)

# Backward fill
data = preprocessor.handle_missing(data, 'value', strategy: :backward_fill)
```

### 9. Data Integrity Checks

```ruby
require_relative 'lib/data_validator'

validator = DataValidator.new

# Define validation rules
rules = {
  'age' => { type: :range, min: 18, max: 100 },
  'email' => { type: :pattern, regex: /\A[\w+\-.]+@[a-z\d\-]+(\.[a-z\d\-]+)*\.[a-z]+\z/i },
  'status' => { type: :enum, values: ['active', 'inactive', 'pending'] },
  'id' => { type: :unique },
  'name' => { type: :not_null }
}

# Check integrity
violations = validator.check_integrity('data/users.csv', rules)
violations.each { |v| puts v }
```

### 10. Complete Workflow Example

```ruby
require_relative 'lib/csv_processor'
require_relative 'lib/csv_cleaner'
require_relative 'lib/data_preprocessor'
require_relative 'lib/data_validator'
require_relative 'lib/database_manager'

# Step 1: Validate input data
validator = DataValidator.new
report = validator.validate('data/raw_data.csv')
puts "Found #{report[:issues].length} issues"

# Step 2: Clean data
cleaner = CSVCleaner.new('data/raw_data.csv')
cleaned = cleaner.clean_data

# Step 3: Preprocess
preprocessor = DataPreprocessor.new

# Handle missing values
cleaned = preprocessor.handle_missing(cleaned, 'age', strategy: :mean)
cleaned = preprocessor.handle_missing(cleaned, 'income', strategy: :median)

# Normalize numeric columns
cleaned = preprocessor.normalize(cleaned, 'age')
cleaned = preprocessor.normalize(cleaned, 'income')

# Encode categorical columns
cleaned = preprocessor.one_hot_encode(cleaned, 'category')
cleaned = preprocessor.label_encode(cleaned, 'status')

# Step 4: Remove outliers
cleaned = preprocessor.remove_outliers(cleaned, 'income', method: :iqr)

# Step 5: Save cleaned data
cleaner.save_to_csv(cleaned, 'data/cleaned_data.csv')

# Step 6: Import to database for further analysis
db = DatabaseManager.new
db.import_csv('data/cleaned_data.csv', 'processed_data')
db.disconnect

puts "Data processing pipeline complete."
```

### 11. Rake Tasks

```bash
# Run tests
rake spec

# Run examples
rake examples

# Clean temporary files
rake clean

# Setup development environment
rake setup

# Run specific example
rake example_validate
```

## Best Practices

1. **Always validate first**: Run `validate` before processing to understand data quality
2. **Handle missing values early**: Address missing data before normalization
3. **Profile your data**: Use `profile` to understand distributions
4. **Test on samples**: Test preprocessing on a small sample first
5. **Backup original data**: Keep original files before transformations
6. **Use database for large datasets**: Import to SQLite for complex queries
7. **Document transformations**: Keep track of preprocessing steps

## Common Use Cases

### Preparing ML Training Data
```bash
# Clean
ruby cli.rb clean raw_data.csv

# Validate
ruby cli.rb validate raw_data_cleaned.csv

# Normalize features
ruby cli.rb normalize raw_data_cleaned.csv feature1
ruby cli.rb normalize raw_data_cleaned.csv feature2

# Encode labels
ruby cli.rb encode raw_data_cleaned.csv label -t label
```

### Exploratory Data Analysis
```bash
# Profile the dataset
ruby cli.rb profile data.csv

# Get statistics
ruby cli.rb stats data.csv

# Validate quality
ruby cli.rb validate data.csv
```

### Data Integration
```bash
# Merge multiple sources
ruby cli.rb merge sales_2025.csv sales_2026.csv -o sales_all.csv

# Import to database
ruby cli.rb db-import sales_all.csv sales

# Query combined data
ruby cli.rb db-query "SELECT * FROM sales WHERE amount > 1000"
```
