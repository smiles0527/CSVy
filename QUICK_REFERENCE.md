# CSVy Quick Reference Card

## Installation
```bash
bundle install
```

## Most Used Commands

### Data Cleaning
```bash
ruby cli.rb clean data/input.csv
```

### Data Validation
```bash
ruby cli.rb validate data/input.csv    # Quality check
ruby cli.rb stats data/input.csv        # Statistics
ruby cli.rb profile data/input.csv      # Detailed profile
```

### Data Preprocessing
```bash
# Encoding
ruby cli.rb encode data/input.csv column_name -t onehot
ruby cli.rb encode data/input.csv column_name -t label

# Normalization
ruby cli.rb normalize data/input.csv column_name -m minmax
ruby cli.rb normalize data/input.csv column_name -m zscore
```

### File Operations
```bash
ruby cli.rb info data/input.csv                           # Show info
ruby cli.rb merge data/file1.csv data/file2.csv -o out.csv  # Merge
```

### Database
```bash
ruby cli.rb db-import data/input.csv table_name    # Import
ruby cli.rb db-export table_name output.csv        # Export
ruby cli.rb db-query "SELECT * FROM table"         # Query
ruby cli.rb db-tables                              # List tables
```

## Rake Tasks
```bash
rake spec           # Run tests
rake examples       # Run all examples
rake clean          # Clean temp files
rake setup          # Setup environment
```

## Quick Workflows

### Clean & Validate
```bash
ruby cli.rb clean data/raw.csv
ruby cli.rb validate data/raw_cleaned.csv
```

### Prepare for ML
```bash
ruby cli.rb clean data/raw.csv
ruby cli.rb encode data/raw_cleaned.csv category -t onehot
ruby cli.rb normalize data/raw_cleaned_onehot_encoded.csv value
```

### Analyze Dataset
```bash
ruby cli.rb profile data/input.csv
ruby cli.rb stats data/input.csv
```

## Programmatic Usage

### Complete Pipeline
```ruby
require_relative 'lib/csv_cleaner'
require_relative 'lib/data_preprocessor'

# Clean
cleaner = CSVCleaner.new('input.csv')
data = cleaner.clean_data

# Preprocess
preprocessor = DataPreprocessor.new
data = preprocessor.handle_missing(data, 'age', strategy: :mean)
data = preprocessor.normalize(data, 'salary')
data = preprocessor.one_hot_encode(data, 'category')

# Save
cleaner.save_to_csv(data, 'output.csv')
```

### Dataframe Operations
```ruby
require_relative 'lib/dataframe_handler'

df = DataframeHandler.new('input.csv')
filtered = df.filter({'age' => (25..35)})
sorted = df.sort_by('salary', order: :descending)
df.save_to_csv('output.csv')
```

## Sample Data Files

- `data/sample_employees.csv` - Employee records
- `data/sample_products.csv` - Product catalog
- `data/sample_weather.csv` - Weather data
- `data/sample_students_dirty.csv` - Data with quality issues

## Help

```bash
ruby cli.rb help                # Show all commands
ruby cli.rb help COMMAND        # Show command help
```

## Documentation

- `README.md` - Project overview
- `USAGE_GUIDE.md` - Comprehensive guide with examples
- `QUICK_REFERENCE.md` - This file
