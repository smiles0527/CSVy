# CSVy - CSV Organizer for Hockey Analytics

A professional Ruby-based CSV processing and organization tool designed for hockey statistics, player data, and game analysis workflows.

## Features

### Core Operations
- **Clean CSV Data**: Remove empty rows, duplicates, and handle missing values
- **Merge CSV Files**: Concatenate or join multiple CSV files
- **File Information**: Display detailed information about CSV file structure

### Data Preprocessing

#### Encoding Methods
- **One-Hot Encoding**: Convert categorical variables to binary columns
  - Creates n binary columns for n categories
  - Example: `Color=['Red','Blue']` → `Color_Red=[1,0], Color_Blue=[0,1]`
  
- **Label Encoding**: Encode categorical variables as integers
  - Maps categories to integers: `['Low','Med','High']` → `[0, 1, 2]`

#### Normalization Methods
- **Min-Max Scaling** (0-1 range)
  ```
  x_normalized = (x - min) / (max - min)
  ```
  - Scales all values to [0, 1] range
  - Example: [10, 20, 30] → [0.0, 0.5, 1.0]
  
- **Z-Score Standardization**
  ```
  x_standardized = (x - μ) / σ
  where μ = mean, σ = standard deviation
  ```
  - Centers data around 0 with std dev of 1
  - Example: [10, 20, 30] → [-1.22, 0.0, 1.22]

#### Data Quality
- **Missing Value Handling**: Multiple strategies
  - `mean`: Fill with column average
  - `median`: Fill with middle value (robust to outliers)
  - `mode`: Fill with most frequent value
  - `forward_fill`: Propagate last valid value
  - `backward_fill`: Use next valid value
  
- **Outlier Detection**
  - **IQR Method**: `outlier if x < Q1 - 1.5×IQR or x > Q3 + 1.5×IQR`
    - Q1 = 25th percentile, Q3 = 75th percentile
    - IQR = Q3 - Q1 (Interquartile Range)
  - **Z-Score Method**: `outlier if |z| > 3`
    - Removes values more than 3 standard deviations from mean

- **Binning**: Convert continuous → categorical
  - Example: Age [0-18, 19-35, 36-60, 61+] → ['child', 'young', 'adult', 'senior']

### Data Analysis & Validation

#### Statistical Measures
- **Descriptive Statistics**:
  ```
  Mean (μ) = Σx / n
  Median = middle value when sorted
  Mode = most frequent value
  Std Dev (σ) = √(Σ(x - μ)² / n)
  Variance (σ²) = Σ(x - μ)² / n
  ```
  
- **Quartiles & Percentiles**:
  - Q1 (25th percentile): 25% of data below
  - Q2 (50th percentile): Median
  - Q3 (75th percentile): 75% of data below
  
- **Data Validation**: Comprehensive quality checks
  - Empty rows detection
  - Duplicate identification
  - Missing value analysis
  - Data type inference
  
- **Data Profiling**: Column-level metrics
  - Cardinality: `unique_values / total_values`
  - Missing rate: `missing_count / total_count × 100%`
  - Distribution analysis
  
- **Integrity Checks**: Custom validation rules
  - Range validation: `min ≤ value ≤ max`
  - Pattern matching: Regex validation
  - Uniqueness constraints
  - Enum validation

### Advanced Features
- **Daru Integration**: Dataframe operations (filter, sort, group, aggregate)
- **SQLite Database**: Import/export CSV data, run SQL queries
- **CLI Interface**: User-friendly command-line interface
- **Comprehensive Testing**: Full RSpec test suite with 90%+ coverage
- **Rake Tasks**: Automated workflows and examples

## Tech Stack

- **Language**: Ruby
- **CSV Handling**: Ruby CSV (Standard Library)
- **Data Manipulation**: Daru (Optional, for advanced operations)
- **CLI Framework**: Thor
- **Testing**: RSpec
- **Environment Management**: Dotenv
- **Database** (Optional): SQLite3

## Installation

### Prerequisites

- Ruby 2.7 or higher
- Bundler

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/smiles0527/CSVy.git
cd CSVy
```

2. **Install dependencies**:
```bash
gem install bundler
bundle install
```

3. **Try with sample hockey data**:
```bash
ruby cli.rb info data/sample_employees.csv
ruby cli.rb validate data/sample_students_dirty.csv
```

## Usage

### Quick Start

```bash
# Install dependencies
bundle install

# Run examples to see features in action
rake examples

# Or run individual examples
rake example_validate
rake example_clean
```

### Command Line Interface

#### Data Cleaning
```bash
# Clean player statistics with missing values
ruby cli.rb clean data/sample_students_dirty.csv
```

#### Data Preprocessing
```bash
# One-hot encode player positions
ruby cli.rb encode data/sample_employees.csv position -t onehot

# Normalize goals scored across season
ruby cli.rb normalize data/sample_employees.csv goals -m minmax

# Standardize plus/minus ratings
ruby cli.rb normalize data/sample_employees.csv plus_minus -m zscore
```

#### Data Validation & Analysis
```bash
# Validate player data quality
ruby cli.rb validate data/sample_employees.csv

# Generate statistics for team performance
ruby cli.rb stats data/sample_products.csv

# Profile game results dataset
ruby cli.rb profile data/sample_weather.csv
```

#### File Operations
```bash
# Merge player stats from multiple seasons
ruby cli.rb merge season1.csv season2.csv -o combined_stats.csv

# Display team standings information
ruby cli.rb info data/sample_products.csv
```

#### Database Operations
```bash
# Import player data to database
ruby cli.rb db-import data/sample_employees.csv players

# Import team standings
ruby cli.rb db-import data/sample_products.csv teams

# Query top scorers
ruby cli.rb db-query "SELECT * FROM players WHERE goals > 50 ORDER BY goals DESC"

# List all hockey data tables
ruby cli.rb db-tables
```

### Programmatic Usage

```ruby
require_relative 'lib/csv_processor'
require_relative 'lib/csv_cleaner'
require_relative 'lib/csv_merger'

# Clean a CSV file
CSVProcessor.clean('data/input.csv')

# Merge CSV files
CSVProcessor.merge('data/file1.csv', 'data/file2.csv', 'output.csv')

# Advanced cleaning
cleaner = CSVCleaner.new('data/input.csv')
cleaned_data = cleaner.clean_data
cleaner.normalize_column(cleaned_data, 'age')
cleaner.save_to_csv(cleaned_data, 'cleaned_output.csv')

# Advanced merging
merger = CSVMerger.new
merged = merger.join_on_column('file1.csv', 'file2.csv', key_column: 'id')
merger.save_to_csv(merged, 'joined.csv')
```

## Testing

Run the test suite:
```bash
bundle exec rspec
```

Run specific test file:
```bash
bundle exec rspec spec/csv_processor_spec.rb
```

Run with verbose output:
```bash
bundle exec rspec --format documentation
```

## Project Structure

```
CSVy/
├── lib/
│   ├── csv_processor.rb       # Main CSV processing orchestrator
│   ├── csv_cleaner.rb         # Data cleaning functions
│   ├── csv_merger.rb          # CSV merging operations
│   ├── data_preprocessor.rb   # One-hot encoding, normalization, outlier removal
│   ├── data_validator.rb      # Validation, statistics, profiling
│   ├── dataframe_handler.rb   # Daru dataframe operations
│   └── database_manager.rb    # SQLite database integration
├── spec/
│   ├── csv_processor_spec.rb
│   ├── csv_cleaner_spec.rb
│   ├── csv_merger_spec.rb
│   ├── data_preprocessor_spec.rb
│   ├── data_validator_spec.rb
│   └── spec_helper.rb
├── data/                       # Sample CSV files included
│   ├── sample_employees.csv
│   ├── sample_products.csv
│   ├── sample_weather.csv
│   └── sample_students_dirty.csv
├── cli.rb                      # Command-line interface
├── Rakefile                    # Rake tasks for automation
├── Gemfile                     # Ruby dependencies
├── .gitignore
├── .env.example
├── USAGE_GUIDE.md             # Comprehensive usage examples
└── README.md
```

## Development

### Running the app in development
```bash
ruby cli.rb [command] [options]
```

### Adding new features
1. Create new methods in appropriate lib files
2. Add corresponding tests in spec/ directory
3. Update CLI commands in cli.rb if needed
4. Run tests to ensure everything works

## Dependencies

- `csv` - Built-in CSV handling
- `thor` - CLI framework
- `daru` - Dataframe operations (optional)
- `rspec` - Testing framework
- `dotenv` - Environment variable management
- `sqlite3` - Database support (optional)

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is open source and available under the MIT License.

## Author

**smiles0527**

## Acknowledgments

- Ruby CSV Standard Library
- Thor CLI Framework
- RSpec Testing Framework
- The Ruby community

## Support

For issues, questions, or contributions, please open an issue on GitHub.

---

CSVy - Professional CSV processing and organization tool for Ruby.
