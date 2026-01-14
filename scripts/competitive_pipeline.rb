#!/usr/bin/env ruby
# Advanced preprocessing pipeline for competitive hockey prediction
# Generates ALL 14+ advanced features for winning competitions

require 'csv'
require 'date'
require 'fileutils'
require_relative '../lib/csv_cleaner'
require_relative '../lib/data_preprocessor'
require_relative '../lib/time_series_features'
require_relative '../lib/advanced_features'
require_relative '../lib/model_validator'

class CompetitivePreprocessor
  def initialize(input_file, output_dir = 'data/processed')
    @input_file = input_file
    @output_dir = output_dir
    @logger = Logger.new(STDOUT)
    
    # Initialize all processors
    @cleaner = CSVCleaner.new(@logger)
    @preprocessor = DataPreprocessor.new(@logger)
    @ts_features = TimeSeriesFeatures.new(@logger)
    @advanced = AdvancedFeatures.new(@logger)
    @validator = ModelValidator.new(@logger)
    
    FileUtils.mkdir_p(@output_dir)
  end

  def run_full_pipeline
    @logger.info "=" * 60
    @logger.info "COMPETITIVE PREPROCESSING PIPELINE"
    @logger.info "=" * 60
    
    # Step 1: Load and clean data
    @logger.info "\n[1/7] Loading and cleaning data..."
    data = CSV.read(@input_file, headers: true).map(&:to_h)
    
    # Guard against empty CSV
    if data.empty?
      @logger.error "CSV file is empty or contains only headers"
      return nil
    end
    
    # Detect data type (standings vs game-level)
    is_game_level = data.first.key?('date') || data.first.key?('game_date') || data.first.key?('home_team')
    team_col = detect_column(data, ['Team', 'team', 'team_name', 'home_team', 'away_team'])
    date_col = detect_column(data, ['date', 'game_date', 'Date', 'DATE'])
    
    @logger.info "Data type: #{is_game_level ? 'Game-level' : 'Standings'}"
    @logger.info "Team column: #{team_col || 'Not found'}"
    @logger.info "Date column: #{date_col || 'Not found'}"
    
    # Remove duplicates
    data = remove_duplicates_simple(data)
    
    # Fill missing values for numeric columns
    numeric_cols = ['GF', 'GA', 'PTS', 'W', 'L', 'GP', 'DIFF', 'goals_for', 'goals_against']
    numeric_cols = numeric_cols.select { |col| data.first.key?(col) }
    data = fill_missing_numeric(data, numeric_cols) if numeric_cols.any?
    
    # Step 2: Basic preprocessing
    @logger.info "\n[2/7] Basic preprocessing..."
    
    # Detect column names
    wins_col = detect_column(data, ['W', 'wins', 'WINS'])
    losses_col = detect_column(data, ['L', 'losses', 'LOSSES'])
    gf_col = detect_column(data, ['GF', 'goals_for', 'GF'])
    ga_col = detect_column(data, ['GA', 'goals_against', 'GA'])
    gp_col = detect_column(data, ['GP', 'games_played', 'GP'])
    diff_col = detect_column(data, ['DIFF', 'diff', 'goal_diff', 'DIFF'])
    pts_col = detect_column(data, ['PTS', 'points', 'PTS'])
    
    # Calculate win percentage if not exists
    unless data.first.key?('win_pct')
      data.each do |row|
        games = (row[gp_col] || row['GP'] || 0).to_f
        wins = (row[wins_col] || row['W'] || 0).to_f
        row['win_pct'] = games > 0 ? (wins / games).round(4) : 0.5
      end
    end
    
    # Normalize numeric columns
    [pts_col, gf_col, ga_col].compact.each do |col|
      next unless data.first.key?(col)
      data = normalize_column_simple(data, col)
    end
    
    # Step 3: Time series features (if game-level data)
    @logger.info "\n[3/7] Engineering time series features..."
    
    if is_game_level && team_col && date_col
      # Sort by date and team
      data.sort_by! { |r| [r[team_col], parse_date(r[date_col])] }
      
      # Rolling averages (momentum indicators)
      [pts_col || 'PTS', gf_col || 'GF', ga_col || 'GA'].each do |col|
        next unless data.first.key?(col)
        data = @ts_features.rolling_window(data, col, 10, stat: :mean, group_by: team_col)
        data = @ts_features.rolling_window(data, col, 5, stat: :mean, group_by: team_col)
      end
      
      # Exponential weighted moving average (recent games weighted more)
      if data.first.key?('win_pct')
        data = @ts_features.ewma(data, 'win_pct', 10, group_by: team_col)
      end
      
      # Lag features (previous game values)
      [pts_col || 'PTS', gf_col || 'GF'].each do |col|
        next unless data.first.key?(col)
        data = @ts_features.lag_features(data, col, [1, 3, 5], group_by: team_col)
      end
      
      # Rest days and back-to-back
      if date_col
        data = @advanced.calculate_rest_days(data, team_col, date_col)
      end
      
      # Momentum score (rolling win rate)
      result_col = detect_column(data, ['result', 'outcome', 'win', 'W/L'])
      if result_col
        data = @advanced.calculate_momentum(data, team_col, result_col, window: 10)
      end
    end
    
    # Step 4: Advanced domain features (ALL 14+ FEATURES)
    @logger.info "\n[4/7] Creating advanced competition features..."
    
    # 1. Team strength index (composite metric)
    if wins_col && losses_col && diff_col
      data = @advanced.calculate_team_strength_index(data, team_col || 'Team', wins_col, losses_col, diff_col)
    end
    
    # 2. Pythagorean expectation (expected wins based on goals)
    if gf_col && ga_col && gp_col
      data = @advanced.calculate_pythagorean_wins(data, gf_col, ga_col, gp_col)
    end
    
    # 3. Interaction features (multiplicative effects)
    if gf_col && data.first.key?('win_pct')
      data = @advanced.create_interaction_features(data, gf_col, 'win_pct', 'offense_power')
    end
    if ga_col && losses_col
      data = @advanced.create_interaction_features(data, ga_col, losses_col, 'defense_weakness')
    end
    
    # 4. Polynomial features (non-linear relationships)
    if diff_col
      data = @advanced.create_polynomial_features(data, diff_col, degree: 2)
    end
    if pts_col
      data = @advanced.create_polynomial_features(data, pts_col, degree: 2)
    end
    
    # 5. Home/away splits
    location_col = detect_column(data, ['location', 'home_away', 'venue', 'HOME', 'AWAY'])
    if location_col && wins_col
      data = @advanced.calculate_home_away_splits(data, team_col || 'Team', location_col, wins_col)
    elsif data.first.key?('HOME') && data.first.key?('AWAY')
      # Handle standings format (HOME: "20-3-2")
      data.each do |row|
        begin
          home_parts = row['HOME'].to_s.split('-')
          away_parts = row['AWAY'].to_s.split('-')
          
          home_wins = home_parts[0].to_f
          home_total = home_parts.map(&:to_i).sum
          away_wins = away_parts[0].to_f
          away_total = away_parts.map(&:to_i).sum
          
          row['home_win_rate'] = home_total > 0 ? (home_wins / home_total).round(3) : 0.5
          row['away_win_rate'] = away_total > 0 ? (away_wins / away_total).round(3) : 0.5
          row['home_away_diff'] = (row['home_win_rate'].to_f - row['away_win_rate'].to_f).round(3)
        rescue StandardError => e
          @logger.warn "Failed to parse HOME/AWAY record: #{e.message}"
          row['home_win_rate'] = 0.5
          row['away_win_rate'] = 0.5
          row['home_away_diff'] = 0.0
        end
      end
    end
    
    # 6. Clutch factor (performance in close games)
    if is_game_level && diff_col && (result_col = detect_column(data, ['result', 'outcome', 'win', 'W/L']))
      data = @advanced.calculate_clutch_factor(data, team_col || 'Team', diff_col, result_col)
    end
    
    # 7. Strength of schedule
    if is_game_level
      opponent_col = detect_column(data, ['opponent', 'opponent_team', 'away_team', 'home_team'])
      if opponent_col && wins_col
        # Calculate opponent win rates
        team_wins = {}
        team_games = {}
        data.each do |row|
          team = row[team_col]
          next unless team
          team_wins[team] ||= 0
          team_games[team] ||= 0
          result_val = row[result_col || 'result']
          team_wins[team] += 1 if result_val.to_s.upcase == 'W'
          team_games[team] += 1
        end
        
        team_win_rates = {}
        team_wins.each { |team, wins| team_win_rates[team] = wins.to_f / team_games[team] if team_games[team] > 0 }
        
        # Calculate SOS for each team
        data.each do |row|
          team = row[team_col]
          opponents = data.select { |r| r[team_col] == team }.map { |r| r[opponent_col] }.compact
          opp_win_rates = opponents.map { |opp| team_win_rates[opp] || 0.5 }
          row['strength_of_schedule'] = opp_win_rates.any? ? (opp_win_rates.sum / opp_win_rates.size).round(3) : 0.5
        end
      end
    end
    
    # 8. Consistency metrics (coefficient of variation)
    score_col = pts_col || gf_col || 'PTS'
    if score_col && team_col && data.first.key?(score_col)
      data = @advanced.calculate_consistency_metrics(data, team_col, score_col)
    end
    
    # 9. Time decay weights (recent games weighted higher)
    if date_col
      data = @advanced.apply_time_decay_weights(data, date_col, decay_rate: 0.05)
    end
    
    # 10. Enhanced team strength index (more sophisticated)
    if wins_col && losses_col && diff_col && gf_col && ga_col && gp_col
      data = @advanced.calculate_enhanced_strength_index(data, team_col || 'Team', wins_col, losses_col, diff_col, gf_col, ga_col, gp_col)
    end
    
    # 11. Playoff pressure indicators
    if pts_col
      playoff_status_col = detect_column(data, ['playoff_status', 'status', 'clinched'])
      data = @advanced.calculate_playoff_pressure(data, team_col || 'Team', pts_col, playoff_status_col)
    end
    
    # 12. Parse streak indicators (STRK field)
    streak_col = detect_column(data, ['STRK', 'streak', 'Streak', 'current_streak'])
    if streak_col
      data = @advanced.parse_streak(data, streak_col)
    end
    
    # 13. Parse L10 (Last 10 games) record
    l10_col = detect_column(data, ['L10', 'last_10', 'Last10'])
    if l10_col
      data = @advanced.parse_l10_record(data, l10_col)
    end
    
    # 14. Parse shootout record
    so_col = detect_column(data, ['S/O', 'SO', 'shootout', 'shootout_record'])
    if so_col
      data = @advanced.parse_shootout_record(data, so_col)
    end
    
    # 15. Head-to-head records (if game-level data)
    if is_game_level
      opponent_col = detect_column(data, ['opponent', 'opponent_team', 'away_team', 'home_team'])
      result_col = detect_column(data, ['result', 'outcome', 'win', 'W/L'])
      if opponent_col && result_col && team_col
        data = @advanced.calculate_h2h_record(data, team_col, opponent_col, result_col)
      end
    end
    
    # 16. Conference/division adjustments (if available)
    conference_col = detect_column(data, ['conference', 'Conference', 'CONF'])
    division_col = detect_column(data, ['division', 'Division', 'DIV'])
    if conference_col && division_col && data.first.key?('win_pct')
      data = @advanced.calculate_conference_adjustments(data, conference_col, division_col, 'win_pct')
    end
    
    # 17. Opponent strength at time of game (for game-level data)
    if is_game_level && date_col && opponent_col && pts_col
      data = @advanced.calculate_opponent_strength_at_game(data, team_col, opponent_col, date_col, pts_col, gf_col, ga_col)
    end
    
    # Step 5: Feature engineering summary
    @logger.info "\n[5/7] Feature engineering summary..."
    @logger.info "Total features: #{data.first.keys.size}"
    @logger.info "Sample count: #{data.size}"
    
    # List all advanced features created
    advanced_features = [
      'team_strength_index', 'enhanced_strength_index', 'offense_rating', 'defense_rating',
      'pythagorean_wins', 'pythagorean_win_pct', 'luck_factor',
      'offense_power', 'defense_weakness', 'momentum_score', 'rest_days', 'is_back_to_back',
      'clutch_factor', 'home_win_rate', 'away_win_rate', 'home_away_diff',
      'strength_of_schedule', 'consistency_score', 'time_weight',
      'pts_from_playoff_line', 'playoff_probability', 'is_clinched', 'is_eliminated',
      'streak_type', 'streak_length', 'is_winning_streak', 'is_losing_streak',
      'l10_wins', 'l10_losses', 'l10_win_rate', 'l10_points', 'l10_points_pct',
      'so_wins', 'so_losses', 'so_win_rate', 'h2h_win_rate',
      'conference_strength', 'division_strength', 'adjusted_win_pct',
      'opponent_strength', 'opponent_pts_per_game', 'opponent_gf_per_game', 'opponent_ga_per_game'
    ]
    
    created_features = advanced_features.select { |f| data.first.key?(f) }
    @logger.info "Advanced features created: #{created_features.size}/#{advanced_features.size}"
    @logger.info "  #{created_features.join(', ')}"
    
    # Step 6: Export processed data
    @logger.info "\n[6/7] Exporting processed data..."
    
    output_file = File.join(@output_dir, 'competitive_features.csv')
    CSV.open(output_file, 'w') do |csv|
      csv << data.first.keys
      data.each { |row| csv << data.first.keys.map { |k| row[k] || '' } }
    end
    
    @logger.info "Exported to: #{output_file}"
    
    # Step 7: Create train/test splits
    @logger.info "\n[7/7] Creating train/test splits..."
    
    if data.size > 100
      if date_col && is_game_level
        # Time series split (expanding window, no data leakage)
        data.sort_by! { |r| parse_date(r[date_col]) }
        split_idx = (data.size * 0.8).to_i
        
        train_data = data[0...split_idx]
        test_data = data[split_idx..-1]
        
        train_file = File.join(@output_dir, 'train.csv')
        test_file = File.join(@output_dir, 'test.csv')
        
        CSV.open(train_file, 'w') do |csv|
          csv << train_data.first.keys
          train_data.each { |row| csv << train_data.first.keys.map { |k| row[k] || '' } }
        end
        
        CSV.open(test_file, 'w') do |csv|
          csv << test_data.first.keys
          test_data.each { |row| csv << test_data.first.keys.map { |k| row[k] || '' } }
        end
        
        @logger.info "Time series split:"
        @logger.info "  Train: #{train_file} (#{train_data.size} samples)"
        @logger.info "  Test: #{test_file} (#{test_data.size} samples)"
      elsif data.first.key?('playoff_status')
        # Stratified split
        split = @validator.stratified_split(data, 'playoff_status', test_size: 0.2)
        
        train_file = File.join(@output_dir, 'train.csv')
        test_file = File.join(@output_dir, 'test.csv')
        
        CSV.open(train_file, 'w') do |csv|
          csv << split[:train].first.keys
          split[:train].each { |row| csv << split[:train].first.keys.map { |k| row[k] || '' } }
        end
        
        CSV.open(test_file, 'w') do |csv|
          csv << split[:test].first.keys
          split[:test].each { |row| csv << split[:test].first.keys.map { |k| row[k] || '' } }
        end
        
        @logger.info "Stratified split:"
        @logger.info "  Train: #{train_file} (#{split[:train].size} samples)"
        @logger.info "  Test: #{test_file} (#{split[:test].size} samples)"
      else
        # Simple 80/20 split
        split_idx = (data.size * 0.8).to_i
        train_data = data[0...split_idx]
        test_data = data[split_idx..-1]
        
        train_file = File.join(@output_dir, 'train.csv')
        test_file = File.join(@output_dir, 'test.csv')
        
        CSV.open(train_file, 'w') do |csv|
          csv << train_data.first.keys
          train_data.each { |row| csv << train_data.first.keys.map { |k| row[k] || '' } }
        end
        
        CSV.open(test_file, 'w') do |csv|
          csv << test_data.first.keys
          test_data.each { |row| csv << test_data.first.keys.map { |k| row[k] || '' } }
        end
        
        @logger.info "Simple split:"
        @logger.info "  Train: #{train_file} (#{train_data.size} samples)"
        @logger.info "  Test: #{test_file} (#{test_data.size} samples)"
      end
    end
    
    @logger.info "\n" + "=" * 60
    @logger.info "PIPELINE COMPLETE - READY TO WIN!"
    @logger.info "=" * 60
    @logger.info "Created #{created_features.size} advanced features"
    @logger.info "Total features in dataset: #{data.first.keys.size}"
    @logger.info "Output: #{output_file}"
    
    # Optional: Generate feature correlation report if target column exists
    target_col = detect_column(data, ['target', 'y', 'outcome', 'result', 'PTS', 'W'])
    if target_col && data.size > 10
      @logger.info "\nGenerating feature correlation analysis..."
      begin
        correlations = @advanced.analyze_feature_correlations(data, target_col, data.first.keys - [target_col])
        
        # Save top correlations
        corr_file = File.join(@output_dir, 'feature_correlations.csv')
        CSV.open(corr_file, 'w') do |csv|
          csv << ['feature', 'correlation', 'abs_correlation']
          correlations.sort_by { |k, v| -v.abs }.first(20).each do |feature, corr|
            csv << [feature, corr, corr.abs]
          end
        end
        @logger.info "Feature correlations saved to: #{corr_file}"
      rescue => e
        @logger.warn "Could not generate correlations: #{e.message}"
      end
    end
    
    output_file
  end
  
  private
  
  def detect_column(data, candidates)
    return nil if data.empty?
    candidates.find { |candidate| data.first.key?(candidate) }
  end
  
  def remove_duplicates_simple(data)
    data.uniq { |row| row.values.join('|') }
  end
  
  def fill_missing_numeric(data, columns)
    columns.each do |col|
      # Only treat nil/empty as missing, include zeros in mean calculation
      values = data.map { |r| r[col] }.reject { |v| v.nil? || v.to_s.strip.empty? }.map(&:to_f)
      mean = values.any? ? values.sum / values.size : 0
      # Only fill truly missing values (nil or empty), not zeros
      data.each { |r| r[col] = mean if r[col].nil? || r[col].to_s.strip.empty? }
    end
    data
  end
  
  def normalize_column_simple(data, col)
    values = data.map { |r| r[col].to_f }
    min = values.min || 0
    max = values.max || 1
    range = max - min
    range = 1 if range.zero?
    
    data.each do |r|
      val = r[col].to_f
      r[col] = ((val - min) / range).round(4)
    end
    data
  end
  
  def parse_date(date_str)
    if date_str.nil? || date_str.to_s.strip.empty?
      @logger.warn "Missing date value, using epoch sentinel"
      return Date.new(1970, 1, 1)  # Sentinel value to prevent data leakage
    end
    Date.parse(date_str.to_s)
  rescue ArgumentError => e
    @logger.error "Unparseable date '#{date_str}': #{e.message}"
    Date.new(1970, 1, 1)  # Sentinel value - filter these rows if needed
  end
  
  def safe_float(value, default: 0.0)
    return default if value.nil? || value.to_s.strip.empty?
    value.to_f
  rescue
    default
  end
  
  def safe_int(value, default: 0)
    return default if value.nil? || value.to_s.strip.empty?
    value.to_i
  rescue
    default
  end
end

# Run if called directly
if __FILE__ == $0
  if ARGV.empty?
    puts "Usage: ruby scripts/competitive_pipeline.rb <input_csv>"
    puts "Example: ruby scripts/competitive_pipeline.rb data/nhl_data.csv"
    exit 1
  end
  
  preprocessor = CompetitivePreprocessor.new(ARGV[0])
  preprocessor.run_full_pipeline
end
