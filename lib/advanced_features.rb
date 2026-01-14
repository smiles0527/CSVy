require 'csv'
require 'logger'

class AdvancedFeatures
  attr_reader :logger

  def initialize(logger = Logger.new(STDOUT))
    @logger = logger
  end

  # Team strength indicators
  def calculate_team_strength_index(data, team_col, wins_col, losses_col, diff_col)
    logger.info "Calculating team strength index"
    
    data.each do |row|
      wins = row[wins_col].to_f
      losses = row[losses_col].to_f
      diff = row[diff_col].to_f
      
      # Composite strength: win rate + normalized goal differential
      games = wins + losses
      win_rate = games > 0 ? wins / games : 0.5
      
      # Strength index (0-100 scale)
      strength = (win_rate * 50) + (diff * 0.5)
      row['team_strength_index'] = strength.round(2)
    end
    
    data
  end

  # Interaction features (multiplicative effects)
  def create_interaction_features(data, col1, col2, interaction_name = nil)
    logger.info "Creating interaction: #{col1} × #{col2}"
    
    interaction_name ||= "#{col1}_x_#{col2}"
    
    data.each do |row|
      val1 = row[col1].to_f
      val2 = row[col2].to_f
      row[interaction_name] = (val1 * val2).round(4)
    end
    
    data
  end

  # Polynomial features (non-linear relationships)
  def create_polynomial_features(data, col, degree: 2)
    logger.info "Creating polynomial features for #{col} (degree #{degree})"
    
    (2..degree).each do |d|
      new_col = "#{col}_pow#{d}"
      data.each do |row|
        val = row[col].to_f
        row[new_col] = (val ** d).round(4)
      end
    end
    
    data
  end

  # Momentum score (recent performance trend)
  def calculate_momentum(data, group_col, result_col, window: 10)
    logger.info "Calculating momentum score (last #{window} games)"
    
    grouped = data.group_by { |row| row[group_col] }
    
    grouped.each do |team, team_data|
      team_data.sort_by! { |r| r['date'] || r['game_number'] || 0 }
      
      team_data.each_with_index do |row, idx|
        recent = team_data[[0, idx - window + 1].max..idx]
        
        # Calculate win rate in window
        wins = recent.count { |r| r[result_col].to_s.upcase == 'W' }
        momentum = wins.to_f / recent.size
        
        row['momentum_score'] = momentum.round(3)
      end
    end
    
    data
  end

  # Rest advantage (days since last game)
  def calculate_rest_days(data, group_col, date_col)
    logger.info "Calculating rest days between games"
    
    grouped = data.group_by { |row| row[group_col] }
    
    grouped.each do |team, team_data|
      team_data.sort_by! { |r| Date.parse(r[date_col].to_s) }
      
      team_data.each_with_index do |row, idx|
        if idx == 0
          row['rest_days'] = 3 # Default for first game
        else
          prev_date = Date.parse(team_data[idx - 1][date_col].to_s)
          curr_date = Date.parse(row[date_col].to_s)
          row['rest_days'] = (curr_date - prev_date).to_i
        end
        
        # Flag back-to-back games
        row['is_back_to_back'] = row['rest_days'].to_i <= 1 ? 1 : 0
      end
    end
    
    data
  end

  # Head-to-head record
  def calculate_h2h_record(data, team1_col, team2_col, result_col)
    logger.info "Calculating head-to-head records"
    
    h2h_wins = Hash.new(0)
    h2h_games = Hash.new(0)
    
    data.each do |row|
      team1 = row[team1_col]
      team2 = row[team2_col]
      result = row[result_col]
      
      matchup = [team1, team2].sort.join('_vs_')
      h2h_games[matchup] += 1
      
      if result.to_s.upcase == 'W'
        h2h_wins["#{team1}_vs_#{team2}"] += 1
      end
    end
    
    # Add win rate against specific opponent
    data.each do |row|
      team1 = row[team1_col]
      team2 = row[team2_col]
      key = "#{team1}_vs_#{team2}"
      
      games = h2h_games[[team1, team2].sort.join('_vs_')]
      wins = h2h_wins[key]
      
      row['h2h_win_rate'] = games > 0 ? (wins.to_f / games).round(3) : 0.5
    end
    
    data
  end

  # Strength of schedule
  def calculate_strength_of_schedule(data, team_col, opponent_col, opponent_wins_col)
    logger.info "Calculating strength of schedule"
    
    # Calculate average opponent win rate
    team_schedules = Hash.new { |h, k| h[k] = [] }
    
    data.each do |row|
      team = row[team_col]
      opponent = row[opponent_col]
      opp_wins = row[opponent_wins_col].to_f
      
      team_schedules[team] << opp_wins
    end
    
    # Add SOS to each row
    data.each do |row|
      team = row[team_col]
      opponents = team_schedules[team]
      
      sos = opponents.any? ? opponents.sum / opponents.size : 0.5
      row['strength_of_schedule'] = sos.round(3)
    end
    
    data
  end

  # Clutch performance (close game win rate)
  def calculate_clutch_factor(data, group_col, goal_diff_col, result_col)
    logger.info "Calculating clutch performance in close games"
    
    grouped = data.group_by { |row| row[group_col] }
    
    grouped.each do |team, team_data|
      close_games = team_data.select { |r| r[goal_diff_col].to_i.abs <= 1 }
      close_wins = close_games.count { |r| r[result_col].to_s.upcase == 'W' }
      
      clutch_factor = close_games.any? ? close_wins.to_f / close_games.size : 0.5
      
      team_data.each do |row|
        row['clutch_factor'] = clutch_factor.round(3)
      end
    end
    
    data
  end

  # Home/away splits
  def calculate_home_away_splits(data, group_col, location_col, wins_col)
    logger.info "Calculating home/away performance splits"
    
    grouped = data.group_by { |row| row[group_col] }
    
    grouped.each do |team, team_data|
      home_games = team_data.select { |r| r[location_col].to_s.upcase == 'HOME' }
      away_games = team_data.select { |r| r[location_col].to_s.upcase == 'AWAY' }
      
      home_wins = home_games.count { |r| r[wins_col].to_i > 0 }
      away_wins = away_games.count { |r| r[wins_col].to_i > 0 }
      
      home_win_rate = home_games.any? ? home_wins.to_f / home_games.size : 0.5
      away_win_rate = away_games.any? ? away_wins.to_f / away_games.size : 0.5
      
      team_data.each do |row|
        row['home_win_rate'] = home_win_rate.round(3)
        row['away_win_rate'] = away_win_rate.round(3)
        row['home_away_diff'] = (home_win_rate - away_win_rate).round(3)
      end
    end
    
    data
  end

  # Pythagorean expectation (expected wins based on goals)
  def calculate_pythagorean_wins(data, gf_col, ga_col, games_col)
    logger.info "Calculating Pythagorean expected wins"
    
    data.each do |row|
      gf = row[gf_col].to_f
      ga = row[ga_col].to_f
      games = row[games_col].to_f
      
      # Pythagorean formula: GF^2 / (GF^2 + GA^2)
      if gf > 0 && ga > 0
        expected_win_pct = (gf ** 2) / ((gf ** 2) + (ga ** 2))
        expected_wins = expected_win_pct * games
        
        row['pythagorean_wins'] = expected_wins.round(2)
        row['pythagorean_win_pct'] = expected_win_pct.round(3)
        
        # Luck factor (actual wins - expected wins)
        actual_wins = row['W'].to_f
        row['luck_factor'] = (actual_wins - expected_wins).round(2)
      end
    end
    
    data
  end

  # Variance/consistency metrics
  def calculate_consistency_metrics(data, group_col, score_col)
    logger.info "Calculating team consistency metrics"
    
    grouped = data.group_by { |row| row[group_col] }
    
    grouped.each do |team, team_data|
      scores = team_data.map { |r| r[score_col].to_f }
      
      mean = scores.sum / scores.size.to_f
      variance = scores.map { |s| (s - mean) ** 2 }.sum / scores.size
      std_dev = Math.sqrt(variance)
      
      # Coefficient of variation (lower = more consistent)
      cv = mean != 0 ? std_dev / mean : 0
      
      team_data.each do |row|
        row['score_std_dev'] = std_dev.round(3)
        row['score_cv'] = cv.round(3)
        row['consistency_score'] = (1 - cv).round(3) # Higher = more consistent
      end
    end
    
    data
  end

  # Time decay weights (recent games matter more)
  def apply_time_decay_weights(data, date_col, decay_rate: 0.05)
    logger.info "Applying time decay weights (decay=#{decay_rate})"
    
    data.sort_by! { |r| Date.parse(r[date_col].to_s) }
    
    max_date = Date.parse(data.last[date_col].to_s)
    
    data.each do |row|
      curr_date = Date.parse(row[date_col].to_s)
      days_ago = (max_date - curr_date).to_i
      
      # Exponential decay: weight = exp(-decay_rate * days_ago)
      weight = Math.exp(-decay_rate * days_ago)
      row['time_weight'] = weight.round(4)
    end
    
    data
  end

  # Conference/division strength adjustments
  def calculate_conference_adjustments(data, conference_col, division_col, win_pct_col)
    logger.info "Calculating conference/division strength adjustments"
    
    # Average win % by conference
    conf_grouped = data.group_by { |r| r[conference_col] }
    conf_avg = {}
    
    conf_grouped.each do |conf, rows|
      avg = rows.map { |r| r[win_pct_col].to_f }.sum / rows.size
      conf_avg[conf] = avg
    end
    
    # Average win % by division
    div_grouped = data.group_by { |r| r[division_col] }
    div_avg = {}
    
    div_grouped.each do |div, rows|
      avg = rows.map { |r| r[win_pct_col].to_f }.sum / rows.size
      div_avg[div] = avg
    end
    
    # Apply adjustments
    data.each do |row|
      conf = row[conference_col]
      div = row[division_col]
      
      row['conference_strength'] = conf_avg[conf].round(3)
      row['division_strength'] = div_avg[div].round(3)
      
      # Adjusted win % (normalize by conference strength)
      win_pct = row[win_pct_col].to_f
      adjusted = win_pct / conf_avg[conf] * 0.5 # Normalize to league average
      row['adjusted_win_pct'] = adjusted.round(3)
    end
    
    data
  end

  # Playoff pressure indicators
  def calculate_playoff_pressure(data, team_col, pts_col, playoff_status_col = nil)
    logger.info "Calculating playoff pressure indicators"
    
    # Sort by points (descending)
    sorted_data = data.sort_by { |r| -(r[pts_col].to_f) }
    
    # Find playoff cutoff (8th place in each conference, or overall)
    cutoff_pts = sorted_data[7] ? sorted_data[7][pts_col].to_f : 0
    
    # Calculate distance from playoff line
    data.each do |row|
      team_pts = row[pts_col].to_f
      row['pts_from_playoff_line'] = (team_pts - cutoff_pts).round(1)
      row['playoff_probability'] = team_pts >= cutoff_pts ? 1.0 : [0, (team_pts / cutoff_pts)].max.round(3)
      
      # Clinched status
      if playoff_status_col && row[playoff_status_col]
        status = row[playoff_status_col].to_s.upcase
        row['is_clinched'] = status == 'X' || status == 'Y' ? 1 : 0
        row['is_eliminated'] = status == 'E' ? 1 : 0
      else
        row['is_clinched'] = team_pts >= cutoff_pts + 10 ? 1 : 0 # Heuristic
        row['is_eliminated'] = team_pts < cutoff_pts - 20 ? 1 : 0 # Heuristic
      end
    end
    
    data
  end

  # Parse streak string (e.g., "W5", "L3", "W1")
  def parse_streak(data, streak_col)
    logger.info "Parsing streak indicators from #{streak_col}"
    
    data.each do |row|
      streak_str = row[streak_col].to_s.upcase.strip
      
      if streak_str.match?(/^([WL])(\d+)$/)
        type = $1
        length = $2.to_i
        
        row['streak_type'] = type == 'W' ? 1 : -1
        row['streak_length'] = length
        row['is_winning_streak'] = type == 'W' ? 1 : 0
        row['is_losing_streak'] = type == 'L' ? 1 : 0
      else
        row['streak_type'] = 0
        row['streak_length'] = 0
        row['is_winning_streak'] = 0
        row['is_losing_streak'] = 0
      end
    end
    
    data
  end

  # Parse L10 (Last 10 games) record (e.g., "8-2-0")
  def parse_l10_record(data, l10_col)
    logger.info "Parsing L10 records from #{l10_col}"
    
    data.each do |row|
      l10_str = row[l10_col].to_s.strip
      
      if l10_str.match?(/^(\d+)-(\d+)-(\d+)$/)
        wins = $1.to_i
        losses = $2.to_i
        ot = $3.to_i
        
        total = wins + losses + ot
        row['l10_wins'] = wins
        row['l10_losses'] = losses
        row['l10_ot'] = ot
        row['l10_win_rate'] = total > 0 ? (wins.to_f / total).round(3) : 0.5
        row['l10_points'] = wins * 2 + ot
        row['l10_points_pct'] = total > 0 ? (row['l10_points'].to_f / (total * 2)).round(3) : 0.5
      else
        row['l10_wins'] = 0
        row['l10_losses'] = 0
        row['l10_ot'] = 0
        row['l10_win_rate'] = 0.5
        row['l10_points'] = 0
        row['l10_points_pct'] = 0.5
      end
    end
    
    data
  end

  # Parse shootout record (e.g., "3-2")
  def parse_shootout_record(data, so_col)
    logger.info "Parsing shootout records from #{so_col}"
    
    data.each do |row|
      so_str = row[so_col].to_s.strip
      
      if so_str.match?(/^(\d+)-(\d+)$/)
        wins = $1.to_i
        losses = $2.to_i
        total = wins + losses
        
        row['so_wins'] = wins
        row['so_losses'] = losses
        row['so_win_rate'] = total > 0 ? (wins.to_f / total).round(3) : 0.5
        row['so_total'] = total
      else
        row['so_wins'] = 0
        row['so_losses'] = 0
        row['so_win_rate'] = 0.5
        row['so_total'] = 0
      end
    end
    
    data
  end

  # Enhanced team strength index (more sophisticated)
  def calculate_enhanced_strength_index(data, team_col, wins_col, losses_col, diff_col, gf_col, ga_col, gp_col)
    logger.info "Calculating enhanced team strength index"
    
    data.each do |row|
      wins = row[wins_col].to_f
      losses = row[losses_col].to_f
      diff = row[diff_col].to_f
      gf = row[gf_col].to_f
      ga = row[ga_col].to_f
      games = row[gp_col].to_f
      
      # Components
      win_rate = games > 0 ? wins / games : 0.5
      goal_diff_per_game = games > 0 ? diff / games : 0
      goals_for_per_game = games > 0 ? gf / games : 0
      goals_against_per_game = games > 0 ? ga / games : 0
      
      # Pythagorean component
      pythagorean_pct = (gf > 0 && ga > 0) ? (gf ** 2) / ((gf ** 2) + (ga ** 2)) : 0.5
      
      # Weighted strength index (0-100 scale)
      strength = (
        win_rate * 30 +                    # Win rate component
        [goal_diff_per_game * 2, 20].min + # Goal diff component (capped)
        pythagorean_pct * 30 +              # Pythagorean component
        [goals_for_per_game * 2, 10].min + # Offense component
        [10 - goals_against_per_game, 10].min # Defense component (inverted)
      )
      
      row['enhanced_strength_index'] = [strength, 100].min.round(2)
      row['offense_rating'] = [goals_for_per_game * 10, 50].min.round(2)
      row['defense_rating'] = [10 - goals_against_per_game, 10].min.round(2)
    end
    
    data
  end

  # Opponent strength at time of game (for game-level data)
  def calculate_opponent_strength_at_game(data, team_col, opponent_col, date_col, pts_col, gf_col, ga_col)
    logger.info "Calculating opponent strength at time of each game"
    
    # Sort by date
    data.sort_by! { |r| parse_date_safe(r[date_col]) }
    
    # Track cumulative stats per team up to each game
    team_stats = Hash.new { |h, k| h[k] = { games: 0, pts: 0, gf: 0, ga: 0 } }
    
    data.each do |row|
      team = row[team_col]
      opponent = row[opponent_col]
      
      # Get opponent's strength BEFORE this game
      opp_stats = team_stats[opponent]
      
      if opp_stats[:games] > 0
        opp_pts_per_game = opp_stats[:pts].to_f / opp_stats[:games]
        opp_gf_per_game = opp_stats[:gf].to_f / opp_stats[:games]
        opp_ga_per_game = opp_stats[:ga].to_f / opp_stats[:games]
        
        row['opponent_pts_per_game'] = opp_pts_per_game.round(3)
        row['opponent_gf_per_game'] = opp_gf_per_game.round(3)
        row['opponent_ga_per_game'] = opp_ga_per_game.round(3)
        row['opponent_strength'] = (opp_pts_per_game * 0.5 + opp_gf_per_game * 0.3 - opp_ga_per_game * 0.2).round(3)
      else
        row['opponent_pts_per_game'] = 1.0 # Default
        row['opponent_gf_per_game'] = 2.5
        row['opponent_ga_per_game'] = 2.5
        row['opponent_strength'] = 1.0
      end
      
      # Update team stats AFTER processing this game
      team_stats[team][:games] += 1
      team_stats[team][:pts] += row[pts_col].to_f
      team_stats[team][:gf] += row[gf_col].to_f if gf_col
      team_stats[team][:ga] += row[ga_col].to_f if ga_col
    end
    
    data
  end

  # Feature correlation analysis
  def analyze_feature_correlations(data, target_col, feature_cols)
    logger.info "Analyzing feature correlations with #{target_col}"
    
    return {} if data.empty? || !data.first.key?(target_col)
    
    target_values = data.map { |r| r[target_col].to_f }
    
    correlations = {}
    feature_cols.each do |col|
      next unless data.first.key?(col)
      
      feature_values = data.map { |r| r[col].to_f }
      corr = calculate_pearson_correlation(target_values, feature_values)
      correlations[col] = corr.round(4) unless corr.nan?
    end
    
    # Sort by absolute correlation
    sorted = correlations.sort_by { |k, v| -v.abs }
    
    logger.info "Top correlated features:"
    sorted.first(10).each do |col, corr|
      logger.info "  #{col}: #{corr.round(4)}"
    end
    
    correlations
  end

  private

  def parse_date_safe(date_str)
    return Date.today if date_str.nil? || date_str.to_s.strip.empty?
    Date.parse(date_str.to_s)
  rescue
    Date.today
  end

  def calculate_pearson_correlation(x, y)
    return 0.0 if x.length != y.length || x.length < 2
    
    n = x.length
    sum_x = x.sum
    sum_y = y.sum
    sum_xy = x.zip(y).map { |a, b| a * b }.sum
    sum_x2 = x.map { |a| a * a }.sum
    sum_y2 = y.map { |a| a * a }.sum
    
    numerator = n * sum_xy - sum_x * sum_y
    denominator = Math.sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y))
    
    return 0.0 if denominator.zero?
    numerator / denominator
  end
end
