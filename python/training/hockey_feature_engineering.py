"""
Hockey-Specific Feature Engineering
Real hockey dynamics: momentum, fatigue, rest, injuries, motivation, special teams
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class HockeyFeatureEngineer:
    """
    Feature engineering for NHL game prediction incorporating real hockey dynamics.
    
    Designed for competition data with built-in features:
    - travel_distance, travel_time, rest_time
    - injuries, clinch_status, elimination_status
    - division_tier (D1/D2/D3)
    """
    
    def __init__(self, df):
        """Initialize with game-level dataframe."""
        self.df = df.copy()
        self.features_created = []
        
    def create_all_features(self):
        """Generate all hockey-specific features."""
        print("\n" + "="*80)
        print("HOCKEY FEATURE ENGINEERING")
        print("="*80)
        
        # Core dynamics
        self.create_fatigue_features()
        self.create_rest_features()
        self.create_injury_features()
        self.create_motivation_features()
        
        # Performance metrics
        self.create_momentum_features()
        self.create_goalie_features()
        self.create_special_teams_features()
        self.create_scoring_features()
        
        # Situational
        self.create_rivalry_features()
        self.create_temporal_features()
        self.create_arena_features()
        
        # Advanced stats
        self.create_advanced_stats()
        self.create_interaction_features()
        
        print(f"\n[OK] Created {len(self.features_created)} new features")
        return self.df, self.features_created
    
    def create_fatigue_features(self):
        """Travel and rest-based fatigue indicators."""
        print("\n   Fatigue Features...")
        
        if 'travel_distance' in self.df.columns and 'rest_time' in self.df.columns:
            # Fatigue index: high travel + low rest = exhaustion
            self.df['fatigue_index_home'] = self.df['travel_distance_home'] / (self.df['rest_time_home'] + 1)
            self.df['fatigue_index_away'] = self.df['travel_distance_away'] / (self.df['rest_time_away'] + 1)
            self.features_created.extend(['fatigue_index_home', 'fatigue_index_away'])
            
            # Back-to-back games (most fatiguing)
            self.df['back_to_back_home'] = (self.df['rest_time_home'] <= 1).astype(int)
            self.df['back_to_back_away'] = (self.df['rest_time_away'] <= 1).astype(int)
            self.features_created.extend(['back_to_back_home', 'back_to_back_away'])
            
            # Cross-country travel (3+ time zones)
            if 'travel_time' in self.df.columns:
                self.df['long_travel_home'] = (self.df['travel_time_home'] > 180).astype(int)  # 3+ hours
                self.df['long_travel_away'] = (self.df['travel_time_away'] > 180).astype(int)
                self.features_created.extend(['long_travel_home', 'long_travel_away'])
    
    def create_rest_features(self):
        """Rest and recovery patterns."""
        print("   Rest Features...")
        
        if 'rest_time' in self.df.columns:
            # Rest categories
            self.df['rest_back_to_back_home'] = (self.df['rest_time_home'] <= 1).astype(int)
            self.df['rest_normal_home'] = ((self.df['rest_time_home'] >= 2) & (self.df['rest_time_home'] <= 3)).astype(int)
            self.df['rest_extended_home'] = (self.df['rest_time_home'] >= 4).astype(int)
            
            self.df['rest_back_to_back_away'] = (self.df['rest_time_away'] <= 1).astype(int)
            self.df['rest_normal_away'] = ((self.df['rest_time_away'] >= 2) & (self.df['rest_time_away'] <= 3)).astype(int)
            self.df['rest_extended_away'] = (self.df['rest_time_away'] >= 4).astype(int)
            
            self.features_created.extend([
                'rest_back_to_back_home', 'rest_normal_home', 'rest_extended_home',
                'rest_back_to_back_away', 'rest_normal_away', 'rest_extended_away'
            ])
            
            # Rest advantage
            self.df['rest_advantage'] = self.df['rest_time_home'] - self.df['rest_time_away']
            self.features_created.append('rest_advantage')
    
    def create_injury_features(self):
        """Injury impact on performance."""
        print("   Injury Features...")
        
        if 'injuries' in self.df.columns:
            # Injury severity
            self.df['has_injuries_home'] = (self.df['injuries_home'] > 0).astype(int)
            self.df['has_injuries_away'] = (self.df['injuries_away'] > 0).astype(int)
            self.df['multiple_injuries_home'] = (self.df['injuries_home'] >= 3).astype(int)
            self.df['multiple_injuries_away'] = (self.df['injuries_away'] >= 3).astype(int)
            
            self.features_created.extend([
                'has_injuries_home', 'has_injuries_away',
                'multiple_injuries_home', 'multiple_injuries_away'
            ])
            
            # Injuries while traveling (compounded fatigue)
            if 'travel_distance' in self.df.columns:
                self.df['injury_travel_home'] = self.df['injuries_home'] * self.df['travel_distance_home']
                self.df['injury_travel_away'] = self.df['injuries_away'] * self.df['travel_distance_away']
                self.features_created.extend(['injury_travel_home', 'injury_travel_away'])
            
            # Injury differential
            self.df['injury_advantage'] = self.df['injuries_away'] - self.df['injuries_home']
            self.features_created.append('injury_advantage')
    
    def create_motivation_features(self):
        """Playoff motivation and elimination effects."""
        print("   Motivation Features...")
        
        if 'clinch_status' in self.df.columns:
            # Clinched early (may rest stars)
            self.df['clinched_home'] = (self.df['clinch_status_home'] == 1).astype(int)
            self.df['clinched_away'] = (self.df['clinch_status_away'] == 1).astype(int)
            self.features_created.extend(['clinched_home', 'clinched_away'])
        
        if 'elimination_status' in self.df.columns:
            # Eliminated (tanking risk)
            self.df['eliminated_home'] = (self.df['elimination_status_home'] == 1).astype(int)
            self.df['eliminated_away'] = (self.df['elimination_status_away'] == 1).astype(int)
            self.features_created.extend(['eliminated_home', 'eliminated_away'])
            
            # Desperate (fighting for playoffs)
            self.df['desperate_home'] = ((self.df['elimination_status_home'] == 0) & 
                                         (self.df.get('clinch_status_home', 0) == 0)).astype(int)
            self.df['desperate_away'] = ((self.df['elimination_status_away'] == 0) & 
                                         (self.df.get('clinch_status_away', 0) == 0)).astype(int)
            self.features_created.extend(['desperate_home', 'desperate_away'])
    
    def create_momentum_features(self):
        """Winning/losing streaks and recent form."""
        print("   Momentum Features...")
        
        # Check for streak columns or calculate from W/L data
        if 'STRK' in self.df.columns:
            # Parse streak strings like "W5", "L3"
            self.df['streak_value'] = self.df['STRK'].str.extract(r'(\d+)').astype(float).fillna(0)
            self.df['streak_type'] = self.df['STRK'].str.extract(r'([WL])').fillna('N')
            self.df['momentum'] = np.where(
                self.df['streak_type'] == 'W',
                self.df['streak_value'],
                -self.df['streak_value']
            )
            self.features_created.append('momentum')
        
        # Check for home/away versions
        if 'STRK_home' in self.df.columns:
            self.df['streak_value_home'] = self.df['STRK_home'].str.extract(r'(\d+)').astype(float).fillna(0)
            self.df['streak_type_home'] = self.df['STRK_home'].str.extract(r'([WL])').fillna('N')
            self.df['momentum_home'] = np.where(
                self.df['streak_type_home'] == 'W',
                self.df['streak_value_home'],
                -self.df['streak_value_home']
            )
            
            self.df['streak_value_away'] = self.df['STRK_away'].str.extract(r'(\d+)').astype(float).fillna(0)
            self.df['streak_type_away'] = self.df['STRK_away'].str.extract(r'([WL])').fillna('N')
            self.df['momentum_away'] = np.where(
                self.df['streak_type_away'] == 'W',
                self.df['streak_value_away'],
                -self.df['streak_value_away']
            )
            
            self.features_created.extend(['momentum_home', 'momentum_away'])
            
            # Momentum advantage
            self.df['momentum_diff'] = self.df['momentum_home'] - self.df['momentum_away']
            self.features_created.append('momentum_diff')
        
        # Recent form (L10 = last 10 games record)
        if 'L10' in self.df.columns:
            # Parse "8-2-0" format (W-L-OT)
            self.df['L10_wins'] = self.df['L10'].str.split('-').str[0].astype(float)
            self.df['recent_form'] = self.df['L10_wins'] / 10.0
            self.features_created.append('recent_form')
        
        if 'L10_home' in self.df.columns:
            self.df['L10_wins_home'] = self.df['L10_home'].str.split('-').str[0].astype(float)
            self.df['L10_wins_away'] = self.df['L10_away'].str.split('-').str[0].astype(float)
            self.df['recent_form_home'] = self.df['L10_wins_home'] / 10.0
            self.df['recent_form_away'] = self.df['L10_wins_away'] / 10.0
            self.features_created.extend(['recent_form_home', 'recent_form_away'])
            
            self.df['form_advantage'] = self.df['recent_form_home'] - self.df['recent_form_away']
            self.features_created.append('form_advantage')
    
    def create_goalie_features(self):
        """Goalie performance metrics."""
        print("   Goalie Features...")
        
        # If save percentage or GAA available
        if 'save_pct' in self.df.columns:
            # Save percentage (single team format)
            self.df['save_pct'] = self.df['save_pct'].fillna(0.900)
            self.features_created.append('save_pct')
        
        if 'save_pct_home' in self.df.columns:
            # Save percentage (home/away format)
            self.df['save_pct_home'] = self.df['save_pct_home'].fillna(0.900)  # League avg
            self.df['save_pct_away'] = self.df['save_pct_away'].fillna(0.900)
            
            # Goalie advantage
            self.df['goalie_advantage'] = self.df['save_pct_home'] - self.df['save_pct_away']
            self.features_created.append('goalie_advantage')
            
            # Elite goalie (>0.920 is Vezina-level)
            self.df['elite_goalie_home'] = (self.df['save_pct_home'] > 0.920).astype(int)
            self.df['elite_goalie_away'] = (self.df['save_pct_away'] > 0.920).astype(int)
            self.features_created.extend(['elite_goalie_home', 'elite_goalie_away'])
        
        # Goals against average
        if 'GAA' in self.df.columns:
            # Single team format
            gaa = self.df['GAA']
            self.df['gaa_normalized'] = 2.7 / (gaa + 0.01)
            self.features_created.append('gaa_normalized')
        elif 'GA' in self.df.columns and 'GP' in self.df.columns:
            # Calculate from totals (single team)
            gaa = self.df['GA'] / self.df['GP']
            self.df['gaa_normalized'] = 2.7 / (gaa + 0.01)
            self.features_created.append('gaa_normalized')
        
        # Home/away format
        if 'GAA_home' in self.df.columns:
            gaa_home = self.df['GAA_home']
            gaa_away = self.df['GAA_away']
            self.df['gaa_normalized_home'] = 2.7 / (gaa_home + 0.01)
            self.df['gaa_normalized_away'] = 2.7 / (gaa_away + 0.01)
            self.features_created.extend(['gaa_normalized_home', 'gaa_normalized_away'])
        elif 'GA_home' in self.df.columns and 'GP_home' in self.df.columns:
            gaa_home = self.df['GA_home'] / self.df['GP_home']
            gaa_away = self.df['GA_away'] / self.df['GP_away']
            self.df['gaa_normalized_home'] = 2.7 / (gaa_home + 0.01)
            self.df['gaa_normalized_away'] = 2.7 / (gaa_away + 0.01)
            self.features_created.extend(['gaa_normalized_home', 'gaa_normalized_away'])
    
    def create_special_teams_features(self):
        """Power play and penalty kill efficiency."""
        print("   Special Teams Features...")
        
        if 'PP_pct' in self.df.columns:
            # Power play percentage
            self.df['pp_home'] = self.df['PP_pct_home'].fillna(0.20)
            self.df['pp_away'] = self.df['PP_pct_away'].fillna(0.20)
            
            # PP advantage
            self.df['pp_advantage'] = self.df['pp_home'] - self.df['pp_away']
            self.features_created.append('pp_advantage')
        
        if 'PK_pct' in self.df.columns:
            # Penalty kill percentage
            self.df['pk_home'] = self.df['PK_pct_home'].fillna(0.80)
            self.df['pk_away'] = self.df['PK_pct_away'].fillna(0.80)
            
            # PK advantage
            self.df['pk_advantage'] = self.df['pk_home'] - self.df['pk_away']
            self.features_created.append('pk_advantage')
            
            # Special teams dominance
            if 'PP_pct' in self.df.columns:
                self.df['special_teams_home'] = self.df['pp_home'] + self.df['pk_home']
                self.df['special_teams_away'] = self.df['pp_away'] + self.df['pk_away']
                self.features_created.extend(['special_teams_home', 'special_teams_away'])
    
    def create_scoring_features(self):
        """Offensive and defensive capabilities."""
        print("   Scoring Features...")
        
        if 'GF' in self.df.columns and 'GA' in self.df.columns and 'GP' in self.df.columns:
            # Goals per game
            self.df['gpg'] = self.df['GF'] / self.df['GP']
            self.df['ga_per_game'] = self.df['GA'] / self.df['GP']
            self.df['goal_diff'] = self.df['gpg'] - self.df['ga_per_game']
            self.features_created.extend(['gpg', 'ga_per_game', 'goal_diff'])
        
        if 'GF_home' in self.df.columns and 'GA_home' in self.df.columns and 'GP_home' in self.df.columns:
            # Goals per game (home/away)
            self.df['gpg_home'] = self.df['GF_home'] / self.df['GP_home']
            self.df['gpg_away'] = self.df['GF_away'] / self.df['GP_away']
            self.df['ga_per_game_home'] = self.df['GA_home'] / self.df['GP_home']
            self.df['ga_per_game_away'] = self.df['GA_away'] / self.df['GP_away']
            
            self.features_created.extend([
                'gpg_home', 'gpg_away', 
                'ga_per_game_home', 'ga_per_game_away'
            ])
            
            # Goal differential per game
            self.df['goal_diff_home'] = self.df['gpg_home'] - self.df['ga_per_game_home']
            self.df['goal_diff_away'] = self.df['gpg_away'] - self.df['ga_per_game_away']
            self.features_created.extend(['goal_diff_home', 'goal_diff_away'])
            
            # Scoring advantage
            self.df['offense_advantage'] = self.df['gpg_home'] - self.df['gpg_away']
            self.df['defense_advantage'] = self.df['ga_per_game_away'] - self.df['ga_per_game_home']
            self.features_created.extend(['offense_advantage', 'defense_advantage'])
    
    def create_rivalry_features(self):
        """Division and rivalry intensity."""
        print("   Rivalry Features...")
        
        if 'division' in self.df.columns:
            # Same division (higher intensity)
            self.df['division_rival'] = (self.df['division_home'] == self.df['division_away']).astype(int)
            self.features_created.append('division_rival')
        
        if 'conference' in self.df.columns:
            # Same conference (playoff implications)
            self.df['conference_game'] = (self.df['conference_home'] == self.df['conference_away']).astype(int)
            self.features_created.append('conference_game')
        
        if 'division_tier' in self.df.columns:
            # Division strength differential (D1 vs D3)
            tier_map = {'D1': 3, 'D2': 2, 'D3': 1}
            self.df['tier_home'] = self.df['division_tier_home'].map(tier_map).fillna(2)
            self.df['tier_away'] = self.df['division_tier_away'].map(tier_map).fillna(2)
            self.df['tier_advantage'] = self.df['tier_home'] - self.df['tier_away']
            self.features_created.append('tier_advantage')
    
    def create_temporal_features(self):
        """Time-of-season effects."""
        print("   Temporal Features...")
        
        if 'game_number' in self.df.columns or 'GP' in self.df.columns:
            # Season progress (0.0 to 1.0)
            if 'game_number' in self.df.columns:
                self.df['season_progress'] = self.df['game_number'] / 82.0
            else:
                self.df['season_progress'] = self.df['GP_home'] / 82.0
            
            self.features_created.append('season_progress')
            
            # Early season inconsistency (first 20 games)
            self.df['early_season'] = (self.df['season_progress'] < 0.25).astype(int)
            
            # Playoff push (last 20 games)
            self.df['playoff_push'] = (self.df['season_progress'] > 0.75).astype(int)
            
            # Mid-season (games 20-60)
            self.df['mid_season'] = ((self.df['season_progress'] >= 0.25) & 
                                     (self.df['season_progress'] <= 0.75)).astype(int)
            
            self.features_created.extend(['early_season', 'playoff_push', 'mid_season'])
        
        if 'date' in self.df.columns:
            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(self.df['date']):
                self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')
            
            # Month effects
            self.df['month'] = self.df['date'].dt.month
            self.df['december_fatigue'] = (self.df['month'] == 12).astype(int)
            self.df['march_intensity'] = (self.df['month'] == 3).astype(int)
            self.df['april_playoffs'] = (self.df['month'] == 4).astype(int)
            
            self.features_created.extend(['december_fatigue', 'march_intensity', 'april_playoffs'])
    
    def create_arena_features(self):
        """Home ice advantage and venue-specific factors."""
        print("   Arena Features...")
        
        if 'HOME' in self.df.columns:
            # Parse home record "20-3-2" (W-L-OT)
            try:
                home_col = 'HOME_home' if 'HOME_home' in self.df.columns else 'HOME'
                home_wins = self.df[home_col].str.split('-').str[0].astype(float)
                home_games = self.df[home_col].str.split('-').apply(lambda x: sum([int(i) for i in x]))
                self.df['home_win_pct'] = home_wins / home_games
                self.features_created.append('home_win_pct')
            except:
                pass
        
        if 'AWAY' in self.df.columns:
            # Away record
            try:
                away_col = 'AWAY_away' if 'AWAY_away' in self.df.columns else 'AWAY'
                away_wins = self.df[away_col].str.split('-').str[0].astype(float)
                away_games = self.df[away_col].str.split('-').apply(lambda x: sum([int(i) for i in x]))
                self.df['away_win_pct'] = away_wins / away_games
                self.features_created.append('away_win_pct')
                
                # Home/away differential
                if 'home_win_pct' in self.df.columns:
                    self.df['home_away_diff'] = self.df['home_win_pct'] - self.df['away_win_pct']
                    self.features_created.append('home_away_diff')
            except:
                pass
        
        # Altitude (Denver Avalanche has significant advantage)
        if 'team_name' in self.df.columns:
            self.df['altitude'] = (self.df['team_name'].str.contains('Avalanche', na=False)).astype(int)
            self.features_created.append('altitude')
        if 'team_name_home' in self.df.columns:
            self.df['altitude_home'] = (self.df['team_name_home'].str.contains('Avalanche', na=False)).astype(int)
            self.df['altitude_away'] = (self.df['team_name_away'].str.contains('Avalanche', na=False)).astype(int)
            self.features_created.extend(['altitude_home', 'altitude_away'])
    
    def create_advanced_stats(self):
        """PDO, Corsi, Fenwick proxies."""
        print("   Advanced Stats...")
        
        if 'GF' in self.df.columns and 'shots' in self.df.columns:
            # Shooting percentage
            self.df['shooting_pct_home'] = self.df['GF_home'] / (self.df['shots_home'] + 1)
            self.df['shooting_pct_away'] = self.df['GF_away'] / (self.df['shots_away'] + 1)
            self.features_created.extend(['shooting_pct_home', 'shooting_pct_away'])
        
        if 'GA' in self.df.columns and 'shots_against' in self.df.columns:
            # Save percentage (team)
            self.df['team_save_pct_home'] = 1 - (self.df['GA_home'] / (self.df['shots_against_home'] + 1))
            self.df['team_save_pct_away'] = 1 - (self.df['GA_away'] / (self.df['shots_against_away'] + 1))
            self.features_created.extend(['team_save_pct_home', 'team_save_pct_away'])
            
            # PDO (shooting% + save%, measures luck)
            if 'GF' in self.df.columns and 'shots' in self.df.columns:
                self.df['pdo_home'] = self.df['shooting_pct_home'] + self.df['team_save_pct_home']
                self.df['pdo_away'] = self.df['shooting_pct_away'] + self.df['team_save_pct_away']
                self.features_created.extend(['pdo_home', 'pdo_away'])
                
                # Luck indicator (PDO >1.02 = lucky, <0.98 = unlucky)
                self.df['lucky_home'] = (self.df['pdo_home'] > 1.02).astype(int)
                self.df['unlucky_home'] = (self.df['pdo_home'] < 0.98).astype(int)
                self.df['lucky_away'] = (self.df['pdo_away'] > 1.02).astype(int)
                self.df['unlucky_away'] = (self.df['pdo_away'] < 0.98).astype(int)
                self.features_created.extend([
                    'lucky_home', 'unlucky_home', 
                    'lucky_away', 'unlucky_away'
                ])
    
    def create_interaction_features(self):
        """Key feature interactions for hockey dynamics."""
        print("   Interaction Features...")
        
        # Rest advantage × momentum
        if 'rest_advantage' in self.df.columns and 'momentum_diff' in self.df.columns:
            self.df['rest_momentum'] = self.df['rest_advantage'] * self.df['momentum_diff']
            self.features_created.append('rest_momentum')
        
        # Fatigue × travel
        if 'fatigue_index_away' in self.df.columns and 'travel_distance_away' in self.df.columns:
            self.df['travel_fatigue'] = self.df['fatigue_index_away'] * self.df['travel_distance_away']
            self.features_created.append('travel_fatigue')
        
        # Injuries × back-to-back
        if 'injuries_away' in self.df.columns and 'back_to_back_away' in self.df.columns:
            self.df['injury_b2b'] = self.df['injuries_away'] * self.df['back_to_back_away']
            self.features_created.append('injury_b2b')
        
        # Playoff desperation × division rival
        if 'desperate_home' in self.df.columns and 'division_rival' in self.df.columns:
            self.df['desperate_rival'] = self.df['desperate_home'] * self.df['division_rival']
            self.features_created.append('desperate_rival')
        
        # Home advantage × recent form
        if 'home_win_pct' in self.df.columns and 'recent_form_home' in self.df.columns:
            self.df['home_form'] = self.df['home_win_pct'] * self.df['recent_form_home']
            self.features_created.append('home_form')
        
        # Special teams × power play opportunities (if available)
        if 'special_teams_home' in self.df.columns and 'offense_advantage' in self.df.columns:
            self.df['special_offense'] = self.df['special_teams_home'] * self.df['offense_advantage']
            self.features_created.append('special_offense')


def engineer_hockey_features(data_path, output_path=None):
    """
    Load data, engineer hockey features, save enhanced dataset.
    
    Args:
        data_path: Path to input CSV with game data
        output_path: Path to save engineered dataset (optional)
    
    Returns:
        DataFrame with original + engineered features
    """
    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Original shape: {df.shape}")
    
    # Engineer features
    engineer = HockeyFeatureEngineer(df)
    df_enhanced, features = engineer.create_all_features()
    
    print(f"\nEnhanced shape: {df_enhanced.shape}")
    print(f"New features: {len(features)}")
    
    # Save if output path provided
    if output_path:
        df_enhanced.to_csv(output_path, index=False)
        print(f"\n[OK] Saved to: {output_path}")
    
    return df_enhanced, features


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python hockey_feature_engineering.py <input_csv> [output_csv]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    df, features = engineer_hockey_features(input_file, output_file)
    
    print("\n" + "="*80)
    print("FEATURE SUMMARY")
    print("="*80)
    for i, feat in enumerate(features, 1):
        print(f"  {i:3d}. {feat}")
