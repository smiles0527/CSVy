"""
================================================================================
  BASELINE MODELS - Hockey Game Prediction
================================================================================

WHAT THIS FILE DOES (in plain English):
--------------------------------------
This file contains several "baseline" models that predict how many goals each
team will score in a hockey game. A "baseline" is a simple starting point—
we use these to compare against fancier models. If a fancy model can't beat
these simple ones, it's probably not worth using.

HOW MODELS WORK (the two steps):
--------------------------------
1. FIT (train): The model looks at past games and learns patterns (e.g., "Team A
   scores 3.2 goals per game on average"). This is like studying before a test.
2. PREDICT: For a new game (e.g., Team A vs Team B at home), the model guesses
   how many goals each team will score. This is like taking the test.

THE 9 MODELS IN THIS FILE:
--------------------------
1. GlobalMeanBaseline: Ignores teams entirely. Every game? "Each team scores
   the league average." (e.g., 3.1, 3.1). Simplest possible—any useful model
   should beat this.
2. TeamMeanBaseline: Uses each team's average goals scored and allowed. "Team A
   usually scores 3.5; Team B usually allows 3.0; so Team A might score ~3.25."
3. HomeAwayBaseline: Same idea but splits stats by home vs away. Captures "home
   ice advantage"—teams often do better at home.
4. MovingAverageBaseline: Only looks at the last N games (e.g., 5). Captures
   "hot streaks" and recent form.
5. WeightedHistoryBaseline: Uses all games but recent ones count more. Last game
   might have weight 1.0, the one before 0.9, before that 0.81, etc.
6. PoissonBaseline: A statistical model. Goals are treated as random events
   (like goals per game) with team strengths as multipliers.
7. DixonColesBaseline: Fancy version of Poisson used in pro soccer/hockey. Fits
   attack/defense strengths by iterating until numbers stabilize.
8. BayesianTeamBaseline: Like TeamMean but "shrinks" teams with few games toward
   the league average. Avoids overreacting to small samples.
9. EnsembleBaseline: Combines multiple models. Like asking several experts and
   averaging their opinions (with better experts weighted more).

KEY TERMS (for beginners):
--------------------------
- DataFrame: A table (rows = games, columns = home_team, home_goals, etc.)
- fit(): Train the model on past games. Must call before predict or evaluate.
- predict_goals(): Guess how many goals each team will score in one game.
- RMSE: Average prediction error (lower = better). sqrt(mean of squared errors).
- R²: How much variance we explain (higher = better). 0 = no better than guessing mean.

CODING 101 (if you've never coded):
-----------------------------------
- VARIABLE: A labeled box that holds a value. x = 5 means "put 5 in the box named x."
- = (equals): Means "store this on the right INTO the name on the left." NOT "equals" in the
  math sense. So x = x + 1 means "take x's value, add 1, put result back in x."
- LOOP (for): "Do this for each item." for team in teams: means "go through teams one by one."
- IF/ELSE: "If this is true, do this. Otherwise do that." Like a fork in the road.
- LIST []: An ordered collection. [1, 2, 3]. Can add with .append(). First item is [0].
- DICTIONARY {}: Labeled storage. {"Bruins": 3.2, "Leafs": 2.9} = "Bruins maps to 3.2."
  Use .get(name, default) to look up: "Give me Bruins' value, or 0 if not found."
- FUNCTION def foo(x): A named recipe. You call foo(5) and it runs the recipe with x=5.
- RETURN: "Send this value back to whoever called this function." The function exits here.

Usage:
    from utils.baseline_model import TeamMeanBaseline
    model = TeamMeanBaseline()
    model.fit(games_df)      # Train on past games
    metrics = model.evaluate(test_df)  # See how good predictions are
"""

# -----------------------------------------------------------------------------
# IMPORTS - We bring in tools from other files so we don't have to write
# everything from scratch. Think of these like borrowing a calculator.
# -----------------------------------------------------------------------------
import json        # For reading/writing text files in JSON format
import pickle      # For saving/loading Python objects to disk (model state)
from pathlib import Path   # For handling file paths (e.g., "output/models/model.pkl")

import pandas as pd   # "DataFrames" = tables of data (rows = games, columns = home_team, goals, etc.)
import numpy as np    # Fast math (averages, arrays, sqrt)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# ^ These measure prediction quality: RMSE/MAE = how far off we are, R² = how much variance we explain


# -----------------------------------------------------------------------------
# COLUMN ALIASES - A dictionary (labeled storage)
# -----------------------------------------------------------------------------
# Different data files call the same thing by different names. One file has
# "home_team", another has "home", another "team_home". This dictionary maps
# "what we want" -> [list of names to try]. When we need home_team, we try
# home_team, then home, then team_home until one exists in the data.
# In Python, {} = dictionary, [] = list. 'key': [value1, value2] = one entry.
# -----------------------------------------------------------------------------
COLUMN_ALIASES = {
    'home_team': ['home_team', 'home', 'team_home', 'h_team'],
    'away_team': ['away_team', 'away', 'team_away', 'a_team', 'visitor', 'visiting_team'],
    'home_goals': ['home_goals', 'home_score', 'h_goals', 'goals_home', 'home_pts'],
    'away_goals': ['away_goals', 'away_score', 'a_goals', 'goals_away', 'away_pts', 'visitor_goals'],
    'game_date': ['game_date', 'date', 'Date', 'game_datetime', 'datetime', 'game_time'],
}


def get_value(game, field, default=None):
    """
    Look up one piece of info from a game (e.g., "home_team", "home_goals").
    game = one row (one game). field = what we want. default = what to return if missing.
    """
    # Get the list of column names to try. .get(field, [field]) = "use [field] if not found"
    aliases = COLUMN_ALIASES.get(field, [field])

    for alias in aliases:   # Try each possible column name, one by one
        if alias in game:   # Does this column exist in our game data?
            val = game[alias]   # Grab the value (e.g., "Bruins" or 3)
            if pd.isna(val):   # NaN = Not a Number = missing/blank in the data
                return default
            return val   # Found it! Send it back and stop.
    return default   # Tried all names, none worked. Use the default.


def get_column(df, field):
    """
    df = the whole table. field = what we need (e.g. "home_team").
    Find which column in the table has that data. Return None if no match.
    """
    aliases = COLUMN_ALIASES.get(field, [field])
    for alias in aliases:
        if alias in df.columns:   # df.columns = the table's column headers
            return alias         # This column exists! Return its name.
    return None   # None = "nothing found" in Python


def poisson_win_confidence(home_lam, away_lam):
    """
    Given expected goals for home and away (e.g., 3.2 and 2.8), compute how
    confident we are that one team wins. Uses the Poisson distribution: a
    statistical model for counting rare events (like goals in a game).

    SIMPLE EXPLANATION: We consider all possible final scores (0-0, 1-0, 2-1,
    etc. up to 12-12). For each score, we ask: "How likely is this given our
    expected goals?" Then we add up the probability of every outcome where
    home wins, plus half the probability of ties (hockey games don't really
    end in ties—OT/SO decides). The result is a number between 0.5 and 1.0
    (the confidence of the more likely winner).
    """
    from scipy.stats import poisson

    # lam = "expected goals" (like an average). Poisson needs it > 0. max(x, 0.3) = use 0.3 if x is smaller
    lam_h = max(float(home_lam), 0.3)
    lam_a = max(float(away_lam), 0.3)

    # We'll add up probabilities. Start at 0.
    max_goals = 12
    home_win_prob = 0.0
    draw_prob = 0.0

    # Nested loops: for each possible home score (0,1,...,12), for each possible away score (0,1,...,12)
    for h in range(max_goals + 1):   # range(13) = 0,1,2,...,12
        ph = poisson.pmf(h, lam_h)   # Probability home scores exactly h
        for a in range(max_goals + 1):
            pa = poisson.pmf(a, lam_a)   # Probability away scores exactly a
            if h > a:
                home_win_prob += ph * pa   # Home wins (h>a). Add this outcome's probability.
            elif h == a:
                draw_prob += ph * pa       # Tie (h=a). Add to draw total.

    # ph*pa = "prob of home h AND away a" (we assume independent). += means "add to current total"
    home_win_prob += draw_prob / 2.0   # Split ties 50/50 (OT/SO decides in real hockey)

    # Return the higher confidence: home winning or away winning (must be between 0.5 and 1.0)
    return max(home_win_prob, 1.0 - home_win_prob)


# =============================================================================
# BASELINE MODEL - The "template" that all 9 baseline models follow
# =============================================================================
# CLASS: A blueprint for making objects. Like a cookie cutter—the class defines
# the shape, and each "instance" (model) we create is one cookie. All cookies
# have the same methods (fit, predict_goals, etc.) but their data differs.
# SELF: Inside a class, "self" means "this specific object." self.team_offense
# = "this model's team_offense dictionary," not some other model's.
# =============================================================================

class BaselineModel:
    """
    Abstract base class: the shared interface for all baseline models.
    "Abstract" = we don't use this directly; we use TeamMeanBaseline, etc.
    """

    def __init__(self, params=None):
        """
        __init__ = "initialize." Runs when we create a new model: model = TeamMeanBaseline()
        params or {} means: use params if provided, otherwise use empty dict {}
        """
        self.params = params or {}   # Store settings. or = "use right side if left is empty/None"
        self.is_fitted = False       # We haven't trained yet. Must call fit() first.

    def fit(self, games_df):
        """Train on past games. Each model type does this differently."""
        raise NotImplementedError("Subclasses must implement fit()")
    
    def predict_goals(self, game):
        """
        Predict goals for both teams.
        
        Parameters
        ----------
        game : dict or Series
            Game record with team identifiers
        
        Returns
        -------
        tuple
            (home_goals_pred, away_goals_pred)
        """
        raise NotImplementedError("Subclasses must implement predict_goals()")

    def predict_winner(self, game):
        """
        Predict who wins AND how confident we are (0.5 to 1.0).
        Uses the same Poisson logic as poisson_win_confidence: consider all
        possible scores, add up probabilities where home wins vs away wins,
        return the winner and that team's win probability.
        """
        from scipy.stats import poisson

        home_goals, away_goals = self.predict_goals(game)
        home_team = get_value(game, 'home_team')
        away_team = get_value(game, 'away_team')

        lam_h = max(home_goals, 0.3)   # Poisson needs positive rate
        lam_a = max(away_goals, 0.3)

        max_goals = 12
        home_win_prob = 0.0
        draw_prob = 0.0
        for h in range(max_goals + 1):
            ph = poisson.pmf(h, lam_h)
            for a in range(max_goals + 1):
                pa = poisson.pmf(a, lam_a)
                if h > a:
                    home_win_prob += ph * pa
                elif h == a:
                    draw_prob += ph * pa

        home_win_prob += draw_prob / 2.0   # Split ties (OT decides in reality)

        # Return (winner name, confidence 0.5-1.0)
        if home_win_prob >= 0.5:
            return home_team, home_win_prob
        else:
            return away_team, 1.0 - home_win_prob
    
    def evaluate(self, games_df):
        """
        Check how good our predictions are. We predict each game, compare to
        actual scores, and compute: RMSE (avg error magnitude), MAE (avg
        absolute error), R² (variance explained), win_accuracy (% correct winners).
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before evaluation")

        # Empty lists to collect our predictions and the real answers
        home_preds = []
        away_preds = []
        home_actuals = []
        away_actuals = []

        # iterrows() = "give me each row one by one." _ = row number (we don't use it)
        for _, game in games_df.iterrows():
            home_pred, away_pred = self.predict_goals(game)   # Our guess
            home_preds.append(home_pred)    # .append() = add to end of list
            away_preds.append(away_pred)
            home_actuals.append(get_value(game, 'home_goals', 0))   # Real score (0 if missing)
            away_actuals.append(get_value(game, 'away_goals', 0))

        # RMSE = sqrt(mean squared error). Lower = better.
        # MAE = mean absolute error. Lower = better.
        # R² = how much variance we explain. Higher = better. Need variety in targets.
        home_rmse = np.sqrt(mean_squared_error(home_actuals, home_preds))
        home_mae = mean_absolute_error(home_actuals, home_preds)
        home_r2 = r2_score(home_actuals, home_preds) if len(set(home_actuals)) > 1 else 0.0

        away_rmse = np.sqrt(mean_squared_error(away_actuals, away_preds))
        away_mae = mean_absolute_error(away_actuals, away_preds)
        away_r2 = r2_score(away_actuals, away_preds) if len(set(away_actuals)) > 1 else 0.0

        # Combined: pool home+away predictions into one RMSE/MAE/R²
        all_preds = home_preds + away_preds
        all_actuals = home_actuals + away_actuals
        combined_rmse = np.sqrt(mean_squared_error(all_actuals, all_preds))
        combined_mae = mean_absolute_error(all_actuals, all_preds)
        combined_r2 = r2_score(all_actuals, all_preds) if len(set(all_actuals)) > 1 else 0.0

        # Win accuracy: Did we pick the right winner? zip = "pair up: 1st of each, 2nd of each..."
        # (hp>ap) = we said home wins. (ha>aa) = home actually won. Same? Count it.
        correct_wins = sum(
            1 for hp, ap, ha, aa in zip(home_preds, away_preds, home_actuals, away_actuals)
            if (hp > ap) == (ha > aa)
        )
        win_accuracy = correct_wins / len(home_preds) if home_preds else 0.0   # Divide by total, or 0 if empty

        return {
            'rmse': home_rmse,
            'mae': home_mae,
            'r2': home_r2,
            'away_rmse': away_rmse,
            'away_mae': away_mae,
            'away_r2': away_r2,
            'combined_rmse': combined_rmse,
            'combined_mae': combined_mae,
            'combined_r2': combined_r2,
            'win_accuracy': win_accuracy,
        }
    
    def get_summary(self):
        """Return a dict of model metadata (name, hyperparams, top teams, etc.)."""
        raise NotImplementedError("Subclasses must implement get_summary()")
    
    def save_model(self, filepath):
        """
        Save the trained model to a file. We can load it later without re-training.
        Saves: model type, settings, and all learned numbers (team stats, etc.)
        """
        filepath = Path(filepath)   # Path = object for file paths (handles / vs \ on different systems)
        filepath.parent.mkdir(parents=True, exist_ok=True)   # Create parent folder(s) if missing
        state = {
            'class': self.__class__.__name__,   # So we know which model type this is
            'params': self.params,
            'is_fitted': self.is_fitted,
        }
        for key, value in self.__dict__.items():
            if key not in state:
                state[key] = value   # Add team_offense, global_mean, etc.

        with open(filepath, 'wb') as f:   # 'wb' = write binary (for pickle)
            pickle.dump(state, f)   # Save state to file. Like freezing the model for later.
    
    @classmethod
    def load_model(cls, filepath):
        """
        Load a saved model from disk. We read the file, figure out which model
        type it was (TeamMean, DixonColes, etc.), and recreate it with all its
        learned stats.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        with open(filepath, 'rb') as f:   # 'rb' = read binary
            state = pickle.load(f)   # Load saved data back into memory

        # class_map: "When file says 'TeamMeanBaseline', we need the real TeamMeanBaseline class"
        class_name = state.pop('class', cls.__name__)
        class_map = {
            'GlobalMeanBaseline': GlobalMeanBaseline,
            'TeamMeanBaseline': TeamMeanBaseline,
            'HomeAwayBaseline': HomeAwayBaseline,
            'MovingAverageBaseline': MovingAverageBaseline,
            'WeightedHistoryBaseline': WeightedHistoryBaseline,
            'PoissonBaseline': PoissonBaseline,
            'DixonColesBaseline': DixonColesBaseline,
            'BayesianTeamBaseline': BayesianTeamBaseline,
            'EnsembleBaseline': EnsembleBaseline,
        }
        model_cls = class_map.get(class_name, cls)   # Get the right class, or use default

        # __new__ = create empty object without running __init__. Then fill in saved data.
        model = model_cls.__new__(model_cls)
        for key, value in state.items():   # .items() = (key, value) pairs
            setattr(model, key, value)     # setattr = "set model.key = value"

        return model


# =============================================================================
# MODEL 1: GLOBAL MEAN - The dumbest possible predictor
# =============================================================================
# Ignores teams completely. "Every team scores the league average." If your
# fancier model can't beat this, it's not adding value.
# =============================================================================

class GlobalMeanBaseline(BaselineModel):
    """
    Predicts (league_avg_home_goals, league_avg_away_goals) for EVERY game.
    Bruins vs Coyotes? (3.1, 2.9). Leafs vs Oilers? (3.1, 2.9). Same every time.
    """

    def __init__(self, params=None):
        super().__init__(params)   # super() = "run the parent class's __init__ first"
        self.global_mean_home = None   # None = "nothing yet." We'll fill these in fit().
        self.global_mean_away = None

    def fit(self, games_df):
        """Compute: What's the average home goals? Average away goals? Done."""
        home_col = get_column(games_df, 'home_goals')   # Which column has home goals?
        away_col = get_column(games_df, 'away_goals')

        if home_col is None or away_col is None:   # or = either one missing?
            raise ValueError("DataFrame must contain home_goals and away_goals columns")
        # raise = stop and show error. We can't continue without the right columns.

        self.global_mean_home = games_df[home_col].mean()   # .mean() = average of column
        self.global_mean_away = games_df[away_col].mean()
        self.n_games = len(games_df)   # len() = how many rows (games)
        self.is_fitted = True   # Now we're trained. predict() can run.

        return self   # Return self so we can chain: model.fit(df).predict(game)
    
    def predict_goals(self, game):
        """We ignore the game entirely. Just spit back the league averages."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")   # Can't predict without training!
        return self.global_mean_home, self.global_mean_away   # Tuple: (home_guess, away_guess)
    
    def get_summary(self):
        """Get model summary."""
        # Return a dictionary: {'model': name, 'global_mean_home': 3.1, ...}
        return {
            'model': 'GlobalMeanBaseline',
            'global_mean_home': round(self.global_mean_home, 3),
            'global_mean_away': round(self.global_mean_away, 3),
            'n_games_trained': self.n_games
        }


# =============================================================================
# MODEL 2: TEAM MEAN - Use each team's scoring and defensive stats
# =============================================================================
# For each team we track: (1) How many goals they usually score? (2) How many
# they usually allow? For A vs B at A's home: home_goals ≈ (A's offense + B's
# defense) / 2. Idea: A scores based on their attack + how leaky B's defense is.
# =============================================================================

class TeamMeanBaseline(BaselineModel):
    """
    home_goals = (home_offense + away_defense) / 2
    away_goals = (away_offense + home_defense) / 2
    Classic sports analytics formula.
    """
    
    def __init__(self, params=None):
        super().__init__(params)
        self.team_offense = {}   # {} = empty dict. Will become {"Bruins": 3.2, "Leafs": 2.9, ...}
        self.team_defense = {}   # Same idea for goals allowed
        self.global_mean = None  # If we see a new team we've never trained on, use this

    def fit(self, games_df):
        """
        For each team: Sum up all goals they scored. Sum up all goals they
        allowed. Divide by games played. Store as offense and defense ratings.
        """
        home_team_col = get_column(games_df, 'home_team')
        away_team_col = get_column(games_df, 'away_team')
        home_goals_col = get_column(games_df, 'home_goals')
        away_goals_col = get_column(games_df, 'away_goals')

        goals_for = {}       # Running total: Bruins have scored X so far
        goals_against = {}  # Running total: Bruins have allowed Y so far
        games_played = {}   # How many games has each team played?

        for _, game in games_df.iterrows():
            home_team = game[home_team_col]
            away_team = game[away_team_col]
            home_goals = game[home_goals_col]
            away_goals = game[away_goals_col]

            goals_for[home_team] = goals_for.get(home_team, 0) + home_goals
            goals_against[home_team] = goals_against.get(home_team, 0) + away_goals
            games_played[home_team] = games_played.get(home_team, 0) + 1

            goals_for[away_team] = goals_for.get(away_team, 0) + away_goals
            goals_against[away_team] = goals_against.get(away_team, 0) + home_goals
            games_played[away_team] = games_played.get(away_team, 0) + 1

        # Turn totals into averages: total_goals / games_played
        for team in games_played:
            self.team_offense[team] = goals_for[team] / games_played[team]
            self.team_defense[team] = goals_against[team] / games_played[team]

        self.global_mean = games_df[home_goals_col].mean()   # For teams not in training
        self.n_teams = len(games_played)
        self.n_games = len(games_df)
        self.is_fitted = True
        
        return self
    
    def predict_goals(self, game):
        """home = (home offense + away defense)/2. away = (away offense + home defense)/2"""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        home_team = get_value(game, 'home_team')
        away_team = get_value(game, 'away_team')

        # Use global_mean for teams not in training data
        home_offense = self.team_offense.get(home_team, self.global_mean)
        home_defense = self.team_defense.get(home_team, self.global_mean)
        away_offense = self.team_offense.get(away_team, self.global_mean)
        away_defense = self.team_defense.get(away_team, self.global_mean)

        home_goals = (home_offense + away_defense) / 2
        away_goals = (away_offense + home_defense) / 2

        return home_goals, away_goals
    
    def get_summary(self):
        """Get model summary."""
        return {
            'model': 'TeamMeanBaseline',
            'n_teams': self.n_teams,
            'n_games_trained': self.n_games,
            'global_mean': round(self.global_mean, 3),
            'top_offense': sorted(self.team_offense.items(), key=lambda x: x[1], reverse=True)[:5],
            'top_defense': sorted(self.team_defense.items(), key=lambda x: x[1])[:5]  # Lower is better
        }


# =============================================================================
# MODEL 3: HOME/AWAY - Account for home ice advantage
# =============================================================================
# Teams often score more at home and allow fewer. So we track separate stats:
# "Goals when at home" vs "goals when away." When predicting, we use the right
# stats: home team's at-home offense vs away team's on-the-road defense.
# =============================================================================

class HomeAwayBaseline(BaselineModel):
    """Same offense/defense idea as TeamMean, but split by home vs away."""
    
    def __init__(self, params=None):
        super().__init__(params)
        self.home_offense = {}   # Team -> avg goals scored when at home
        self.away_offense = {}   # Team -> avg goals scored when away
        self.home_defense = {}   # Team -> avg goals allowed when at home
        self.away_defense = {}   # Team -> avg goals allowed when away
        self.global_home_mean = None
        self.global_away_mean = None

    def fit(self, games_df):
        """For each team: When they're HOME, goals for/against. When AWAY, goals for/against."""
        home_team_col = get_column(games_df, 'home_team')
        away_team_col = get_column(games_df, 'away_team')
        home_goals_col = get_column(games_df, 'home_goals')
        away_goals_col = get_column(games_df, 'away_goals')

        home_goals_for = {}
        home_goals_against = {}
        home_games = {}
        away_goals_for = {}
        away_goals_against = {}
        away_games = {}

        for _, game in games_df.iterrows():
            home_team = game[home_team_col]
            away_team = game[away_team_col]
            home_goals = game[home_goals_col]
            away_goals = game[away_goals_col]

            home_goals_for[home_team] = home_goals_for.get(home_team, 0) + home_goals
            home_goals_against[home_team] = home_goals_against.get(home_team, 0) + away_goals
            home_games[home_team] = home_games.get(home_team, 0) + 1

            away_goals_for[away_team] = away_goals_for.get(away_team, 0) + away_goals
            away_goals_against[away_team] = away_goals_against.get(away_team, 0) + home_goals
            away_games[away_team] = away_games.get(away_team, 0) + 1

        for team in set(list(home_games.keys()) + list(away_games.keys())):
            if team in home_games and home_games[team] > 0:
                self.home_offense[team] = home_goals_for[team] / home_games[team]
                self.home_defense[team] = home_goals_against[team] / home_games[team]

            if team in away_games and away_games[team] > 0:
                self.away_offense[team] = away_goals_for[team] / away_games[team]
                self.away_defense[team] = away_goals_against[team] / away_games[team]

        self.global_home_mean = games_df[home_goals_col].mean()
        self.global_away_mean = games_df[away_goals_col].mean()
        self.home_advantage = self.global_home_mean - self.global_away_mean
        self.n_games = len(games_df)
        self.is_fitted = True
        
        return self
    
    def predict_goals(self, game):
        """
        Match home-team-at-home stats with away-team-on-road stats, then average.
        This naturally incorporates home ice advantage.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        home_team = get_value(game, 'home_team')
        away_team = get_value(game, 'away_team')

        # Home: home team's scoring at home vs away team's defense on road
        home_off = self.home_offense.get(home_team, self.global_home_mean)
        away_def = self.away_defense.get(away_team, self.global_home_mean)

        # Away: away team's scoring on road vs home team's defense at home
        away_off = self.away_offense.get(away_team, self.global_away_mean)
        home_def = self.home_defense.get(home_team, self.global_away_mean)

        home_goals = (home_off + away_def) / 2
        away_goals = (away_off + home_def) / 2

        return home_goals, away_goals
    
    def get_summary(self):
        """Get model summary."""
        return {
            'model': 'HomeAwayBaseline',
            'n_games_trained': self.n_games,
            'global_home_mean': round(self.global_home_mean, 3),
            'global_away_mean': round(self.global_away_mean, 3),
            'home_advantage': round(self.home_advantage, 3)
        }


# =============================================================================
# MODEL 4: MOVING AVERAGE - Recent form matters
# =============================================================================
# Ignores old games. Only looks at the last N games (e.g., 5). Captures hot
# streaks, slumps, roster changes. "How has Team A done lately?"
# =============================================================================

class MovingAverageBaseline(BaselineModel):
    """
    Uses only the last `window` games (default 5) for each team.
    params = {'window': 10} to use last 10 games instead.
    """
    
    def __init__(self, params=None):
        super().__init__(params)
        self.window = self.params.get('window', 5)
        self.team_history = {}   # Each team: list of (goals_scored, goals_allowed) per game
        self.global_mean = None

    def fit(self, games_df):
        """Store every game in order. Later we'll use only the last N for each team."""
        home_team_col = get_column(games_df, 'home_team')
        away_team_col = get_column(games_df, 'away_team')
        home_goals_col = get_column(games_df, 'home_goals')
        away_goals_col = get_column(games_df, 'away_goals')

        for _, game in games_df.iterrows():
            home_team = game[home_team_col]
            away_team = game[away_team_col]
            home_goals = game[home_goals_col]
            away_goals = game[away_goals_col]

            # "not in" = we've never seen this team before. Create a blank list for them.
            if home_team not in self.team_history:
                self.team_history[home_team] = []
            if away_team not in self.team_history:
                self.team_history[away_team] = []

            # .append() = add to end of list. Building a history: game1, game2, game3...
            self.team_history[home_team].append((home_goals, away_goals))
            self.team_history[away_team].append((away_goals, home_goals))

        self.global_mean = games_df[home_goals_col].mean()
        self.n_games = len(games_df)
        self.is_fitted = True
        
        return self
    
    def _get_recent_avg(self, team):
        """Get average goals-for and goals-against from the last N games."""
        if team not in self.team_history or len(self.team_history[team]) == 0:
            return self.global_mean, self.global_mean

        # [1,2,3,4,5][-2:] gives [4,5]. Negative index = count from end. -2: = "last 2"
        recent = self.team_history[team][-self.window:]
        # g = (3, 2). g[0]=3, g[1]=2. So [g[0] for g in recent] = list of goals-scored
        avg_for = np.mean([g[0] for g in recent])
        avg_against = np.mean([g[1] for g in recent])

        return avg_for, avg_against
    
    def predict_goals(self, game):
        """Predict based on recent game window."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        home_team = get_value(game, 'home_team')
        away_team = get_value(game, 'away_team')
        
        home_off, home_def = self._get_recent_avg(home_team)
        away_off, away_def = self._get_recent_avg(away_team)
        
        home_goals = (home_off + away_def) / 2
        away_goals = (away_off + home_def) / 2
        
        return home_goals, away_goals
    
    def get_summary(self):
        """Get model summary."""
        return {
            'model': 'MovingAverageBaseline',
            'window': self.window,
            'n_games_trained': self.n_games,
            'n_teams': len(self.team_history),
            'global_mean': round(self.global_mean, 3)
        }


# =============================================================================
# MODEL 5: WEIGHTED HISTORY - Recent games count more
# =============================================================================
# Uses ALL games but gives more weight to recent ones. With decay=0.9: most
# recent game has weight 1.0, one before that 0.9, two before 0.81, etc.
# Balances "recent form" with "enough data."
# =============================================================================

class WeightedHistoryBaseline(BaselineModel):
    """
    decay=0.9: Last game weight 1.0, second-to-last 0.9, third 0.81, fourth 0.73...
    Older games still matter but less. params = {'decay': 0.95} for slower decay.
    """
    
    def __init__(self, params=None):
        super().__init__(params)
        self.decay = self.params.get('decay', 0.9)
        self.team_history = {}
        self.global_mean = None

    def fit(self, games_df):
        """Build chronological game history per team (same structure as MovingAverageBaseline)."""
        home_team_col = get_column(games_df, 'home_team')
        away_team_col = get_column(games_df, 'away_team')
        home_goals_col = get_column(games_df, 'home_goals')
        away_goals_col = get_column(games_df, 'away_goals')
        
        for _, game in games_df.iterrows():
            home_team = game[home_team_col]
            away_team = game[away_team_col]
            home_goals = game[home_goals_col]
            away_goals = game[away_goals_col]
            
            if home_team not in self.team_history:
                self.team_history[home_team] = []
            if away_team not in self.team_history:
                self.team_history[away_team] = []
            
            self.team_history[home_team].append((home_goals, away_goals))
            self.team_history[away_team].append((away_goals, home_goals))
        
        self.global_mean = games_df[home_goals_col].mean()
        self.n_games = len(games_df)
        self.is_fitted = True
        
        return self
    
    def _get_weighted_avg(self, team):
        """
        Weighted average: newest game × 1.0, next × decay, next × decay², ...
        Then divide by sum of weights so it's a proper average.
        """
        if team not in self.team_history or len(self.team_history[team]) == 0:
            return self.global_mean, self.global_mean

        history = self.team_history[team]
        n = len(history)

        weighted_for = 0
        weighted_against = 0
        total_weight = 0

        # enumerate gives (index, item). i=0 for first (oldest), i=n-1 for last (newest)
        for i, (gf, ga) in enumerate(history):
            weight = self.decay ** (n - 1 - i)   # decay^0=1 for newest. decay^1, decay^2... for older
            weighted_for += gf * weight
            weighted_against += ga * weight
            total_weight += weight

        return weighted_for / total_weight, weighted_against / total_weight
    
    def predict_goals(self, game):
        """Predict using weighted historical averages."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        home_team = get_value(game, 'home_team')
        away_team = get_value(game, 'away_team')
        
        home_off, home_def = self._get_weighted_avg(home_team)
        away_off, away_def = self._get_weighted_avg(away_team)
        
        home_goals = (home_off + away_def) / 2
        away_goals = (away_off + home_def) / 2
        
        return home_goals, away_goals
    
    def get_summary(self):
        """Get model summary."""
        return {
            'model': 'WeightedHistoryBaseline',
            'decay': self.decay,
            'n_games_trained': self.n_games,
            'n_teams': len(self.team_history),
            'global_mean': round(self.global_mean, 3)
        }


# =============================================================================
# MODEL 6: POISSON - Statistical model for goal counts
# =============================================================================
# Goals in a game are "count" data (0, 1, 2, 3...). The Poisson distribution
# is the standard model for counts. We estimate: attack strength (how much
# above/below league average does this team score?), defense strength (how
# much do they allow?), and home advantage. Expected goals = league_avg ×
# attack × opponent_defense × home_factor.
# =============================================================================

class PoissonBaseline(BaselineModel):
    """
    Attack/defense are RATIOS to league average. 1.2 = 20% above avg, 0.8 = 20% below.
    Home factor = home_goals_avg / away_goals_avg (usually > 1).
    """
    
    def __init__(self, params=None):
        super().__init__(params)
        self.attack_strength = {}  # Team -> attack multiplier (relative to league avg)
        self.defense_strength = {}  # Team -> defense multiplier (relative to league avg)
        self.league_avg = None      # League-wide mean goals
        self.home_factor = None    # Home advantage multiplier (home_avg / away_avg)

    def fit(self, games_df):
        """Attack = team's avg goals / league avg. Defense = team's avg allowed / league avg."""
        home_team_col = get_column(games_df, 'home_team')
        away_team_col = get_column(games_df, 'away_team')
        home_goals_col = get_column(games_df, 'home_goals')
        away_goals_col = get_column(games_df, 'away_goals')

        self.league_avg = games_df[home_goals_col].mean()
        self.home_factor = games_df[home_goals_col].mean() / games_df[away_goals_col].mean()
        # home_factor > 1 means home teams score more on average
        goals_for = {}
        goals_against = {}
        games_played = {}
        
        for _, game in games_df.iterrows():
            home_team = game[home_team_col]
            away_team = game[away_team_col]
            home_goals = game[home_goals_col]
            away_goals = game[away_goals_col]
            
            # Home team
            goals_for[home_team] = goals_for.get(home_team, 0) + home_goals
            goals_against[home_team] = goals_against.get(home_team, 0) + away_goals
            games_played[home_team] = games_played.get(home_team, 0) + 1
            
            # Away team
            goals_for[away_team] = goals_for.get(away_team, 0) + away_goals
            goals_against[away_team] = goals_against.get(away_team, 0) + home_goals
            games_played[away_team] = games_played.get(away_team, 0) + 1
        
        for team in games_played:
            avg_for = goals_for[team] / games_played[team]
            avg_against = goals_against[team] / games_played[team]

            self.attack_strength[team] = avg_for / self.league_avg
            self.defense_strength[team] = avg_against / self.league_avg
        
        self.n_games = len(games_df)
        self.n_teams = len(games_played)
        self.is_fitted = True
        
        return self
    
    def predict_goals(self, game):
        """Expected goals = league_avg × attack × opponent_defense × home_factor (or ÷ for away)."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        home_team = get_value(game, 'home_team')
        away_team = get_value(game, 'away_team')

        # Unknown teams default to league average (multiplier = 1.0)
        home_attack = self.attack_strength.get(home_team, 1.0)
        home_defense = self.defense_strength.get(home_team, 1.0)
        away_attack = self.attack_strength.get(away_team, 1.0)
        away_defense = self.defense_strength.get(away_team, 1.0)

        home_goals = self.league_avg * home_attack * away_defense * self.home_factor
        away_goals = self.league_avg * away_attack * home_defense / self.home_factor

        return home_goals, away_goals
    
    def get_summary(self):
        """Get model summary."""
        return {
            'model': 'PoissonBaseline',
            'league_avg': round(self.league_avg, 3),
            'home_factor': round(self.home_factor, 3),
            'n_teams': self.n_teams,
            'n_games_trained': self.n_games,
            'top_attack': sorted(self.attack_strength.items(), key=lambda x: x[1], reverse=True)[:5],
            'top_defense': sorted(self.defense_strength.items(), key=lambda x: x[1])[:5]
        }


def compare_baselines(games_df, test_df=None, models=None):
    """
    Compare all baselines side-by-side. Trains each on 80% of data, tests on 20%,
    returns a table with RMSE, MAE, R², win accuracy for each model. If you don't
    pass test_df, we automatically use the last 20% of games_df as the test set.
    """
    if test_df is None:
        # int() = round down. len(df)*0.8 = 80% of rows. iloc = "rows by position"
        split_idx = int(len(games_df) * 0.8)
        train_df = games_df.iloc[:split_idx]
        test_df = games_df.iloc[split_idx:]
    else:
        train_df = games_df

    if models is None:
        models = [
            GlobalMeanBaseline(),
            TeamMeanBaseline(),
            HomeAwayBaseline(),
            MovingAverageBaseline({'window': 5}),
            MovingAverageBaseline({'window': 10}),
            WeightedHistoryBaseline({'decay': 0.9}),
            WeightedHistoryBaseline({'decay': 0.95}),
            PoissonBaseline(),
            DixonColesBaseline(),
            DixonColesBaseline({'decay': 0.98}),
            BayesianTeamBaseline(),
            BayesianTeamBaseline({'prior_weight': 10}),
        ]
    
    results = []
    for model in models:
        model.fit(train_df)
        metrics = model.evaluate(test_df)
        summary = model.get_summary()
        results.append({
            'model': summary['model'],
            'rmse': metrics['rmse'],
            'mae': metrics['mae'],
            'r2': metrics['r2'],
            'combined_rmse': metrics['combined_rmse'],
            **{k: v for k, v in summary.items() if k != 'model'}
        })

    return pd.DataFrame(results).sort_values('rmse')   # Best model first (lowest RMSE)


# =============================================================================
# MODELS 7–9: ADVANCED - Dixon-Coles, Bayesian, Ensemble
# =============================================================================
# These are more sophisticated models used in pro sports analytics and research.
# =============================================================================

# MODEL 7: Dixon-Coles - Industry standard for soccer/hockey prediction
# ---------------------------------------------------------------------
# Like Poisson but fits attack/defense by iterating: "Given current defense
# estimates, update attack. Given current attack, update defense." Repeat
# until numbers stop changing. Can weight recent games more (decay < 1).
# ---------------------------------------------------------------------

class DixonColesBaseline(BaselineModel):
    """
    Iterative fit: repeatedly update attack given defense, defense given attack,
    then home advantage. Stops when changes are tiny (converged).
    params: max_iter (default 50), tol (1e-6), decay (1.0=no decay), home_adv (1.15)
    """

    def __init__(self, params=None):
        super().__init__(params)
        self.max_iter = self.params.get('max_iter', 50)   # How many update cycles
        self.tol = self.params.get('tol', 1e-6)           # Stop when change < this
        self.decay = self.params.get('decay', 1.0)        # 1.0 = all games equal weight
        self.attack = {}
        self.defense = {}
        self.home_adv = self.params.get('home_adv', 1.15)
        self.league_avg = None

    def fit(self, games_df):
        """
        Loop: (1) Update each team's attack given current defense estimates.
        (2) Update each team's defense given current attack. (3) Update home
        advantage. (4) Rescale so attack means = 1. Repeat until numbers settle.
        """
        ht_col = get_column(games_df, 'home_team')
        at_col = get_column(games_df, 'away_team')
        hg_col = get_column(games_df, 'home_goals')
        ag_col = get_column(games_df, 'away_goals')

        hteams = games_df[ht_col].values
        ateams = games_df[at_col].values
        hgoals = games_df[hg_col].values.astype(float)
        agoals = games_df[ag_col].values.astype(float)
        n = len(games_df)

        weights = np.array([self.decay ** (n - 1 - i) for i in range(n)])   # Recent = higher weight

        teams = sorted(set(hteams) | set(ateams))
        atk = {t: 1.0 for t in teams}
        dfn = {t: 1.0 for t in teams}
        home_adv = self.home_adv
        self.league_avg = float(np.mean(np.concatenate([hgoals, agoals])))

        for iteration in range(self.max_iter):
            old_atk = dict(atk)
            old_dfn = dict(dfn)
            old_ha = home_adv

            # Attack update: (goals they scored) / (expected goals they should have scored)
            for t in teams:
                mask_h = hteams == t
                mask_a = ateams == t

                numerator = float(
                    np.sum(hgoals[mask_h] * weights[mask_h]) +
                    np.sum(agoals[mask_a] * weights[mask_a])
                )

                denom = 0.0
                if mask_h.any():
                    opp_def = np.array([dfn[a] for a in ateams[mask_h]])
                    denom += float(np.sum(home_adv * opp_def * weights[mask_h]))
                if mask_a.any():
                    opp_def = np.array([dfn[h] for h in hteams[mask_a]])
                    denom += float(np.sum(opp_def * weights[mask_a]))

                atk[t] = (numerator / denom) if denom > 0 else 1.0

            # Defense update: (goals they allowed) / (expected goals against them)
            for t in teams:
                mask_h = hteams == t
                mask_a = ateams == t

                numerator = float(
                    np.sum(agoals[mask_h] * weights[mask_h]) +
                    np.sum(hgoals[mask_a] * weights[mask_a])
                )

                denom = 0.0
                if mask_h.any():
                    opp_atk = np.array([atk[a] for a in ateams[mask_h]])
                    denom += float(np.sum(opp_atk * weights[mask_h]))  # they're away
                if mask_a.any():
                    opp_atk = np.array([atk[h] for h in hteams[mask_a]])
                    denom += float(np.sum(home_adv * opp_atk * weights[mask_a]))

                dfn[t] = (numerator / denom) if denom > 0 else 1.0

            # Home adv: (actual home goals) / (expected without home boost)
            num_ha = float(np.sum(hgoals * weights))
            den_ha = 0.0
            for i in range(n):
                den_ha += atk[hteams[i]] * dfn[ateams[i]] * weights[i]
            home_adv = (num_ha / den_ha) if den_ha > 0 else 1.15

            # Normalize attack so average = 1 (avoids scaling ambiguity)
            mean_atk = np.mean(list(atk.values()))
            if mean_atk > 0:
                for t in teams:
                    atk[t] /= mean_atk

            delta = max(
                max(abs(atk[t] - old_atk[t]) for t in teams),
                max(abs(dfn[t] - old_dfn[t]) for t in teams),
                abs(home_adv - old_ha),
            )
            if delta < self.tol:
                break

        self.attack = atk
        self.defense = dfn
        self.home_adv = home_adv
        self.n_games = n
        self.n_teams = len(teams)
        self.is_fitted = True
        return self

    def predict_goals(self, game):
        """
        λ_home = league_avg × home_attack × away_defense × home_adv
        λ_away = league_avg × away_attack × home_defense (away gets no boost)
        """
        home_team = get_value(game, 'home_team')
        away_team = get_value(game, 'away_team')

        h_atk = self.attack.get(home_team, 1.0)
        h_def = self.defense.get(home_team, 1.0)
        a_atk = self.attack.get(away_team, 1.0)
        a_def = self.defense.get(away_team, 1.0)

        home_goals = self.league_avg * h_atk * a_def * self.home_adv
        away_goals = self.league_avg * a_atk * h_def
        return home_goals, away_goals

    def get_summary(self):
        top_atk = sorted(self.attack.items(), key=lambda x: x[1], reverse=True)[:5]
        top_def = sorted(self.defense.items(), key=lambda x: x[1])[:5]
        return {
            'model': 'DixonColesBaseline',
            'decay': self.decay,
            'home_adv': round(self.home_adv, 4),
            'league_avg': round(self.league_avg, 3),
            'n_teams': self.n_teams,
            'n_games_trained': self.n_games,
            'iterations': self.max_iter,
            'top_attack': top_atk,
            'top_defense': top_def,
        }


# MODEL 8: Bayesian - Don't overtrust teams with few games
# -------------------------------------------------------
# If a team played 3 games and scored 10 goals, that's 3.33/game—but maybe
# they got lucky. "Shrinkage" pulls them toward the league average. Teams
# with many games stay close to their raw average; teams with few games get
# pulled more. prior_weight = how many "fake games at league avg" to add.
# -------------------------------------------------------

class BayesianTeamBaseline(BaselineModel):
    """
    estimate = (real_games × raw_avg + prior_weight × league_avg) / (real_games + prior_weight)
    prior_weight=5 means "pretend they played 5 extra games at league average."
    """

    def __init__(self, params=None):
        super().__init__(params)
        self.prior_weight = self.params.get('prior_weight', 5)
        self.attack = {}
        self.defense = {}
        self.league_home_avg = None
        self.league_away_avg = None

    def fit(self, games_df):
        """Apply shrinkage formula to each team's attack and defense."""
        ht_col = get_column(games_df, 'home_team')
        at_col = get_column(games_df, 'away_team')
        hg_col = get_column(games_df, 'home_goals')
        ag_col = get_column(games_df, 'away_goals')

        self.league_home_avg = float(games_df[hg_col].mean())
        self.league_away_avg = float(games_df[ag_col].mean())
        self.league_avg = (self.league_home_avg + self.league_away_avg) / 2

        goals_for = {}
        goals_against = {}
        gp = {}

        for _, game in games_df.iterrows():
            ht = game[ht_col]; at = game[at_col]
            hg = game[hg_col]; ag = game[ag_col]

            for team, gf, ga in [(ht, hg, ag), (at, ag, hg)]:
                goals_for.setdefault(team, 0.0)
                goals_against.setdefault(team, 0.0)
                gp.setdefault(team, 0)
                goals_for[team] += gf
                goals_against[team] += ga
                gp[team] += 1

        pw = self.prior_weight
        for team in gp:
            n = gp[team]
            raw_atk = goals_for[team] / n
            raw_def = goals_against[team] / n
            # Small n = more pull toward league_avg. Large n = stay close to raw.
            self.attack[team] = (raw_atk * n + self.league_avg * pw) / (n + pw)
            self.defense[team] = (raw_def * n + self.league_avg * pw) / (n + pw)

        self.n_games = len(games_df)
        self.n_teams = len(gp)
        self.is_fitted = True
        return self

    def predict_goals(self, game):
        """
        Expected goals using shrunk attack/defense, with home/away factors
        derived from league averages (home teams score more on average).
        """
        home_team = get_value(game, 'home_team')
        away_team = get_value(game, 'away_team')

        h_atk = self.attack.get(home_team, self.league_avg)
        h_def = self.defense.get(home_team, self.league_avg)
        a_atk = self.attack.get(away_team, self.league_avg)
        a_def = self.defense.get(away_team, self.league_avg)

        # Home/away factors from league averages (home teams score more)
        home_factor = self.league_home_avg / self.league_avg
        away_factor = self.league_away_avg / self.league_avg

        home_goals = (h_atk * (a_def / self.league_avg)) * home_factor
        away_goals = (a_atk * (h_def / self.league_avg)) * away_factor

        return home_goals, away_goals

    def get_summary(self):
        return {
            'model': 'BayesianTeamBaseline',
            'prior_weight': self.prior_weight,
            'league_avg': round(self.league_avg, 3),
            'n_teams': self.n_teams,
            'n_games_trained': self.n_games,
            'top_attack': sorted(self.attack.items(), key=lambda x: x[1], reverse=True)[:5],
            'top_defense': sorted(self.defense.items(), key=lambda x: x[1])[:5],
        }


# MODEL 9: Ensemble - Combine multiple models
# -------------------------------------------
# Ask several models for a prediction, then average (with weights). Better
# models get higher weight. " inverse_rmse" = weight ∝ 1/RMSE (lower error →
# higher weight). Like a panel of experts voting.
# -------------------------------------------

class EnsembleBaseline(BaselineModel):
    """
    params: models=list of baselines, weights=optional explicit weights,
    method='inverse_rmse' (auto-weight by performance) or 'uniform'.
    """

    def __init__(self, params=None):
        super().__init__(params)
        self.sub_models = self.params.get('models', [])
        self.weights = self.params.get('weights', None)
        self.method = self.params.get('method', 'inverse_rmse')

    def fit(self, games_df):
        """
        Train each sub-model on 75% of data. Use the other 25% to measure each
        model's RMSE. Weight = 1/RMSE (better models get higher weight). Then
        re-train all sub-models on 100% of data (weights stay fixed).
        """
        cal_split = int(len(games_df) * 0.75)
        train_part = games_df.iloc[:cal_split]
        cal_part = games_df.iloc[cal_split:]

        for m in self.sub_models:
            m.fit(train_part)

        if self.weights is None:
            if self.method == 'inverse_rmse':
                rmses = []   # Measure each model's error on calibration set
                for m in self.sub_models:
                    metrics = m.evaluate(cal_part)
                    rmses.append(metrics['combined_rmse'])
                inv = [1.0 / r for r in rmses]   # 1/RMSE: low error → big number (good)
                total = sum(inv)
                self.weights = [w / total for w in inv]   # Divide each by total so weights sum to 1.0
            else:
                self.weights = [1.0 / len(self.sub_models)] * len(self.sub_models)

        for m in self.sub_models:
            m.fit(games_df)   # Re-train on FULL data now that weights are set

        self.n_games = len(games_df)
        self.is_fitted = True
        return self

    def predict_goals(self, game):
        """Ask each sub-model, then blend: home_goals = sum(weight_i × pred_i)."""
        h_preds, a_preds = [], []
        for m in self.sub_models:
            h, a = m.predict_goals(game)
            h_preds.append(h)
            a_preds.append(a)

        # sum(w*h for w,h in zip(...)) = weight1*pred1 + weight2*pred2 + ... (weighted average)
        home_goals = sum(w * h for w, h in zip(self.weights, h_preds))
        away_goals = sum(w * a for w, a in zip(self.weights, a_preds))
        return home_goals, away_goals

    def get_summary(self):
        sub_names = []
        for m in self.sub_models:
            try:
                sub_names.append(m.get_summary()['model'])
            except Exception:
                sub_names.append(type(m).__name__)
        return {
            'model': 'EnsembleBaseline',
            'sub_models': sub_names,
            'weights': [round(w, 4) for w in (self.weights or [])],
            'method': self.method,
            'n_games_trained': self.n_games,
        }


# "Baseline" without a prefix means TeamMeanBaseline (the most commonly used one)
Baseline = TeamMeanBaseline
