# TODO - Wharton HS Hockey Competition

**Deadline:** Phase 1 due March 2, 2026 (9:00 AM EST)

---

## Phase 0: Data Collection (Before Data Release)

**CONFIRMED BUILT-IN FEATURES:**
✅ Travel time, travel distance (already in dataset!)
✅ Rest time / days since last game (already in dataset!)
✅ Injuries count (already in dataset!)
✅ Clinch status, elimination status (already in dataset!)
✅ Division tiers (D1/D2/D3) (already in dataset!)

**STILL NEED TO COLLECT:**
- [ ] Collect special teams data (power play/penalty kill percentages)
  - Scrape from NHL stats API or Hockey Reference, calculate rolling 10-game averages
- [ ] Find goalie starter announcements (if available pre-game)
  - Check Daily Faceoff, Rotoworld for confirmed starters, update dataset day-of-game
- [ ] Research coaching changes, player trades during season
  - Create events table [date, team, event_type, description], join to games by date
- [ ] Identify division rivalries and historical matchups
  - Flag division games, calculate head-to-head records, add "rivalry intensity" score
- [ ] Document arena-specific factors (ice quality, altitude)
  - Create arena metadata table [arena_id, city, altitude], join to home team games

---

## ✅ COMPLETED: Explainability & Debugging

### SHAP Integration
- [x] Created ModelExplainer class with SHAP support
- [x] CLI command: explain-model (generates SHAP reports)
- [x] CLI command: explain-prediction (single prediction analysis)
- [x] Auto-generates summary plots, importance plots, dependence plots
- [x] Interactive HTML reports with visualizations
- [x] Support for XGBoost, LightGBM, Random Forest, Linear models

### Debugging Tools
- [x] CLI command: debug-errors (error pattern analysis)
- [x] CLI command: debug-features (data quality checks)
- [x] Missing value detection
- [x] Outlier detection (configurable sigma threshold)
- [x] Constant feature detection
- [x] High correlation detection
- [x] Feature quality scoring
- [x] Systematic bias detection
- [x] Error distribution analysis
- [x] Worst prediction identification

### Documentation
- [x] Complete explainability guide (docs/guides/EXPLAINABILITY_DEBUG_GUIDE.md)
- [x] Example scripts (scripts/explainability_examples.rb)
- [x] RSpec tests (spec/model_explainer_spec.rb)
- [x] Updated README with new features
- [x] Updated requirements.txt with SHAP dependencies

---

## Phase 1: Ruby - Data Preprocessing & Hyperparameters

### 1.1 Initial Data Processing
- [ ] Download competition dataset
- [ ] Run diagnostics: `ruby cli.rb diagnose data/hockey_challenge.csv`
- [ ] Run preprocessing: `ruby cli.rb competitive-pipeline data/hockey_challenge.csv -o data/processed`
- [ ] Verify output files exist in `data/processed/`

### 1.2 Additional Feature Engineering (Beyond Built-Ins)
- [ ] **USE BUILT-IN**: travel_distance, travel_time, rest_time, injuries
  - These are already in the dataset! Just need to normalize/bin them
- [ ] Fatigue interactions: `travel_distance * (1 / rest_time)` (tired from travel)
  - Create fatigue_index = travel_distance / max(rest_time, 1), high values = exhausted
- [ ] Rest bins: back-to-back (rest_time=0-1), normal (2-3), extended (4+)
  - Create categorical bins, one-hot encode for linear models
- [ ] Injury severity: interactions with `injuries * is_away` (worse when traveling)
  - Binary flags: has_injuries, multiple_injuries (>2), key_injury_home_advantage
- [ ] Clinch motivation: `clinch_status` interaction with playoff proximity
  - Flag teams that clinched early (may rest stars), vs teams fighting for position
- [ ] Elimination effect: teams eliminated early may tank/rest players
  - Binary flag for eliminated teams, interaction with game_number (late season tanking)
- [ ] Presidents' Trophy curse: teams with `award` may underperform in playoffs
  - Historical stat: track post-season performance vs regular season for award winners
- [ ] External factors: Weather data (if home games, temperature affects ice quality)
  - Scrape weather APIs for game day temps at arena locations, create binary flag for extreme cold (<0°F)
- [ ] External factors: Playoff implications (motivation level)
  - Calculate games behind division leader, assign higher weight to games in March/April
- [ ] Team dynamics: Line chemistry (top line vs depth scoring)
  - Use rolling average of top-6 forward points vs bottom-6 as ratio
- [ ] Temporal: Time of season (early = inconsistent, late = playoff push)
  - Game number / total games as decimal (0.0 to 1.0), create quadratic feature
- [ ] Temporal: Month effects (December fatigue, March playoff race)
  - One-hot encode months, interaction with win% to capture seasonal performance patterns
- [ ] Situational: Revenge games (playing team that recently beat you)
  - Track last 5 meetings, flag if opponent won last matchup by 3+ goals
- [ ] Situational: Division rivals (higher intensity)
  - Binary flag for same division, weight by standings proximity
- [ ] Situational: Conference matchups (playoff seeding impact)
  - Binary flag for conference games after game 60, interaction with playoff position
- [ ] Advanced stats: PDO (shooting % + save %, luck indicator)
  - Calculate (GF/shots) + (1 - GA/shots_against), rolling 10-game window, flag if >1.02 or <0.98
- [ ] Advanced stats: Fenwick/Corsi (shot attempt differential)
  - Use shot attempts for/against ratio, exponentially weighted by recency
- [ ] Advanced stats: Expected goals (xG) vs actual goals (over/underperformance)
  - If xG data available, calculate xG - actual_goals, positive = unlucky, negative = overperforming
- [ ] Advanced stats: Special teams efficiency (PP%, PK%)
  - PP goals / PP opportunities, PK saves / PK situations, rolling 15-game averages
- [ ] Advanced stats: Zone start percentage (offensive vs defensive zone)
  - If available, (O-zone starts) / (O-zone + D-zone starts), higher = more offensive deployment
- [ ] Momentum: Winning/losing streaks (psychological factor)
  - Current consecutive W or L count, cap at 10, create quadratic feature for momentum effect
- [ ] Momentum: Goals in last period (late-game collapse indicator)
  - Calculate avg goals allowed in 3rd period over last 10 games, flag if >1.5
- [ ] Momentum: One-goal game record (clutch performance)
  - (Wins in 1-goal games) / (total 1-goal games) over season, rolling window
- [ ] Goalie performance: Recent save percentage
  - Last 5 starts save%, weighted by games played, fill missing with team average
- [ ] Goalie performance: Goals against average (GAA)
  - Goals allowed / games, last 10 starts, normalize by league average
- [ ] Goalie performance: Quality start percentage
  - Define quality start as save% >.900, calculate rate over last 10 starts
- [ ] Player injuries: Key player out (star forward, #1 defenseman, starting goalie)
  - Binary flags for top-3 scorers, #1 goalie, top-2 defensemen absence from lineup
- [ ] Coaching: New coach effect (system changes)
  - Flag first 10 games with new coach, track if trending up/down in those games
- [ ] Arena effects: Home ice advantage percentage
  - (Home wins) / (home games), compare to league average (typically ~55%)
- [ ] Arena effects: Altitude (if applicable, affects stamina)
  - Binary flag for Denver (5280ft), interaction with opponent's elevation (fatigue for low-elevation teams)

### 1.3 Generate Hyperparameters
- [ ] Model 2: `ruby cli.rb hyperparam-grid config/hyperparams/model2_linear_regression.yaml`
- [ ] Model 3: `ruby cli.rb hyperparam-bayesian config/hyperparams/model3_elo.yaml --iterations 30`
- [ ] Model 4a: `ruby cli.rb hyperparam-genetic config/hyperparams/model4_xgboost.yaml --population 50 --generations 20`
- [ ] Model 4b: `ruby cli.rb hyperparam-grid config/hyperparams/model4_random_forest.yaml`
- [ ] Model 5: `ruby cli.rb hyperparam-grid config/hyperparams/model5_ensemble.yaml`

### 1.4 Generate Tracking Dashboards
- [ ] Generate dashboards: `ruby cli.rb report-all`
- [ ] Commit and push all CSVs to GitHub

---

## P2.1 Advanced Feature Engineeringn

### Advanced Feature Engineering (Pre-Training)
- [ ] Create interaction features (home advantage × rest days, goalie save% × opponent offense)
  - Multiply normalized features, test top 20 combinations by correlation with target
- [ ] Implement feature selection (remove multicollinear features, VIF > 10)
  - Use `from statsmodels.stats.outliers_influence import variance_inflation_factor`, drop features with VIF > 10 iteratively
- [ ] Create lag features (opponent's last 3 game scores, team's scoring trend)
  - Use `df.groupby('team')['goals'].shift(1)` for lag-1, repeat for lag-2, lag-3
- [ ] Engineer rolling statistics (5-game, 10-game windows for all key metrics)
  - `df.groupby('team')['metric'].rolling(window=10, min_periods=3).mean()`, handle start-of-season with min_periods
- [ ] Build opponent strength metrics (weighted by recency)
  - Calculate opponent win%, weight recent games 2x using exponential decay: `weights = np.exp(-0.1 * games_ago)`
- [ ] Create contextual features (score differential × time remaining)
  - Multiply current goal diff by remaining game time fraction, captures urgency factor
- [ ] Implement target encoding for categorical variables (teams, divisions)
  - Use sklearn TargetEncoder with 5-fold CV to prevent leakage, smooth with prior mean

### 2.2 Model Training
- [ ] Model 1: Train baseline (mean/median predictions)
  - `y_pred = np.full(len(X_test), y_train.mean())`, calculate RMSE as benchmark
- [ ] Model 2: Train all Linear Regression configs, record metrics to CSV
  - Loop through CSV rows, `LinearRegression(**params).fit(X_train, y_train)`, write metrics back
- [ ] Model 2 Advanced: Add Ridge/Lasso/ElasticNet regularization
  - Test alpha values [0.01, 0.1, 1, 10], use GridSearchCV with 5-fold time series split
- [ ] Model 3: Train all ELO configs, record metrics to CSV
  - Initialize all teams at 1500, update after each game: `new_elo = old_elo + k * (actual - expected)`
- [ ] Model 3 Advanced: Implement K-factor decay over season
  - Start k=32, decay to k=16 by season end: `k = 32 * (1 - game_number/total_games) + 16`
- [ ] Model 4a: Train all XGBoost configs, record metrics to CSV
  - `xgb.XGBRegressor(**params, early_stopping_rounds=50).fit(X_train, y_train, eval_set=[(X_val, y_val)])`
- [ ] Model 4a Advanced: Implement early stopping, custom objective function
  - Custom objective for MAE: `def mae_obj(y_pred, y_true): grad = np.sign(y_pred - y_true); hess = np.ones_like(y_pred)`
- [ ] Model 4b: Train all Random Forest configs, record metrics to CSV
  - `RandomForestRegressor(**params, n_jobs=-1).fit(X_train, y_train)`, save feature_importances_
- [ ] Model 4b Advanced: Test ExtraTrees for comparison
  - `ExtraTreesRegressor(n_estimators=500, max_features='sqrt')`, typically less overfitting than RF
- [ ] Model 4c (Optional): LightGBM (faster than XGBoost, similar accuracy)
  - `lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, num_leaves=31, early_stopping_rounds=50)`
- [ ] Model 4d (Optional): CatBoost (handles categorical features natively)
  - `CatBoostRegressor(iterations=1000, cat_features=['team', 'opponent', 'division'])`, no encoding needed
- [ ] Model 5: Train ensemble (stacking, weighted voting, rank averaging)
  - Stacking: `StackingRegressor(estimators=[rf, xgb, lgb], final_estimator=Ridge())`
- [ ] Model 5 Advanced: Dynamic ensemble (weight models by recent performance)
  - Calculate RMSE on last 20 games per model, weights = 1/RMSE, renormalize to sum=1
- [ ] Model 6 (Optional): Neural network (LSTM for sequential game data)
  - `tf.keras.Sequential([LSTM(64, input_shape=(seq_len, features)), Dense(32, 'relu'), Dense(1)])`, use last 10 games as sequence
- [ ] Model 7 (Optional): Gradient Boosting with quantile regression (prediction intervals)
  - `GradientBoostingRegressor(loss='quantile', alpha=0.1)` for 10th percentile, alpha=0.9 for 90th

### 2.3 Validation & Testing
- [ ] Implement time series cross-validation (expanding window, no data leakage)
  - `TimeSeriesSplit(n_splits=5)`, train on games 1-N, test on N+1 to N+M, never shuffle
- [ ] Implement walk-forward validation (retrain after each game week)
  - Retrain model every 7 games, test on next 7, simulates real deployment scenario
- [ ] Generate learning curves
  - Train on [10%, 20%, ..., 100%] of data, plot train/val RMSE to detect underfitting/overfitting
- [ ] Analyze residuals (check for patterns by team, home/away, season phase)
  - `residuals = y_true - y_pred`, plot residuals vs features, test for heteroscedasticity
- [ ] Check for overfitting (train vs validation gap < 0.1 RMSE)
  - If train RMSE = 1.2 and val RMSE = 2.5, increase regularization or reduce features
- [ ] Perform sensitivity analysis (which features matter most)
  - Permutation importance: shuffle each feature, measure RMSE increase, rank by impact
- [ ] Test on holdout set (final 10% of season, never touched during training)
  - Set aside from start, only evaluate once at very end to get unbiased performance estimate

### 2.4 Uncertainty Quantification
- [ ] Implement prediction intervals (80%, 90%, 95% confidence)
- [ ] Quantile regression (predict 10th, 50th, 90th percentiles)
- [ ] Bootstrap predictions (1000 iterations for confidence bounds)
- [ ] Conformal prediction (distribution-free uncertainty)
- [ ] Ensemble disagreement as uncertainty proxy (high disagreement = high uncertainty)
  - Format: `submission.csv` with columns [game_id, predicted_score_home, predicted_score_away]
- [ ] Generate visualizations (feature importance, scatter plots, residuals)
  - `plt.scatter(y_true, y_pred)`, add diagonal line, calculate R², annotate with RMSE
- [ ] Create SHAP plots (explain individual predictions)
  - `shap.TreeExplainer(model)`, `shap.summary_plot(shap_values, X_test)`, shows feature contributions
- [ Calculate std dev of predictions across all models, high std = uncertain prediction

---

## Phase 3: Post-Training Analysis

### 3.1 Ruby Analysis
- [ ] Pull updated CSVs: `git pull`
- [ ] Generate final reports: `ruby cli.rb report-all --open`
- [ ] Find best params for each model: `ruby cli.rb best-params <file> --metric rmse`
- [ ] Optimize ensemble: `ruby cli.rb ensemble-optimize predictions/ --actuals test.csv`
- [ ] Analyze diversity: `ruby cli.rb diversity-analysis predictions/ test.csv`

### 3.2 Python Visualizations & Submission
- [ ] Create final predictions file for submission
  - Format: `submission.csv` with columns [game_id, predicted_score_home, predicted_score_away]
- [ ] Generate visualizations (feature importance, scatter plots, residuals)
  - `plt.scatter(y_true, y_pred)`, add diagonal line, calculate R², annotate with RMSE
- [ ] Create SHAP plots (explain individual predictions)
  - `shap.TreeExplainer(model)`, `shap.summary_plot(shap_values, X_test)`, shows feature contributions
- [ ] Generate partial dependence plots (feature effect visualization)
  - `from sklearn.inspection import PartialDependenceDisplay`, plot for top 5 features
- [ ] Create prediction interval plots (show uncertainty)
  - Plot actual vs predicted with error bars: `plt.errorbar(x, y_pred, yerr=[lower_bound, upper_bound])`
- [ ] Verify submission format
  - Check no missing values, scores >= 0, integer values (if required), correct column names
- [ ] Run final sanity checks (no negative scores, reasonable ranges)
  - Assert `all(preds >= 0)`, `all(preds <= 15)`, mean predicted score ~3 goals (NHL average)

---

## Phase 4: Presentation (March 16-23)

- [ ] Create slide deck
- [ ] Document methodology
- [ ] Prepare results summary
- [ ] Clean and comment code

---

## Success Criteria

- [ ] RMSE < 2.0 (target: 1.7)
- [ ] Ensemble beats best individual by 15%+
- [ ] Prediction intervals capture 90% of actual results
- [ ] Model performs well on all game types (division, conference, regular season phases)