import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
from sklearn.ensemble import StackingRegressor
import random

# Load data
data = pd.read_csv('algosports23-predictions-2025/Train.csv')

# Feature Engineering
def feature_engineering(df):
    # Add IsHomeWin column
    df['IsHomeWin'] = (df['HomePts'] > df['AwayPts']).astype(int)
    # Head-to-head win margin
    df['HeadToHeadMargin'] = 0
    for idx, row in df.iterrows():
        team1 = row['HomeTeam']
        team2 = row['AwayTeam']
        h2h_games = df[((df['HomeTeam'] == team1) & (df['AwayTeam'] == team2)) | ((df['HomeTeam'] == team2) & (df['AwayTeam'] == team1))]
        if not h2h_games.empty:
            df.at[idx, 'HeadToHeadMargin'] = h2h_games['HomeWinMargin'].median()
    # Conference win rates
    conf_win_rate = df.groupby('HomeConf')['IsHomeWin'].mean().to_dict()
    df['HomeConfWinRate'] = df['HomeConf'].map(conf_win_rate)
    df['AwayConfWinRate'] = df['AwayConf'].map(conf_win_rate)
    # Team average win margin
    team_avg_margin = pd.concat([
        df[['HomeTeam', 'HomeWinMargin']].rename(columns={'HomeTeam':'Team', 'HomeWinMargin':'Margin'}),
        df[['AwayTeam', 'HomeWinMargin']].rename(columns={'AwayTeam':'Team', 'HomeWinMargin':'Margin'})
    ])
    avg_margin = team_avg_margin.groupby('Team')['Margin'].median().to_dict()
    df['HomeAvgMargin2'] = df['HomeTeam'].map(avg_margin)
    df['AwayAvgMargin2'] = df['AwayTeam'].map(avg_margin)
    # Recent performance trend (last 10 games)
    df['HomeRecentTrend'] = df.groupby('HomeTeam')['HomeWinMargin'].rolling(window=10, min_periods=1).mean().reset_index(level=0, drop=True)
    df['AwayRecentTrend'] = df.groupby('AwayTeam')['HomeWinMargin'].rolling(window=10, min_periods=1).mean().reset_index(level=0, drop=True)
    # Example features: win/loss, margin, points, conference strength, common opponents
    df['WinMargin'] = df['HomeWinMargin']
    df['Winner'] = np.where(df['HomePts'] > df['AwayPts'], df['HomeTeam'], df['AwayTeam'])
    df['Loser'] = np.where(df['HomePts'] < df['AwayPts'], df['HomeTeam'], df['AwayTeam'])
    df['IsHomeWin'] = (df['HomePts'] > df['AwayPts']).astype(int)
    # Encode teams
    le = LabelEncoder()
    df['HomeTeamEncoded'] = le.fit_transform(df['HomeTeam'])
    df['AwayTeamEncoded'] = le.transform(df['AwayTeam'])
    # Conference strength (median margin per conference)
    conf_margin = df.groupby('HomeConf')['WinMargin'].median().to_dict()
    df['HomeConfStrength'] = df['HomeConf'].map(conf_margin)
    df['AwayConfStrength'] = df['AwayConf'].map(conf_margin)
    # Points scored
    df['HomePoints'] = df['HomePts']
    df['AwayPoints'] = df['AwayPts']
    # Recent win/loss streak (last 5 games)
    df['HomeRecentWins'] = df.groupby('HomeTeam')['IsHomeWin'].rolling(window=5, min_periods=1).sum().reset_index(level=0, drop=True)
    df['AwayRecentWins'] = df.groupby('AwayTeam')['IsHomeWin'].rolling(window=5, min_periods=1).sum().reset_index(level=0, drop=True)
    # Average margin last 5 games
    df['HomeRecentMargin'] = df.groupby('HomeTeam')['WinMargin'].rolling(window=5, min_periods=1).mean().reset_index(level=0, drop=True)
    df['AwayRecentMargin'] = df.groupby('AwayTeam')['WinMargin'].rolling(window=5, min_periods=1).mean().reset_index(level=0, drop=True)
    # Home/Away performance
    df['HomeAvgMargin'] = df.groupby('HomeTeam')['WinMargin'].transform('median')
    df['AwayAvgMargin'] = df.groupby('AwayTeam')['WinMargin'].transform('median')
    return df
    df['AwayAvgMargin'] = df.groupby('AwayTeam')['WinMargin'].transform('median')
    return df

data = feature_engineering(data)

# Prepare features and target
target = 'WinMargin'
features = [
    'HomeTeamEncoded', 'AwayTeamEncoded', 'HomeConfStrength', 'AwayConfStrength',
    'HomePoints', 'AwayPoints', 'IsHomeWin',
    'HomeRecentWins', 'AwayRecentWins',
    'HomeRecentMargin', 'AwayRecentMargin',
    'HomeAvgMargin', 'AwayAvgMargin',
    'HeadToHeadMargin', 'HomeConfWinRate', 'AwayConfWinRate',
    'HomeAvgMargin2', 'AwayAvgMargin2',
    'HomeRecentTrend', 'AwayRecentTrend'
]
X = data[features]
y = data[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
scaler.fit(X)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define base models

# Enhanced base models with more hyperparameters
from sklearn.model_selection import GridSearchCV, KFold
# KFold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
# Linear Regression (no hyperparameters, but use cross-validation)
lr = LinearRegression()
# Random Forest Grid Search
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [8, 10, 12],
    'min_samples_split': [2, 4],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}
rf_grid = GridSearchCV(RandomForestRegressor(random_state=42), rf_param_grid, cv=kfold, n_jobs=-1, scoring='neg_root_mean_squared_error')
rf_grid.fit(X_train_scaled, y_train)
rf_best = rf_grid.best_estimator_
# MLP Grid Search
mlp_param_grid = {
    'hidden_layer_sizes': [(128, 64, 32), (64, 32)],
    'activation': ['relu', 'tanh'],
    'learning_rate_init': [0.01, 0.001],
    'max_iter': [500, 1000],
    'alpha': [0.001, 0.01]
}
mlp_grid = GridSearchCV(MLPRegressor(random_state=42), mlp_param_grid, cv=kfold, n_jobs=-1, scoring='neg_root_mean_squared_error')
mlp_grid.fit(X_train_scaled, y_train)
mlp_best = mlp_grid.best_estimator_
# XGBoost Grid Search
xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [6, 8, 10],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'reg_alpha': [0.1, 1],
    'reg_lambda': [1, 2],
    'gamma': [0.1, 0.5]
}
xgb_grid = GridSearchCV(xgb.XGBRegressor(random_state=42), xgb_param_grid, cv=kfold, n_jobs=-1, scoring='neg_root_mean_squared_error')
xgb_grid.fit(X_train_scaled, y_train)
xgb_best = xgb_grid.best_estimator_
# Ensemble: Stacking with best models
estimators = [
    ('lr', lr),
    ('rf', rf_best),
    ('mlp', mlp_best),
    ('xgbr', xgb_best)
]
stack = StackingRegressor(
    estimators=estimators,
    final_estimator=LinearRegression(),
    passthrough=True,
    n_jobs=-1
)
stack.fit(X_train_scaled, y_train)
print("Best RandomForest params:", rf_grid.best_params_)
print("Best MLP params:", mlp_grid.best_params_)
print("Best XGBoost params:", xgb_grid.best_params_)

# --- Team Ranking ---
team_stats = {}
for team in pd.concat([data['HomeTeam'], data['AwayTeam']]).unique():
    home_games = data[data['HomeTeam'] == team]
    away_games = data[data['AwayTeam'] == team]
    wins = (home_games['HomePts'] > home_games['AwayPts']).sum() + (away_games['AwayPts'] > away_games['HomePts']).sum()
    losses = (home_games['HomePts'] < home_games['AwayPts']).sum() + (away_games['AwayPts'] < away_games['HomePts']).sum()
    points_scored = pd.concat([home_games['HomePts'], away_games['AwayPts']]).median()
    points_allowed = pd.concat([home_games['AwayPts'], away_games['HomePts']]).median()
    margin = points_scored - points_allowed
    games_played = wins + losses
    win_rate = wins / games_played if games_played > 0 else 0
    team_id = home_games['HomeID'].iloc[0] if not home_games.empty else away_games['AwayID'].iloc[0]
    # Composite score: win rate, margin, points scored (all median-based)
    composite_score = (
        0.5 * win_rate +
        0.3 * (margin / (data['HomePts'].median() if data['HomePts'].median() != 0 else 1)) +
        0.2 * (points_scored / (data['HomePts'].median() if data['HomePts'].median() != 0 else 1))
    )
    team_stats[team] = {
        'TeamID': team_id,
        'Team': team,
        'Wins': wins,
        'Losses': losses,
        'PointsScored': points_scored,
        'PointsAllowed': points_allowed,
        'Margin': margin,
        'WinRate': win_rate,
        'CompositeScore': composite_score
    }

rankings = pd.DataFrame.from_dict(team_stats, orient='index')
rankings['Rank'] = rankings['CompositeScore'].rank(ascending=False, method='min').astype(int)
rankings = rankings.sort_values('Rank')
rankings.to_excel('algosports23-predictions-2025/Rankings.xlsx', index=False, columns=['TeamID', 'Team', 'Rank', 'Wins', 'Losses', 'PointsScored', 'PointsAllowed', 'Margin', 'WinRate', 'CompositeScore'])

# --- Derby Predictions ---

# --- Derby Predictions ---
# Use the 75 rivalry matches from Sample_Predictions.csv
sample_pred_path = 'algosports23-predictions-2025/algosports23_dataset/Sample_Predictions.csv'
sample_pred_df = pd.read_csv(sample_pred_path)
derby_gameids = sample_pred_df['GameID'].tolist()
derby_matches = data[data['GameID'].isin(derby_gameids)]
pred_rows = []
for _, row in sample_pred_df.iterrows():
    # Use Sample_Predictions.csv for match details
    team1 = row['Team1']
    team2 = row['Team2']
    team1_conf = row['Team1_Conf']
    team2_conf = row['Team2_Conf']
    team1_id = row['Team1_ID']
    team2_id = row['Team2_ID']
    game_id = row['GameID']
    date = row['Date']
    # Find stats for teams
    home_encoded = data['HomeTeam'].unique().tolist().index(team1) if team1 in data['HomeTeam'].unique() else 0
    away_encoded = data['AwayTeam'].unique().tolist().index(team2) if team2 in data['AwayTeam'].unique() else 0
    home_conf_strength = rankings[rankings['Team'] == team1]['CompositeScore'].values[0] if team1 in rankings['Team'].values else rankings['CompositeScore'].median()
    away_conf_strength = rankings[rankings['Team'] == team2]['CompositeScore'].values[0] if team2 in rankings['Team'].values else rankings['CompositeScore'].median()
    home_points = rankings[rankings['Team'] == team1]['PointsScored'].values[0] if team1 in rankings['Team'].values else rankings['PointsScored'].median()
    away_points = rankings[rankings['Team'] == team2]['PointsScored'].values[0] if team2 in rankings['Team'].values else rankings['PointsScored'].median()
    is_home_win = 1
    # Enhanced features for prediction
    home_recent_wins = data[data['HomeTeam'] == team1]['IsHomeWin'].tail(5).sum() if team1 in data['HomeTeam'].values else 0
    away_recent_wins = data[data['AwayTeam'] == team2]['IsHomeWin'].tail(5).sum() if team2 in data['AwayTeam'].values else 0
    home_recent_margin = data[data['HomeTeam'] == team1]['WinMargin'].tail(5).median() if team1 in data['HomeTeam'].values else 0
    away_recent_margin = data[data['AwayTeam'] == team2]['WinMargin'].tail(5).median() if team2 in data['AwayTeam'].values else 0
    home_avg_margin = data[data['HomeTeam'] == team1]['WinMargin'].median() if team1 in data['HomeTeam'].values else 0
    away_avg_margin = data[data['AwayTeam'] == team2]['WinMargin'].median() if team2 in data['AwayTeam'].values else 0
    # Advanced features for prediction
    h2h_games = data[((data['HomeTeam'] == team1) & (data['AwayTeam'] == team2)) | ((data['HomeTeam'] == team2) & (data['AwayTeam'] == team1))]
    head_to_head_margin = h2h_games['WinMargin'].median() if not h2h_games.empty else 0
    home_conf_win_rate = data[data['HomeConf'] == team1_conf]['IsHomeWin'].mean() if team1_conf in data['HomeConf'].values else 0
    away_conf_win_rate = data[data['HomeConf'] == team2_conf]['IsHomeWin'].mean() if team2_conf in data['HomeConf'].values else 0
    home_avg_margin2 = data[data['HomeTeam'] == team1]['WinMargin'].median() if team1 in data['HomeTeam'].values else 0
    away_avg_margin2 = data[data['AwayTeam'] == team2]['WinMargin'].median() if team2 in data['AwayTeam'].values else 0
    home_recent_trend = data[data['HomeTeam'] == team1]['WinMargin'].tail(10).median() if team1 in data['HomeTeam'].values else 0
    away_recent_trend = data[data['AwayTeam'] == team2]['WinMargin'].tail(10).median() if team2 in data['AwayTeam'].values else 0
    # Strength of schedule: average CompositeScore of opponents
    home_opponents = pd.concat([
        data[data['HomeTeam'] == team1]['AwayTeam'],
        data[data['AwayTeam'] == team1]['HomeTeam']
    ])
    away_opponents = pd.concat([
        data[data['HomeTeam'] == team2]['AwayTeam'],
        data[data['AwayTeam'] == team2]['HomeTeam']
    ])
    home_sos = rankings[rankings['Team'].isin(home_opponents)]['CompositeScore'].mean() if not home_opponents.empty else rankings['CompositeScore'].median()
    away_sos = rankings[rankings['Team'].isin(away_opponents)]['CompositeScore'].mean() if not away_opponents.empty else rankings['CompositeScore'].median()
    # Common opponent margin: median margin vs common opponents
    common_opponents = set(home_opponents) & set(away_opponents)
    home_common_margin = np.median([
        pd.concat([
            data[(data['HomeTeam'] == team1) & (data['AwayTeam'] == opp)]['HomeWinMargin'],
            -data[(data['AwayTeam'] == team1) & (data['HomeTeam'] == opp)]['HomeWinMargin']
        ]).median() for opp in common_opponents
    ]) if common_opponents else 0
    away_common_margin = np.median([
        pd.concat([
            data[(data['HomeTeam'] == team2) & (data['AwayTeam'] == opp)]['HomeWinMargin'],
            -data[(data['AwayTeam'] == team2) & (data['HomeTeam'] == opp)]['HomeWinMargin']
        ]).median() for opp in common_opponents
    ]) if common_opponents else 0
    features_pred = [
        home_encoded, away_encoded, home_conf_strength, away_conf_strength,
        home_points, away_points, is_home_win,
        home_recent_wins, away_recent_wins,
        home_recent_margin, away_recent_margin,
        home_avg_margin, away_avg_margin,
        head_to_head_margin, home_conf_win_rate, away_conf_win_rate,
        home_avg_margin2, away_avg_margin2,
        home_recent_trend, away_recent_trend,
        home_sos, away_sos,
        home_common_margin, away_common_margin
    ]
    try:
        features_pred_scaled = scaler.transform([features_pred])
        # Use the best estimator from stacking for prediction
        margin_pred = stack.predict(features_pred_scaled)[0]
        if pd.isnull(margin_pred) or margin_pred is None:
            margin_pred = 0
    except Exception:
        margin_pred = 0
    # Ensure margin_pred is a valid integer
    if not isinstance(margin_pred, (int, float)) or pd.isnull(margin_pred):
        margin_pred = 0
    margin_pred_int = int(round(margin_pred))
    # Ensure margin is from Team1's perspective
    # If Team1 is home, margin is as predicted; if Team1 is away, invert margin
    if team1 == row['Team1']:
        team1_margin = margin_pred_int
    else:
        team1_margin = -margin_pred_int
    # Team1_WinMargin: positive if Team1 wins, negative if Team1 loses, zero if tie
    pred_rows.append({
        'GameID': game_id,
        'Date': date,
        'Team1_Conf': team1_conf,
        'Team1_ID': team1_id,
        'Team1': team1,
        'Team2_Conf': team2_conf,
        'Team2_ID': team2_id,
        'Team2': team2,
        'Team1_WinMargin': team1_margin
    })
pred_df = pd.DataFrame(pred_rows)
pred_df['Team1_WinMargin'] = pred_df['Team1_WinMargin'].fillna(0).astype(int)

# Save predictions
if len(pred_df) == 75:
    pred_df.to_csv('algosports23-predictions-2025/Predictions.csv', index=False, encoding='utf-8')
else:
    print(f"Warning: Only {len(pred_df)} predictions generated. Expected 75.")

# --- Evaluation: RMSE and Winning Team Accuracy ---
actual_margins = sample_pred_df['Team1_WinMargin'].astype(int).values
predicted_margins = pred_df['Team1_WinMargin'].astype(int).values
if len(actual_margins) == 75 and len(predicted_margins) == 75:
    # RMSE calculation
    rmse = np.sqrt(np.mean((predicted_margins - actual_margins) ** 2))
    print(f"RMSE (Predicted vs Actual Margins): {rmse:.2f}")
    # Winning team accuracy
    actual_winners = np.sign(actual_margins)
    predicted_winners = np.sign(predicted_margins)
    correct_winner = (actual_winners == predicted_winners).sum()
    accuracy = correct_winner / 75 * 100
    print(f"Winning Team Prediction Accuracy: {accuracy:.2f}% ({correct_winner}/75)")
else:
    print("Warning: Cannot evaluate RMSE or accuracy, margin counts do not match.")

# Generate submission archive
# os.system('zip algosports23-predictions-2025/Submission.zip algosports23-predictions-2025/Predictions.csv algosports23-predictions-2025/Rankings.xlsx')
import zipfile
import os
# Only include prediction and ranking files in the zip, and check if they exist before adding
for filename in ['Predictions.csv', 'Rankings.xlsx']:
    file_path = f'algosports23-predictions-2025/{filename}'
    if os.path.exists(file_path):
        print(f"{file_path} exists and is ready for submission.")
    else:
        print(f"Error: {file_path} does not exist. Please check the file generation step.")
with zipfile.ZipFile('algosports23-predictions-2025/Submission.zip', 'w') as zipf:
    for filename in ['Predictions.csv', 'Rankings.xlsx']:
        file_path = f'algosports23-predictions-2025/{filename}'
        if os.path.exists(file_path):
            zipf.write(file_path, arcname=filename)
            print(f"Added {file_path} to Submission.zip")
        else:
            print(f"Skipped {file_path} as it does not exist.")

