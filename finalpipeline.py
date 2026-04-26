# Create the final pipeline Python file for the user
import os
import zipfile
import pandas as pd
import numpy as np
import zipfile
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge

# =========================
# LOAD DATA
# =========================
train_df = pd.read_csv("algosports23-predictions-2025/Train.csv")
pred_df = pd.read_csv("algosports23-predictions-2025/Predictions.csv")
sample_pred = pd.read_csv("algosports23-predictions-2025/Sample_Predictions.csv")
sample_rank = pd.read_excel("algosports23-predictions-2025/Sample_Rankings.xlsx")

print("Data loaded successfully")

# =========================
# PREPROCESS
# =========================
train_df["margin"] = train_df["HomePts"] - train_df["AwayPts"]

# =========================
# ELO MODEL
# =========================
def compute_elo(df, k=20):
    teams = pd.unique(df[["HomeTeam", "AwayTeam"]].values.ravel())
    elo = {team: 2000 for team in teams}
    
    for _, row in df.iterrows():
        A, B = row["HomeTeam"], row["AwayTeam"]
        Ra, Rb = elo[A], elo[B]
        
        Ea = 1 / (1 + 10 ** ((Rb - Ra) / 400))
        Sa = 1 if row["margin"] > 0 else 0
        
        margin_factor = np.log(abs(row["margin"]) + 1)
        
        elo[A] += k * margin_factor * (Sa - Ea)
        elo[B] += k * margin_factor * ((1 - Sa) - (1 - Ea))
    
    return elo

elo = compute_elo(train_df)
print("Elo computed")

# =========================
# BAYESIAN-STYLE STRENGTH
# =========================
home_strength = train_df.groupby("HomeTeam")["margin"].mean()
away_strength = train_df.groupby("AwayTeam")["margin"].mean()

teams = set(home_strength.index).union(set(away_strength.index))

bayes_strength = {}
for t in teams:
    h = home_strength.get(t, 0)
    a = -away_strength.get(t, 0)
    bayes_strength[t] = 0.6 * h + 0.4 * a

# =========================
# RECENT FORM
# =========================
recent_form = {}

for team in teams:
    games = train_df[
        (train_df["HomeTeam"] == team) |
        (train_df["AwayTeam"] == team)
    ].tail(5)
    
    recent_form[team] = games["margin"].mean() if len(games) > 0 else 0

# =========================
# FINAL STRENGTH
# =========================
# ELO is the most recent and dynamic, so we give it a decent weight. Bayesian strength captures overall quality, and recent form captures momentum.
final_strength = {}

for t in teams:
    elo_norm = (elo.get(t, 2000) - 2000) / 100 # Normalize around 2000 with a scale of 100
    
    final_strength[t] = (
        0.5 * bayes_strength.get(t, 0) + # Strength baseline
        0.3 * elo_norm + # Elo contribution
        0.2 * recent_form.get(t, 0) # Recent form contribution
    )

# =========================
# ML FEATURES
# =========================
train_df["elo_home"] = train_df["HomeTeam"].map(elo)
train_df["elo_away"] = train_df["AwayTeam"].map(elo)

train_df["elo_diff"] = train_df["elo_home"] - train_df["elo_away"]

train_df["form_home"] = train_df.groupby("HomeTeam")["margin"].rolling(5).mean().reset_index(0, drop=True)
train_df["form_away"] = train_df.groupby("AwayTeam")["margin"].rolling(5).mean().reset_index(0, drop=True)

train_df["form_diff"] = train_df["form_home"] - train_df["form_away"]

train_df.fillna(0, inplace=True)

X = train_df[["elo_diff", "form_diff"]]
y = train_df["margin"]

# =========================
# TRAIN ML MODELS
# =========================
xgb = XGBRegressor(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.04,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)

ridge = Ridge(alpha=1.0)

xgb.fit(X, y)
ridge.fit(X, y)

print("ML models trained")

# =========================
# ESTIMATE NOISE (SIGMA)
# =========================
train_pred = 0.6 * xgb.predict(X) + 0.4 * ridge.predict(X)
residuals = y - train_pred

# Robust estimate (less sensitive to outliers)
sigma = np.std(residuals)
print(f"Estimated sigma: {sigma:.3f}")

# =========================
# SIMULATION PREDICTION
# =========================
rng = np.random.default_rng(42)

def predict_margin(team1, team2, n_sim=2000):
    # ===== Strength model =====
    s1 = final_strength.get(team1, 0)
    s2 = final_strength.get(team2, 0)
    strength_pred = s1 - s2

    # ===== Elo feature =====
    elo_diff = elo.get(team1, 2000) - elo.get(team2, 2000)

    # ===== ML prediction =====
    X_input = np.array([[elo_diff, 0]])
    xgb_pred = xgb.predict(X_input)[0]
    ridge_pred = ridge.predict(X_input)[0]
    ml_pred = 0.6 * xgb_pred + 0.4 * ridge_pred

    # ===== Mean prediction =====
    mu = (
        0.5 * strength_pred +
        0.3 * ml_pred +
        0.2 * (elo_diff / 25)
    )

    # ===== Monte Carlo simulation =====
    sims = rng.normal(loc=mu, scale=sigma, size=n_sim)

    # Use MEDIAN (robust)
    return np.median(sims)

# =========================
# PREDICTIONS
# =========================
pred_df["Team1_WinMargin"] = pred_df.apply(
    lambda row: predict_margin(row["Team1"], row["Team2"]),
    axis=1
)

# =========================
# CALIBRATION (SIMULATION VERSION)
# =========================
# Slightly lower scaling (simulation already spreads predictions)
pred_df["Team1_WinMargin"] *= 1.25
# Smooth extremes
pred_df["Team1_WinMargin"] = np.tanh(pred_df["Team1_WinMargin"] / 18) * 35
# Remove bias
pred_df["Team1_WinMargin"] -= pred_df["Team1_WinMargin"].mean()
# Round
pred_df["Team1_WinMargin"] = pred_df["Team1_WinMargin"].round(1)


# --- Evaluation: RMSE and Winning Team Accuracy ---
actual_margins = sample_pred['Team1_WinMargin'].astype(int).values
predicted_margins = pred_df['Team1_WinMargin'].astype(int).values
if len(actual_margins) == 75 and len(predicted_margins) == 75:
    rmse = np.sqrt(np.mean((predicted_margins - actual_margins) ** 2))
    print(f"RMSE (Predicted vs Actual Margins): {rmse:.2f}")
    actual_winners = np.sign(actual_margins)
    predicted_winners = np.sign(predicted_margins)
    correct_winner = (actual_winners == predicted_winners).sum()
    accuracy = correct_winner / 75 * 100
    print(f"Winning Team Prediction Accuracy: {accuracy:.2f}% ({correct_winner}/75)")
else:
    print("Warning: Cannot evaluate RMSE or accuracy, margin counts do not match.")


submission = pred_df[["GameID", "Team1_WinMargin"]]
# populate the field "Team1_WinMargin" with the expected victory margin from the perspective of Team1.
# drop the "Team1_WinMargin" column from the sample_pred and merge on "GameID" to ensure correct formatting
sample_pred = sample_pred.drop(columns=["Team1_WinMargin"])
submission = sample_pred.merge(submission, on="GameID", how="inner")

submission.to_csv("algosports23-predictions-2025/Predictions.csv", index=False)

print("algosports23-predictions-2025/Predictions.csv saved")

# =========================
# RANKINGS
# =========================
ranking_df = pd.DataFrame({
    "Team": list(final_strength.keys()),
    "Strength": list(final_strength.values())
})

ranking_df = ranking_df.sort_values("Strength", ascending=False).reset_index(drop=True)
ranking_df["Rank"] = np.arange(1, len(ranking_df) + 1)

final_rankings = sample_rank[["TeamID", "Team"]].merge(
    ranking_df[["Team", "Rank"]],
    on="Team",
    how="left"
)

final_rankings.to_excel("algosports23-predictions-2025/Rankings.xlsx", index=False)
print("algosports23-predictions-2025/Rankings.xlsx saved")

# --- Submission Packaging ---
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


# file_path = "algosports23-predictions-2025/rpl_championship_pipeline.py"
# with open(file_path, "w") as f:
#     f.write(pipeline_code)

# file_path