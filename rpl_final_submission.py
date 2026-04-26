import os
import zipfile
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from lightgbm import LGBMClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score

# =========================
# CONFIG
# =========================
DATA_DIR = "algosports23-predictions-2025"

BASE_ELO = 2000
ELO_K = 30
FORM_WINDOW = 5
N_SIM = 3000
RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

# =========================
# LOAD DATA
# =========================
train_df = pd.read_csv(f"{DATA_DIR}/Train.csv")
pred_df = pd.read_csv(f"{DATA_DIR}/Predictions.csv")
sample_pred = pd.read_csv(f"{DATA_DIR}/Sample_Predictions.csv")
sample_rank = pd.read_excel(f"{DATA_DIR}/Sample_Rankings.xlsx")

train_df["margin"] = train_df["HomePts"] - train_df["AwayPts"]
train_df = train_df.sort_values(["Date", "GameID"]).reset_index(drop=True)

# =========================
# ELO MODEL
# =========================
def compute_elo(df):
    teams = pd.unique(df[["HomeTeam", "AwayTeam"]].values.ravel())
    elo = {t: BASE_ELO for t in teams}

    for _, r in df.iterrows():
        A, B = r["HomeTeam"], r["AwayTeam"]
        Ea = 1 / (1 + 10 ** ((elo[B] - elo[A]) / 400))
        Sa = 1 if r["margin"] > 0 else 0.5 if r["margin"] == 0 else 0

        update = ELO_K * np.log(abs(r["margin"]) + 1) * (Sa - Ea)

        elo[A] += update
        elo[B] -= update

    return elo

elo = compute_elo(train_df)
teams = set(train_df["HomeTeam"]).union(set(train_df["AwayTeam"]))

# =========================
# STRENGTH MODEL
# =========================
home_strength = train_df.groupby("HomeTeam")["margin"].mean()
away_strength = train_df.groupby("AwayTeam")["margin"].mean()

bayes_strength = {
    t: 0.6 * home_strength.get(t, 0) - 0.4 * away_strength.get(t, 0)
    for t in teams
}

recent_form = {}
for t in teams:
    vals = train_df[(train_df["HomeTeam"] == t) | (train_df["AwayTeam"] == t)]["margin"].tail(FORM_WINDOW)
    recent_form[t] = vals.mean() if len(vals) else 0

final_strength = {
    t: 0.5 * bayes_strength[t]
       + 0.3 * ((elo[t] - BASE_ELO) / 100)
       + 0.2 * recent_form[t]
    for t in teams
}

# =========================
# FEATURES
# =========================
train_df["elo_diff"] = train_df["HomeTeam"].map(elo) - train_df["AwayTeam"].map(elo)

train_df["form_diff"] = (
    train_df.groupby("HomeTeam")["margin"].shift().rolling(FORM_WINDOW).mean()
    - train_df.groupby("AwayTeam")["margin"].shift().rolling(FORM_WINDOW).mean()
)

train_df[["elo_diff", "form_diff"]] = train_df[["elo_diff", "form_diff"]].fillna(0)

X = train_df[["elo_diff", "form_diff"]]
y = train_df["margin"]

# =========================
# MODELS
# =========================
ridge = Ridge(alpha=1.0)
lgb = LGBMRegressor(objective="regression", n_estimators=800, learning_rate=0.4, random_state=42)

ridge.fit(X, y)
lgb.fit(X, y)

# Classification (margin buckets)
bins = [-np.inf, -10, -3, 3, 10, np.inf]
train_df["margin_bin"] = pd.cut(y, bins=bins, labels=[0,1,2,3,4])

clf = LGBMClassifier(objective="multiclass", num_class=5, random_state=42)
clf.fit(X, train_df["margin_bin"])

# =========================
# SIGMA
# =========================
pred_train = 0.15 * ridge.predict(X) + 0.7 * lgb.predict(X) + 0.15 * clf.predict(X)
sigma = np.std(y - pred_train)

# pred_train = 0.6 * cat.predict(X) + 0.2 * ridge.predict(X) + 0.2 * lgreg.predict(X)
mae = mean_absolute_error(y, pred_train)
acc = accuracy_score(np.sign(y), np.sign(pred_train))
print(f"Sigma: {sigma:.4f}")
print(f"MAE: {mae:.4f}")
print(f"Win/Loss Accuracy: {acc:.2%}")

# =========================
# FEATURE BUILDER
# =========================
def build_row(t1, t2):
    return pd.DataFrame([{
        "elo_diff": elo[t1] - elo[t2],
        "form_diff": recent_form[t1] - recent_form[t2]
    }])

# =========================
# FINAL STRONG PREDICTION
# =========================
def predict_game(t1, t2):

    row = build_row(t1, t2)

    # --- regression ---
    ml = 0.7 * lgb.predict(row)[0] + 0.3 * ridge.predict(row)[0]

    # --- strength ---
    strength = final_strength[t1] - final_strength[t2]

    base_mu = 0.5 * strength + 0.3 * ml + 0.2 * (row["elo_diff"].iloc[0] / 25)

    # --- probabilities ---
    probs = clf.predict_proba(row)[0]

    p_win = probs[3] + probs[4]
    expected_margin = np.dot(probs, [-15, -6, 0, 6, 15])

    direction = 1 if p_win > 0.5 else -1

    # --- strong blending ---
    mu = (
        0.45 * base_mu +
        0.25 * expected_margin +
        0.20 * direction * (p_win * 12) +
        0.10 * (row["elo_diff"].iloc[0] / 25)
    )

    # --- shrink ---
    shrink = np.clip(1 - sigma / 25, 0.5, 1.0)
    mu *= shrink

    sims = rng.normal(mu, sigma, N_SIM)
    return np.median(sims)

# =========================
# GENERATE PREDICTIONS
# =========================
pred_df["Team1_WinMargin"] = pred_df.apply(
    lambda r: predict_game(r["Team1"], r["Team2"]),
    axis=1
)

pred_df["Team1_WinMargin"] = (
    np.tanh(pred_df["Team1_WinMargin"] / 18) * 35
)

pred_df["Team1_WinMargin"] -= pred_df["Team1_WinMargin"].mean()
pred_df["Team1_WinMargin"] = pred_df["Team1_WinMargin"].round().astype(int)

# =========================
# SAVE PREDICTIONS
# =========================
submission = pred_df[["GameID", "Team1_WinMargin"]]
submission = sample_pred.drop(columns=["Team1_WinMargin"]).merge(submission, on="GameID")

submission.to_csv(f"{DATA_DIR}/Predictions.csv", index=False)

# =========================
# ELITE RANKING (SIMULATION)
# =========================
team_stats = {t: {"w":0,"l":0,"t":0,"m":0} for t in teams}

for _, r in pred_df.iterrows():
    t1, t2, m = r["Team1"], r["Team2"], r["Team1_WinMargin"]

    if m > 0:
        team_stats[t1]["w"] += 1
        team_stats[t2]["l"] += 1
    elif m < 0:
        team_stats[t2]["w"] += 1
        team_stats[t1]["l"] += 1
    else:
        team_stats[t1]["t"] += 1
        team_stats[t2]["t"] += 1

    team_stats[t1]["m"] += m
    team_stats[t2]["m"] -= m

ranking_rows = []

for t in teams:
    w = team_stats[t]["w"]
    l = team_stats[t]["l"]
    ti = team_stats[t]["t"]

    g = w + l + ti
    win_pct = (w + 0.5 * ti) / g if g > 0 else 0

    score = (
        0.55 * final_strength[t] +
        0.30 * win_pct +
        0.15 * team_stats[t]["m"]
    )

    ranking_rows.append({"Team": t, "Score": score})

ranking_df = pd.DataFrame(ranking_rows)
ranking_df = ranking_df.sort_values("Score", ascending=False)
ranking_df["Rank"] = range(1, len(ranking_df) + 1)

final_rank = sample_rank[["TeamID", "Team"]].merge(ranking_df, on="Team")
final_rank = final_rank[["TeamID", "Team", "Rank"]]

final_rank.to_excel(f"{DATA_DIR}/Rankings.xlsx", index=False)

# =========================
# ZIP SUBMISSION
# =========================
with zipfile.ZipFile(f"{DATA_DIR}/Submission.zip", "w") as z:
    z.write(f"{DATA_DIR}/Predictions.csv", "Predictions.csv")
    z.write(f"{DATA_DIR}/Rankings.xlsx", "Rankings.xlsx")

print("🏆 FINAL SUBMISSION FILE READY")