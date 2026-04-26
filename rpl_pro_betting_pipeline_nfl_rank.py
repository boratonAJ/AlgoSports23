import os
import zipfile
import numpy as np
import pandas as pd

from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, accuracy_score


# =========================
# CONFIG (DEFAULTS PRESERVED)
# =========================
DATA_DIR = "algosports23-predictions-2025"

TRAIN_PATH = f"{DATA_DIR}/Train.csv"
PRED_PATH = f"{DATA_DIR}/Predictions.csv"
SAMPLE_PRED_PATH = f"{DATA_DIR}/Sample_Predictions.csv"
SAMPLE_RANK_PATH = f"{DATA_DIR}/Sample_Rankings.xlsx"

OUTPUT_PRED_PATH = f"{DATA_DIR}/Predictions.csv"
OUTPUT_RANK_PATH = f"{DATA_DIR}/Rankings.xlsx"
OUTPUT_RANK_DIAG_PATH = f"{DATA_DIR}/Rankings_Diagnostics.xlsx"
OUTPUT_ZIP_PATH = f"{DATA_DIR}/Submission.zip"

BASE_ELO = 1500
ELO_K = 30
FORM_WINDOW = 5
EWM_ALPHA = 0.95
N_SIM = 2000
RANDOM_STATE = 42

SEEDS = [42, 99]
DEPTHS = [8, 10]

# =========================
# LOAD
# =========================
train_df = pd.read_csv(TRAIN_PATH)
pred_df = pd.read_csv(PRED_PATH)
sample_pred = pd.read_csv(SAMPLE_PRED_PATH)
sample_rank = pd.read_excel(SAMPLE_RANK_PATH)

print("Data loaded successfully.")

# =========================
# PREPROCESS
# =========================
train_df = train_df.copy()
pred_df = pred_df.copy()

if "Date" in train_df.columns:
    train_df["Date"] = pd.to_datetime(train_df["Date"], errors="coerce")
if "Date" in pred_df.columns:
    pred_df["Date"] = pd.to_datetime(pred_df["Date"], errors="coerce")

train_df["margin"] = train_df["HomePts"] - train_df["AwayPts"]
train_df["win"] = (train_df["margin"] > 0).astype(int)

sort_cols = [c for c in ["Date", "GameID"] if c in train_df.columns]
if sort_cols:
    train_df = train_df.sort_values(sort_cols).reset_index(drop=True)

# =========================
# HOME ADVANTAGE
# =========================
home_adv = float(train_df["margin"].mean())

# =========================
# ELO
# =========================
def compute_elo(df: pd.DataFrame) -> dict:
    teams_local = pd.unique(df[["HomeTeam", "AwayTeam"]].values.ravel())
    elo_local = {t: BASE_ELO for t in teams_local}

    for _, r in df.iterrows():
        team_a, team_b = r["HomeTeam"], r["AwayTeam"]
        expected_a = 1 / (1 + 10 ** ((elo_local[team_b] - elo_local[team_a]) / 400))
        actual_a = 1 if r["margin"] > 0 else 0
        update = ELO_K * np.log(abs(r["margin"]) + 1) * (actual_a - expected_a)
        elo_local[team_a] += update
        elo_local[team_b] -= update

    return elo_local

elo = compute_elo(train_df)
teams = set(train_df["HomeTeam"]).union(set(train_df["AwayTeam"]))

# =========================
# TEAM STRENGTH / OFF / DEF
# =========================
home_strength = train_df.groupby("HomeTeam")["margin"].mean()
away_strength = train_df.groupby("AwayTeam")["margin"].mean()

bayes_strength = {}
for team in teams:
    h = float(home_strength.get(team, 0.0))
    a = float(-away_strength.get(team, 0.0))
    bayes_strength[team] = 0.6 * h + 0.4 * a

recent_form = {}
for team in teams:
    vals = train_df.loc[
        (train_df["HomeTeam"] == team) | (train_df["AwayTeam"] == team),
        "margin"
    ].tail(FORM_WINDOW)
    recent_form[team] = float(vals.mean()) if len(vals) > 0 else 0.0

team_offense = {}
team_defense = {}

home_offense = train_df.groupby("HomeTeam")["HomePts"].mean()
away_offense = train_df.groupby("AwayTeam")["AwayPts"].mean()

home_defense = train_df.groupby("HomeTeam")["AwayPts"].mean()
away_defense = train_df.groupby("AwayTeam")["HomePts"].mean()

for team in teams:
    off_vals = []
    def_vals = []

    if team in home_offense.index:
        off_vals.append(home_offense[team])
    if team in away_offense.index:
        off_vals.append(away_offense[team])

    if team in home_defense.index:
        def_vals.append(home_defense[team])
    if team in away_defense.index:
        def_vals.append(away_defense[team])

    team_offense[team] = float(np.mean(off_vals)) if off_vals else 0.0
    team_defense[team] = float(np.mean(def_vals)) if def_vals else 0.0

final_strength = {}
for team in teams:
    elo_norm = (elo.get(team, BASE_ELO) - BASE_ELO) / 100.0
    final_strength[team] = (
        0.5 * bayes_strength.get(team, 0.0)
        + 0.3 * elo_norm
        + 0.2 * recent_form.get(team, 0.0)
    )

# =========================
# DATE / REST FEATURES
# =========================
if "Date" in train_df.columns:
    train_df["prev_home"] = train_df.groupby("HomeTeam")["Date"].shift(1)
    train_df["prev_away"] = train_df.groupby("AwayTeam")["Date"].shift(1)

    train_df["rest_home"] = (train_df["Date"] - train_df["prev_home"]).dt.days
    train_df["rest_away"] = (train_df["Date"] - train_df["prev_away"]).dt.days
    train_df["rest_diff"] = train_df["rest_home"] - train_df["rest_away"]
else:
    train_df["rest_diff"] = 0.0

# =========================
# FEATURE ENGINEERING
# =========================
train_df["elo_diff"] = train_df["HomeTeam"].map(elo) - train_df["AwayTeam"].map(elo)
train_df["off_diff"] = train_df["HomeTeam"].map(team_offense) - train_df["AwayTeam"].map(team_offense)
train_df["def_diff"] = train_df["HomeTeam"].map(team_defense) - train_df["AwayTeam"].map(team_defense)

train_df["form_home"] = train_df.groupby("HomeTeam")["margin"].transform(
    lambda s: s.shift(1).rolling(FORM_WINDOW, min_periods=1).mean()
)
train_df["form_away"] = train_df.groupby("AwayTeam")["margin"].transform(
    lambda s: (-s).shift(1).rolling(FORM_WINDOW, min_periods=1).mean()
)

train_df["ewm_home"] = train_df.groupby("HomeTeam")["margin"].transform(
    lambda s: s.shift(1).ewm(alpha=EWM_ALPHA, adjust=False).mean()
)
train_df["ewm_away"] = train_df.groupby("AwayTeam")["margin"].transform(
    lambda s: (-s).shift(1).ewm(alpha=EWM_ALPHA, adjust=False).mean()
)

train_df["form_diff"] = train_df["form_home"] - train_df["form_away"]
train_df["ewm_diff"] = train_df["ewm_home"] - train_df["ewm_away"]
train_df["momentum_diff"] = train_df["form_diff"] - train_df["ewm_diff"]

FEATURES = [
    "elo_diff",
    "off_diff",
    "def_diff",
    "rest_diff",
    "form_diff",
    "ewm_diff",
    "momentum_diff",
]

train_df[FEATURES] = train_df[FEATURES].fillna(0.0)

X = train_df[FEATURES].copy()
y = train_df["margin"].copy()
y_win = train_df["win"].copy()

# =========================
# OOF STACK + SIGMA
# =========================
tscv = TimeSeriesSplit(n_splits=5)

oof_mid = np.zeros(len(X))
oof_sigma = np.zeros(len(X))
oof_prob = np.zeros(len(X))

for tr_idx, val_idx in tscv.split(X):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr = y.iloc[tr_idx]

    mids, lows, highs = [], [], []

    for seed in SEEDS:
        for depth in DEPTHS:
            reg = CatBoostRegressor(
                iterations=1000,
                depth=depth,
                learning_rate=0.4,
                verbose=False,
                random_seed=seed
            )
            reg.fit(X_tr, y_tr)

            low = CatBoostRegressor(
                loss_function="Quantile:alpha=0.5",
                iterations=1000,
                learning_rate=0.4,
                depth=depth,
                verbose=False
            )
            high = CatBoostRegressor(
                loss_function="Quantile:alpha=0.5",
                iterations=1000,
                learning_rate=0.4,
                depth=depth,
                verbose=False
            )

            low.fit(X_tr, y_tr)
            high.fit(X_tr, y_tr)

            mids.append(reg.predict(X_val))
            lows.append(low.predict(X_val))
            highs.append(high.predict(X_val))

    mid = np.mean(mids, axis=0)
    low = np.mean(lows, axis=0)
    high = np.mean(highs, axis=0)
    sigma = (high - low) / 1.28

    clf_fold = CatBoostClassifier(
        iterations=400,
        depth=6,
        learning_rate=0.4,
        loss_function="MultiClass",
        verbose=False
    )
    clf_fold.fit(X_tr, y_win.iloc[tr_idx])
    prob = clf_fold.predict_proba(X_val)[:, 1]

    oof_mid[val_idx] = mid
    oof_sigma[val_idx] = sigma
    oof_prob[val_idx] = prob

print("OOF complete.")

# =========================
# CALIBRATION
# =========================
iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(oof_prob, y_win)
oof_prob_cal = iso.predict(oof_prob)

# =========================
# META MODEL
# =========================
meta_X = pd.DataFrame({
    "mid": oof_mid,
    "prob": oof_prob_cal,
    "sigma": oof_sigma,
    "elo": X["elo_diff"].values,
})

meta_scaler = StandardScaler()
meta = RidgeCV(alphas=np.logspace(-3, 3, 10))
meta.fit(meta_scaler.fit_transform(meta_X), y)

print("Meta model trained.")

# =========================
# FINAL TRAIN
# =========================
models_mid, models_low, models_high = [], [], []

for seed in SEEDS:
    for depth in DEPTHS:
        reg = CatBoostRegressor(
            iterations=1000,
            depth=depth,
            learning_rate=0.4,
            verbose=False,
            random_seed=seed
        )
        reg.fit(X, y)
        models_mid.append(reg)

        low = CatBoostRegressor(
            loss_function="Quantile:alpha=0.5",
            iterations=1000,
            learning_rate=0.4,
            depth=depth,
            verbose=False
        )
        high = CatBoostRegressor(
            loss_function="Quantile:alpha=0.5",
            iterations=1000,
            learning_rate=0.4,
            depth=depth,
            verbose=False
        )

        low.fit(X, y)
        high.fit(X, y)

        models_low.append(low)
        models_high.append(high)

clf = CatBoostClassifier(
    iterations=400,
    depth=6,
    learning_rate=0.4,
    loss_function="MultiClass",
    verbose=False
)
clf.fit(X, y_win)

print("Final models trained.")

# =========================
# PREDICTION HELPERS
# =========================
rng = np.random.default_rng(RANDOM_STATE)

def build_row(team1, team2):
    return pd.DataFrame([{
        "elo_diff": elo.get(team1, BASE_ELO) - elo.get(team2, BASE_ELO),
        "off_diff": team_offense.get(team1, 0.0) - team_offense.get(team2, 0.0),
        "def_diff": team_defense.get(team1, 0.0) - team_defense.get(team2, 0.0),
        "rest_diff": 0.0,
        "form_diff": recent_form.get(team1, 0.0) - recent_form.get(team2, 0.0),
        "ewm_diff": recent_form.get(team1, 0.0) - recent_form.get(team2, 0.0),
        "momentum_diff": 0.0
    }])

def predict_game(team1, team2):
    Xp = build_row(team1, team2)

    mid = np.mean([m.predict(Xp)[0] for m in models_mid])
    low = np.mean([m.predict(Xp)[0] for m in models_low])
    high = np.mean([m.predict(Xp)[0] for m in models_high])

    sigma = float(np.clip((high - low) / 1.28, 3, 25))
    prob = float(iso.predict([clf.predict_proba(Xp)[:, 1][0]])[0])

    meta_row = pd.DataFrame([{
        "mid": mid,
        "prob": prob,
        "sigma": sigma,
        "elo": Xp["elo_diff"].iloc[0]
    }])

    mu = float(meta.predict(meta_scaler.transform(meta_row))[0])

    # Neutral-site correction
    mu -= home_adv * 0.5

    sims = rng.normal(mu, sigma, N_SIM)
    return float(np.median(sims))

# =========================
# GENERATE PREDICTIONS
# =========================
pred_df["Team1_WinMargin"] = pred_df.apply(
    lambda r: predict_game(r["Team1"], r["Team2"]),
    axis=1
)

pred_df["Team1_WinMargin"] *= 1.3
pred_df["Team1_WinMargin"] = np.tanh(pred_df["Team1_WinMargin"] / 18) * 35
pred_df["Team1_WinMargin"] -= pred_df["Team1_WinMargin"].mean()
pred_df["Team1_WinMargin"] = pred_df["Team1_WinMargin"].round(0).astype(int)

# =========================
# SAVE PREDICTIONS
# =========================
submission = pred_df[["GameID", "Team1_WinMargin"]].copy()

sample_pred_template = sample_pred.copy()
if "Team1_WinMargin" in sample_pred_template.columns:
    sample_pred_template = sample_pred_template.drop(columns=["Team1_WinMargin"])

submission = sample_pred_template.merge(submission, on="GameID", how="inner")
submission.to_csv(OUTPUT_PRED_PATH, index=False)

print(f"{OUTPUT_PRED_PATH} saved.")

# =========================
# NFL-STYLE RANKINGS
# =========================
team_rows = []

for team in teams:
    home_games = train_df[train_df["HomeTeam"] == team]
    away_games = train_df[train_df["AwayTeam"] == team]

    home_wins = (home_games["margin"] > 0).sum()
    home_losses = (home_games["margin"] < 0).sum()
    home_ties = (home_games["margin"] == 0).sum()

    away_wins = (away_games["margin"] < 0).sum()
    away_losses = (away_games["margin"] > 0).sum()
    away_ties = (away_games["margin"] == 0).sum()

    wins = int(home_wins + away_wins)
    losses = int(home_losses + away_losses)
    ties = int(home_ties + away_ties)

    points_scored = float(home_games["HomePts"].sum() + away_games["AwayPts"].sum())
    points_allowed = float(home_games["AwayPts"].sum() + away_games["HomePts"].sum())

    games_played = wins + losses + ties
    psg = points_scored / games_played if games_played > 0 else 0.0
    pag = points_allowed / games_played if games_played > 0 else 0.0

    win_pct = (wins + 0.5 * ties) / games_played if games_played > 0 else 0.0
    pdiff = psg - pag

    team_rows.append({
        "Team": team,
        "W": wins,
        "L": losses,
        "T": ties,
        "W-L-T": f"{wins}-{losses}-{ties}",
        "PSG": round(psg, 2),
        "PAG": round(pag, 2),
        "WinPct": win_pct,
        "PD": round(pdiff, 2)
    })

ranking_df = pd.DataFrame(team_rows)

ranking_df = ranking_df.sort_values(
    by=["WinPct", "PD", "PSG", "PAG"],
    ascending=[False, False, False, True]
).reset_index(drop=True)

ranking_df["Rank"] = np.arange(1, len(ranking_df) + 1)

# Submission rankings
final_rankings = sample_rank[["TeamID", "Team"]].merge(
    ranking_df[["Team", "Rank"]],
    on="Team",
    how="left"
)

final_rankings["Rank"] = final_rankings["Rank"].astype(int)
final_rankings.to_excel(OUTPUT_RANK_PATH, index=False)

# Diagnostic rankings
ranking_diag = sample_rank[["TeamID", "Team"]].merge(
    ranking_df[["Team", "Rank", "W-L-T", "PSG", "PAG", "WinPct", "PD"]],
    on="Team",
    how="left"
)
ranking_diag.to_excel(OUTPUT_RANK_DIAG_PATH, index=False)

print(f"{OUTPUT_RANK_PATH} saved.")
print(f"{OUTPUT_RANK_DIAG_PATH} saved.")

# =========================
# ZIP
# =========================
with zipfile.ZipFile(OUTPUT_ZIP_PATH, "w") as z:
    z.write(OUTPUT_PRED_PATH, "Predictions.csv")
    z.write(OUTPUT_RANK_PATH, "Rankings.xlsx")

print(f"{OUTPUT_ZIP_PATH} created.")

# =========================
# SIGMA & METRICS
# =========================
# In-sample meta-model predictions for training data
meta_X_train = pd.DataFrame({
    "mid": oof_mid,
    "prob": oof_prob,
    "sigma": oof_sigma,
    "elo": X["elo_diff"]
})
pred_train = meta.predict(meta_scaler.transform(meta_X_train))

sigma = np.std(y - pred_train)
mae = mean_absolute_error(y, pred_train)
acc = accuracy_score(np.sign(y), np.sign(pred_train))
print(f"Sigma: {sigma:.4f}")
print(f"MAE: {mae:.4f}")
print(f"Win/Loss Accuracy: {acc:.2%}")

print("🏆 PRO SPORTS BETTING MODEL WITH NFL-STYLE RANKING COMPLETE")