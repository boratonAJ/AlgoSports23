import os
import zipfile
import pandas as pd
import numpy as np

from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, accuracy_score

# =========================
# CONFIG
# =========================
DATA_DIR = "algosports23-predictions-2025"
TRAIN_PATH = f"{DATA_DIR}/Train.csv"
PRED_PATH = f"{DATA_DIR}/Predictions.csv"
SAMPLE_PRED_PATH = f"{DATA_DIR}/Sample_Predictions.csv"
SAMPLE_RANK_PATH = f"{DATA_DIR}/Sample_Rankings.xlsx"

OUTPUT_PRED_PATH = f"{DATA_DIR}/Predictions.csv"
OUTPUT_RANK_PATH = f"{DATA_DIR}/Rankings.xlsx"
OUTPUT_ZIP_PATH = f"{DATA_DIR}/Submission.zip"

# Keep defaults exactly as in your script
BASE_ELO = 1500
ELO_K = 30
FORM_WINDOW = 5
EWM_ALPHA = 0.95
N_SIM = 3000
RANDOM_STATE = 42

# =========================
# LOAD DATA
# =========================
train_df = pd.read_csv(TRAIN_PATH)
pred_df = pd.read_csv(PRED_PATH)
sample_pred = pd.read_csv(SAMPLE_PRED_PATH)
sample_rank = pd.read_excel(SAMPLE_RANK_PATH)

print("Data loaded successfully")

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

sort_cols = [c for c in ["Date", "GameID"] if c in train_df.columns]
if sort_cols:
    train_df = train_df.sort_values(sort_cols).reset_index(drop=True)

# Neutral-site correction clue from competition description:
# train games had home advantage, derby predictions are neutral venue
home_advantage = float(train_df["margin"].mean())

# =========================
# ELO MODEL
# =========================
def compute_elo(df, k=20, base_elo=1500):
    teams = pd.unique(df[["HomeTeam", "AwayTeam"]].values.ravel())
    elo = {t: base_elo for t in teams}

    for _, row in df.iterrows():
        A, B = row["HomeTeam"], row["AwayTeam"]
        Ra, Rb = elo[A], elo[B]

        Ea = 1 / (1 + 10 ** ((Rb - Ra) / 400))
        Sa = 1 if row["margin"] > 0 else 0.5 if row["margin"] == 0 else 0

        margin_factor = np.log(abs(row["margin"]) + 1)
        update = k * margin_factor * (Sa - Ea)

        elo[A] += update
        elo[B] -= update

    return elo

elo = compute_elo(train_df, k=ELO_K, base_elo=BASE_ELO)
print("Elo computed")

# =========================
# BAYESIAN STRENGTH
# =========================
home_strength = train_df.groupby("HomeTeam")["margin"].mean()
away_strength = train_df.groupby("AwayTeam")["margin"].mean()

teams = set(home_strength.index).union(set(away_strength.index))

bayes_strength = {}
for t in teams:
    h = home_strength.get(t, 0.0)
    a = -away_strength.get(t, 0.0)
    bayes_strength[t] = 0.6 * h + 0.4 * a

# =========================
# RECENT FORM
# =========================
def compute_recent_form(df, teams):
    home = df[["HomeTeam", "margin"]].copy()
    home.columns = ["Team", "tm"]

    away = df[["AwayTeam", "margin"]].copy()
    away.columns = ["Team", "tm"]
    away["tm"] = -away["tm"]

    all_games = pd.concat([home, away], ignore_index=True)

    recent = {}
    for t in teams:
        vals = all_games.loc[all_games["Team"] == t, "tm"].tail(FORM_WINDOW)
        recent[t] = vals.mean() if len(vals) > 0 else 0.0

    return recent

recent_form = compute_recent_form(train_df, teams)

# =========================
# FINAL STRENGTH
# =========================
final_strength = {}
for t in teams:
    elo_norm = (elo.get(t, BASE_ELO) - BASE_ELO) / 100
    final_strength[t] = (
        0.5 * bayes_strength.get(t, 0.0)
        + 0.3 * elo_norm
        + 0.2 * recent_form.get(t, 0.0)
    )

# =========================
# DATE-DRIVEN REST FEATURES
# =========================
if "Date" in train_df.columns:
    train_df["prev_home_date"] = train_df.groupby("HomeTeam")["Date"].shift(1)
    train_df["prev_away_date"] = train_df.groupby("AwayTeam")["Date"].shift(1)

    train_df["rest_home"] = (train_df["Date"] - train_df["prev_home_date"]).dt.days
    train_df["rest_away"] = (train_df["Date"] - train_df["prev_away_date"]).dt.days
    train_df["rest_diff"] = train_df["rest_home"] - train_df["rest_away"]
else:
    train_df["rest_diff"] = 0.0

# =========================
# OFFENSE / DEFENSE FEATURES
# =========================
home_offense = train_df.groupby("HomeTeam")["HomePts"].mean()
away_offense = train_df.groupby("AwayTeam")["AwayPts"].mean()

home_defense = train_df.groupby("HomeTeam")["AwayPts"].mean()
away_defense = train_df.groupby("AwayTeam")["HomePts"].mean()

team_offense = {}
team_defense = {}

for t in teams:
    off_vals = []
    def_vals = []

    if t in home_offense.index:
        off_vals.append(home_offense[t])
    if t in away_offense.index:
        off_vals.append(away_offense[t])

    if t in home_defense.index:
        def_vals.append(home_defense[t])
    if t in away_defense.index:
        def_vals.append(away_defense[t])

    team_offense[t] = float(np.mean(off_vals)) if off_vals else 0.0
    team_defense[t] = float(np.mean(def_vals)) if def_vals else 0.0

# =========================
# FEATURE ENGINEERING
# =========================
train_df["elo_home"] = train_df["HomeTeam"].map(elo)
train_df["elo_away"] = train_df["AwayTeam"].map(elo)

train_df["bayes_home"] = train_df["HomeTeam"].map(bayes_strength)
train_df["bayes_away"] = train_df["AwayTeam"].map(bayes_strength)

train_df["recent_home"] = train_df["HomeTeam"].map(recent_form)
train_df["recent_away"] = train_df["AwayTeam"].map(recent_form)

train_df["off_home"] = train_df["HomeTeam"].map(team_offense)
train_df["off_away"] = train_df["AwayTeam"].map(team_offense)

train_df["def_home"] = train_df["HomeTeam"].map(team_defense)
train_df["def_away"] = train_df["AwayTeam"].map(team_defense)

train_df["elo_diff"] = train_df["elo_home"] - train_df["elo_away"]
train_df["bayes_diff"] = train_df["bayes_home"] - train_df["bayes_away"]
train_df["recent_diff"] = train_df["recent_home"] - train_df["recent_away"]
train_df["off_diff"] = train_df["off_home"] - train_df["off_away"]
train_df["def_diff"] = train_df["def_home"] - train_df["def_away"]

# rolling features (leakage-safe)
train_df["away_margin"] = -train_df["margin"]

train_df["form_home"] = (
    train_df.groupby("HomeTeam")["margin"]
    .transform(lambda s: s.shift(1).rolling(FORM_WINDOW, min_periods=1).mean())
)

train_df["form_away"] = (
    train_df.groupby("AwayTeam")["away_margin"]
    .transform(lambda s: s.shift(1).rolling(FORM_WINDOW, min_periods=1).mean())
)

train_df["form_diff"] = train_df["form_home"] - train_df["form_away"]

# EWM
train_df["ewm_home"] = (
    train_df.groupby("HomeTeam")["margin"]
    .transform(lambda s: s.shift(1).ewm(alpha=EWM_ALPHA, adjust=False).mean())
)

train_df["ewm_away"] = (
    train_df.groupby("AwayTeam")["away_margin"]
    .transform(lambda s: s.shift(1).ewm(alpha=EWM_ALPHA, adjust=False).mean())
)

train_df["ewm_diff"] = train_df["ewm_home"] - train_df["ewm_away"]

# volatility
train_df["std_home"] = (
    train_df.groupby("HomeTeam")["margin"]
    .transform(lambda s: s.shift(1).rolling(FORM_WINDOW, min_periods=2).std())
)

train_df["std_away"] = (
    train_df.groupby("AwayTeam")["away_margin"]
    .transform(lambda s: s.shift(1).rolling(FORM_WINDOW, min_periods=2).std())
)

train_df["vol_diff"] = train_df["std_home"] - train_df["std_away"]

# interactions
train_df["elo_x_form"] = train_df["elo_diff"] * train_df["form_diff"]
train_df["elo_x_bayes"] = train_df["elo_diff"] * train_df["bayes_diff"]
train_df["elo_x_off"] = train_df["elo_diff"] * train_df["off_diff"]

# Fill numeric columns only
feature_engineered_cols = [
    "elo_diff", "bayes_diff", "recent_diff",
    "off_diff", "def_diff", "rest_diff",
    "form_diff", "ewm_diff", "vol_diff",
    "elo_x_form", "elo_x_bayes", "elo_x_off"
]

train_df[feature_engineered_cols] = train_df[feature_engineered_cols].fillna(0.0)

feature_cols = [
    "elo_diff",
    "bayes_diff",
    "recent_diff",
    "off_diff",
    "def_diff",
    "rest_diff",
    "form_diff",
    "ewm_diff",
    "vol_diff",
    "elo_x_form",
    "elo_x_bayes",
    "elo_x_off",
]

X = train_df[feature_cols].copy()
y = train_df["margin"].copy()

# =========================
# TRAIN MODELS
# =========================
# Keep defaults exactly as in your script
cat = CatBoostRegressor(
    iterations=800,
    depth=12,
    learning_rate=0.4,
    loss_function="RMSE",
    verbose=False,
    random_seed=RANDOM_STATE
)

ridge = Ridge(alpha=1.0)

cat.fit(X, y)
ridge.fit(X, y)

# --- Margin bin classifier ---
margin_bins = [-np.inf, -10, -3, 3, 10, np.inf]
margin_labels = [0, 1, 2, 3, 4]  # 0 BigLoss, 1 SmallLoss, 2 Draw, 3 SmallWin, 4 BigWin

train_df["margin_bin"] = pd.cut(
    y,
    bins=margin_bins,
    labels=margin_labels
).astype(int)

clf = CatBoostClassifier(
    iterations=500,
    depth=10,
    learning_rate=0.4,
    loss_function="MultiClass",
    verbose=False,
    random_seed=RANDOM_STATE
)

clf.fit(X, train_df["margin_bin"])

print("Models trained (regressor, ridge, classifier)")

# =========================
# SIGMA & METRICS
# =========================
pred_train = 0.6 * cat.predict(X) + 0.4 * ridge.predict(X)
sigma = np.std(y - pred_train)
mae = mean_absolute_error(y, pred_train)
acc = accuracy_score(np.sign(y), np.sign(pred_train))

print(f"Sigma: {sigma:.4f}")
print(f"MAE: {mae:.4f}")
print(f"Win/Loss Accuracy: {acc:.2%}")

rng = np.random.default_rng(RANDOM_STATE)

# =========================
# PREDICTION HELPERS
# =========================
bucket_names = ["BigLoss", "SmallLoss", "Draw", "SmallWin", "BigWin"]
bucket_margin_centers = np.array([-15, -6, 0, 6, 15], dtype=float)

def build_row(t1, t2):
    elo_diff = elo.get(t1, BASE_ELO) - elo.get(t2, BASE_ELO)
    bayes_diff = bayes_strength.get(t1, 0.0) - bayes_strength.get(t2, 0.0)
    recent_diff = recent_form.get(t1, 0.0) - recent_form.get(t2, 0.0)
    off_diff = team_offense.get(t1, 0.0) - team_offense.get(t2, 0.0)
    def_diff = team_defense.get(t1, 0.0) - team_defense.get(t2, 0.0)

    # Prediction-time proxies
    form_diff = recent_diff
    ewm_diff = recent_diff
    vol_diff = 0.0
    rest_diff = 0.0  # unknown future rest without schedule-derived construction

    return pd.DataFrame([{
        "elo_diff": elo_diff,
        "bayes_diff": bayes_diff,
        "recent_diff": recent_diff,
        "off_diff": off_diff,
        "def_diff": def_diff,
        "rest_diff": rest_diff,
        "form_diff": form_diff,
        "ewm_diff": ewm_diff,
        "vol_diff": vol_diff,
        "elo_x_form": elo_diff * form_diff,
        "elo_x_bayes": elo_diff * bayes_diff,
        "elo_x_off": elo_diff * off_diff
    }])[feature_cols]

def expand_classifier_probs(raw_probs, seen_classes, full_n=5):
    full = np.zeros(full_n, dtype=float)
    for j, cls in enumerate(seen_classes):
        full[int(cls)] = raw_probs[j]
    return full

# =========================================================
# PREDICTION FUNCTION
# =========================================================
def predict_margin_and_likelihood(t1, t2):
    row = build_row(t1, t2)

    # Regression blend
    ml = 0.6 * cat.predict(row)[0] + 0.4 * ridge.predict(row)[0]

    # Strength anchor
    strength = final_strength.get(t1, 0.0) - final_strength.get(t2, 0.0)

    # Neutral-site correction: remove train-era home advantage effect
    mu = (
        0.5 * strength +
        0.3 * ml +
        0.2 * (row["elo_diff"].iloc[0] / 25.0)
    )
    mu -= 0.5 * home_advantage

    sims = rng.normal(mu, sigma, N_SIM)
    median_margin = float(np.median(sims))

    # Class-safe bucket probabilities
    raw_bucket_probs = clf.predict_proba(row)[0]
    bucket_probs = expand_classifier_probs(raw_bucket_probs, clf.classes_, full_n=5)

    likelihood_dict = dict(zip(bucket_names, bucket_probs))

    p_team1_win = float(bucket_probs[3] + bucket_probs[4])
    p_team2_win = float(bucket_probs[0] + bucket_probs[1])
    p_close = float(bucket_probs[1] + bucket_probs[2] + bucket_probs[3])
    expected_margin = float(np.dot(bucket_probs, bucket_margin_centers))

    likelihood_dict["P_Team1Win"] = p_team1_win
    likelihood_dict["P_Team2Win"] = p_team2_win
    likelihood_dict["P_Close"] = p_close
    likelihood_dict["ExpectedMargin"] = expected_margin
    likelihood_dict["PredictedSigma"] = float(sigma)

    return median_margin, likelihood_dict

# =========================
# PREDICTIONS
# =========================
likelihood_cols = [
    "BigLoss", "SmallLoss", "Draw", "SmallWin", "BigWin",
    "P_Team1Win", "P_Team2Win", "P_Close", "ExpectedMargin", "PredictedSigma"
]

def predict_row(r):
    margin, likelihoods = predict_margin_and_likelihood(r["Team1"], r["Team2"])
    return pd.Series([margin] + [likelihoods.get(col, 0.0) for col in likelihood_cols])

meta_features = pred_df.apply(predict_row, axis=1)

pred_df["Team1_WinMargin"] = meta_features.iloc[:, 0]
for i, col in enumerate(likelihood_cols):
    pred_df[col] = meta_features.iloc[:, i + 1]

# calibration
pred_df["Team1_WinMargin"] *= 1.35
pred_df["Team1_WinMargin"] = np.tanh(pred_df["Team1_WinMargin"] / 18) * 35
pred_df["Team1_WinMargin"] -= pred_df["Team1_WinMargin"].mean()
pred_df["Team1_WinMargin"] = pred_df["Team1_WinMargin"].round().astype(int)

# =========================
# SAVE
# =========================
submission = pred_df[["GameID", "Team1_WinMargin"]].copy()

template = sample_pred.drop(columns=["Team1_WinMargin"])
submission = template.merge(submission, on="GameID", how="inner")
submission.to_csv(OUTPUT_PRED_PATH, index=False)

# Optional diagnostic likelihood file
likelihood_output_cols = ["GameID", "Team1", "Team2", "Date"] if "Date" in pred_df.columns else ["GameID", "Team1", "Team2"]
likelihood_output_cols += ["Team1_WinMargin"] + likelihood_cols

pred_df[likelihood_output_cols].to_csv(
    f"{DATA_DIR}/Prediction_Likelihoods.csv",
    index=False
)

print("Predictions saved")

# =========================
# RANKINGS
# =========================
ranking_df = pd.DataFrame({
    "Team": list(final_strength.keys()),
    "Strength": list(final_strength.values())
}).sort_values("Strength", ascending=False)

ranking_df["Rank"] = range(1, len(ranking_df) + 1)

final_rankings = sample_rank[["TeamID", "Team"]].merge(
    ranking_df[["Team", "Rank"]],
    on="Team",
    how="left"
)

final_rankings.to_excel(OUTPUT_RANK_PATH, index=False)
print("Rankings saved")

# =========================
# ZIP
# =========================
with zipfile.ZipFile(OUTPUT_ZIP_PATH, "w") as z:
    z.write(OUTPUT_PRED_PATH, "Predictions.csv")
    z.write(OUTPUT_RANK_PATH, "Rankings.xlsx")

print("Submission.zip created")