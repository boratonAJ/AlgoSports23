import os
import zipfile
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge

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

BASE_ELO = 2000
ELO_K = 20
FORM_WINDOW = 5
EWM_ALPHA = 0.35
N_SIM = 2000
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
train_df["margin"] = train_df["HomePts"] - train_df["AwayPts"]

sort_cols = [c for c in ["Date", "GameID"] if c in train_df.columns]
if sort_cols:
    train_df = train_df.sort_values(sort_cols).reset_index(drop=True)

# =========================
# ELO MODEL
# =========================
def compute_elo(df, k=20, base_elo=2000):
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
    h = home_strength.get(t, 0)
    a = -away_strength.get(t, 0)
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

    all_games = pd.concat([home, away])
    recent = {}
    for t in teams:
        vals = all_games.loc[all_games["Team"] == t, "tm"].tail(FORM_WINDOW)
        recent[t] = vals.mean() if len(vals) > 0 else 0
    return recent

recent_form = compute_recent_form(train_df, teams)

# =========================
# FINAL STRENGTH
# =========================
final_strength = {}
for t in teams:
    elo_norm = (elo.get(t, BASE_ELO) - BASE_ELO) / 100
    final_strength[t] = (
        0.5 * bayes_strength.get(t, 0)
        + 0.3 * elo_norm
        + 0.2 * recent_form.get(t, 0)
    )

# =========================
# FEATURE ENGINEERING
# =========================
train_df["elo_home"] = train_df["HomeTeam"].map(elo)
train_df["elo_away"] = train_df["AwayTeam"].map(elo)

train_df["bayes_home"] = train_df["HomeTeam"].map(bayes_strength)
train_df["bayes_away"] = train_df["AwayTeam"].map(bayes_strength)

train_df["recent_home"] = train_df["HomeTeam"].map(recent_form)
train_df["recent_away"] = train_df["AwayTeam"].map(recent_form)

train_df["elo_diff"] = train_df["elo_home"] - train_df["elo_away"]
train_df["bayes_diff"] = train_df["bayes_home"] - train_df["bayes_away"]
train_df["recent_diff"] = train_df["recent_home"] - train_df["recent_away"]

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
    .transform(lambda s: s.shift(1).ewm(alpha=EWM_ALPHA).mean())
)
train_df["ewm_away"] = (
    train_df.groupby("AwayTeam")["away_margin"]
    .transform(lambda s: s.shift(1).ewm(alpha=EWM_ALPHA).mean())
)

train_df["ewm_diff"] = train_df["ewm_home"] - train_df["ewm_away"]

# volatility
train_df["std_home"] = (
    train_df.groupby("HomeTeam")["margin"]
    .transform(lambda s: s.shift(1).rolling(FORM_WINDOW).std())
)
train_df["std_away"] = (
    train_df.groupby("AwayTeam")["away_margin"]
    .transform(lambda s: s.shift(1).rolling(FORM_WINDOW).std())
)

train_df["vol_diff"] = train_df["std_home"] - train_df["std_away"]

# interactions
train_df["elo_x_form"] = train_df["elo_diff"] * train_df["form_diff"]
train_df["elo_x_bayes"] = train_df["elo_diff"] * train_df["bayes_diff"]

train_df.fillna(0, inplace=True)

feature_cols = [
    "elo_diff",
    "bayes_diff",
    "recent_diff",
    "form_diff",
    "ewm_diff",
    "vol_diff",
    "elo_x_form",
    "elo_x_bayes"
]

X = train_df[feature_cols]
y = train_df["margin"]

# =========================
# TRAIN MODELS
# =========================
cat = CatBoostRegressor(
    iterations=500,
    depth=6,
    learning_rate=0.5,
    loss_function="RMSE",
    verbose=False,
    random_seed=RANDOM_STATE
)

ridge = Ridge(alpha=1.0)

cat.fit(X, y)
ridge.fit(X, y)

print("Models trained")

# =========================
# SIGMA
# =========================
pred_train = 0.6 * cat.predict(X) + 0.4 * ridge.predict(X)
sigma = np.std(y - pred_train)
print("Sigma:", sigma)

rng = np.random.default_rng(RANDOM_STATE)

# =========================
# PREDICTION FUNCTION
# =========================
def build_row(t1, t2):
    elo_diff = elo.get(t1, BASE_ELO) - elo.get(t2, BASE_ELO)
    bayes_diff = bayes_strength.get(t1, 0) - bayes_strength.get(t2, 0)
    recent_diff = recent_form.get(t1, 0) - recent_form.get(t2, 0)

    form_diff = recent_diff
    ewm_diff = recent_diff
    vol_diff = 0

    return pd.DataFrame([{
        "elo_diff": elo_diff,
        "bayes_diff": bayes_diff,
        "recent_diff": recent_diff,
        "form_diff": form_diff,
        "ewm_diff": ewm_diff,
        "vol_diff": vol_diff,
        "elo_x_form": elo_diff * form_diff,
        "elo_x_bayes": elo_diff * bayes_diff
    }])

def predict_margin(t1, t2):
    row = build_row(t1, t2)
    ml = 0.6 * cat.predict(row)[0] + 0.4 * ridge.predict(row)[0]
    strength = final_strength.get(t1, 0) - final_strength.get(t2, 0)
    mu = 0.5 * strength + 0.3 * ml + 0.2 * (row["elo_diff"].iloc[0] / 25)

    sims = rng.normal(mu, sigma, N_SIM)
    return np.median(sims)

# =========================
# PREDICTIONS
# =========================
pred_df["Team1_WinMargin"] = pred_df.apply(
    lambda r: predict_margin(r["Team1"], r["Team2"]), axis=1
)

# calibration
pred_df["Team1_WinMargin"] *= 1.25
pred_df["Team1_WinMargin"] = np.tanh(pred_df["Team1_WinMargin"] / 18) * 35
pred_df["Team1_WinMargin"] -= pred_df["Team1_WinMargin"].mean()
pred_df["Team1_WinMargin"] = pred_df["Team1_WinMargin"].round(1)

# =========================
# SAVE
# =========================
submission = pred_df[["GameID", "Team1_WinMargin"]]
template = sample_pred.drop(columns=["Team1_WinMargin"])
submission = template.merge(submission, on="GameID")
submission.to_csv(OUTPUT_PRED_PATH, index=False)

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