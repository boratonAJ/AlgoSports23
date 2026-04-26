import os
import zipfile
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor, LGBMClassifier
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

BASE_ELO = 1500
ELO_K = 30
FORM_WINDOW = 5
EWM_ALPHA = 0.95
N_SIM = 3000
RANDOM_STATE = 42

EV_THRESHOLD = 0.02
PROB_THRESHOLD = 0.55
INITIAL_BANKROLL = 1000

rng = np.random.default_rng(RANDOM_STATE)

# =========================
# LOAD DATA
# =========================
train_df = pd.read_csv(TRAIN_PATH)
pred_df = pd.read_csv(PRED_PATH)
sample_pred = pd.read_csv(SAMPLE_PRED_PATH)
sample_rank = pd.read_excel(SAMPLE_RANK_PATH)

train_df["margin"] = train_df["HomePts"] - train_df["AwayPts"]

# =========================
# ELO
# =========================
def compute_elo(df):
    teams = pd.unique(df[["HomeTeam", "AwayTeam"]].values.ravel())
    elo = {t: BASE_ELO for t in teams}

    for _, row in df.iterrows():
        A, B = row["HomeTeam"], row["AwayTeam"]
        Ra, Rb = elo[A], elo[B]

        Ea = 1 / (1 + 10 ** ((Rb - Ra) / 400))
        Sa = 1 if row["margin"] > 0 else 0.5 if row["margin"] == 0 else 0

        update = ELO_K * np.log(abs(row["margin"]) + 1) * (Sa - Ea)

        elo[A] += update
        elo[B] -= update

    return elo

elo = compute_elo(train_df)

# =========================
# STRENGTH FEATURES
# =========================
home_strength = train_df.groupby("HomeTeam")["margin"].mean()
away_strength = train_df.groupby("AwayTeam")["margin"].mean()

teams = set(home_strength.index).union(set(away_strength.index))

bayes_strength = {
    t: 0.6 * home_strength.get(t, 0) - 0.4 * away_strength.get(t, 0)
    for t in teams
}

def compute_recent_form(df):
    home = df[["HomeTeam", "margin"]].rename(columns={"HomeTeam": "Team"})
    away = df[["AwayTeam", "margin"]].rename(columns={"AwayTeam": "Team"})
    away["margin"] *= -1

    all_games = pd.concat([home, away])

    return {
        t: all_games.loc[all_games["Team"] == t, "margin"].tail(FORM_WINDOW).mean()
        for t in teams
    }

recent_form = compute_recent_form(train_df)

# =========================
# FEATURE ENGINEERING
# =========================
def add_features(df):
    df = df.copy()

    df["elo_home"] = df["HomeTeam"].map(elo)
    df["elo_away"] = df["AwayTeam"].map(elo)
    df["elo_diff"] = df["elo_home"] - df["elo_away"]

    df["bayes_home"] = df["HomeTeam"].map(bayes_strength)
    df["bayes_away"] = df["AwayTeam"].map(bayes_strength)
    df["bayes_diff"] = df["bayes_home"] - df["bayes_away"]

    df["recent_home"] = df["HomeTeam"].map(recent_form)
    df["recent_away"] = df["AwayTeam"].map(recent_form)
    df["recent_diff"] = df["recent_home"] - df["recent_away"]

    df["form_diff"] = df["recent_diff"]
    df["ewm_diff"] = df["recent_diff"]
    df["vol_diff"] = 0

    df["elo_x_form"] = df["elo_diff"] * df["form_diff"]
    df["elo_x_bayes"] = df["elo_diff"] * df["bayes_diff"]

    return df.fillna(0)

train_df = add_features(train_df)

feature_cols = [
    "elo_diff", "bayes_diff", "recent_diff",
    "form_diff", "ewm_diff", "vol_diff",
    "elo_x_form", "elo_x_bayes"
]

X = train_df[feature_cols]
y = train_df["margin"]

# =========================
# MODELS
# =========================
ridge = Ridge(alpha=0.95, random_state=RANDOM_STATE)

lgb = LGBMRegressor(
    objective="regression",   # or "regression_l1", "quantile", etc.
    n_estimators=800,
    learning_rate=0.4,
    random_state=42
)

ridge.fit(X, y)
lgb.fit(X, y)

# --- Margin bin classifier ---
margin_bins = [-np.inf, -10, -3, 3, 10, np.inf]
margin_labels = [0, 1, 2, 3, 4]  # 0: BigLoss, 1: SmallLoss, 2: Draw, 3: SmallWin, 4: BigWin
train_df["margin_bin"] = pd.cut(y, bins=margin_bins, labels=margin_labels)

clf = LGBMClassifier(objective="multiclass", num_class=3)
clf.fit(X, train_df["margin_bin"])

# =========================
# SIGMA & METRICS
# =========================
pred_train = 0.6 * lgb.predict(X) + 0.4 * ridge.predict(X)
sigma = np.std(y - pred_train)
mae = mean_absolute_error(y, pred_train)
acc = accuracy_score(np.sign(y), np.sign(pred_train))
print(f"Sigma: {sigma:.4f}")
print(f"MAE: {mae:.4f}")
print(f"Win/Loss Accuracy: {acc:.2%}")

# =========================
# PREDICTION FUNCTION
# =========================
def predict_game(t1, t2):
    row = pd.DataFrame([{
        "elo_diff": elo.get(t1, BASE_ELO) - elo.get(t2, BASE_ELO),
        "bayes_diff": bayes_strength.get(t1, 0) - bayes_strength.get(t2, 0),
        "recent_diff": recent_form.get(t1, 0) - recent_form.get(t2, 0),
        "form_diff": 0,
        "ewm_diff": 0,
        "vol_diff": 0,
        "elo_x_form": 0,
        "elo_x_bayes": 0
    }])

    ml = 0.6 * lgb.predict(row)[0] + 0.4 * ridge.predict(row)[0]

    sims = rng.normal(ml, sigma, N_SIM)
    margin = np.median(sims)

    probs = clf.predict_proba(row)[0]

    return margin, {
        "P_Team1Win": probs[2],
        "P_Team2Win": probs[0]
    }

# =========================
# PREDICTIONS
# =========================
def predict_row(r):
    margin, probs = predict_game(r["Team1"], r["Team2"])
    return pd.Series([margin, probs["P_Team1Win"], probs["P_Team2Win"]])

pred_df[["Team1_WinMargin", "P_Team1Win", "P_Team2Win"]] = pred_df.apply(predict_row, axis=1)

# =========================
# 🔥 PROFIT ENGINE
# =========================
# Dummy odds (replace with real market odds)
pred_df["Odds_Team1"] = 2.0
pred_df["Odds_Team2"] = 2.0

def compute_ev(p, odds):
    return p * (odds - 1) - (1 - p)

pred_df["EV_Team1"] = compute_ev(pred_df["P_Team1Win"], pred_df["Odds_Team1"])
pred_df["EV_Team2"] = compute_ev(pred_df["P_Team2Win"], pred_df["Odds_Team2"])

def decide_bet(row):
    if row["EV_Team1"] > EV_THRESHOLD and row["P_Team1Win"] > PROB_THRESHOLD:
        return "Team1"
    elif row["EV_Team2"] > EV_THRESHOLD and row["P_Team2Win"] > PROB_THRESHOLD:
        return "Team2"
    return "No Bet"

pred_df["Bet"] = pred_df.apply(decide_bet, axis=1)

def kelly(p, odds):
    b = odds - 1
    return max((b * p - (1 - p)) / b, 0)

pred_df["Bet_Size"] = 0.5 * pred_df.apply(
    lambda r: max(
        kelly(r["P_Team1Win"], r["Odds_Team1"]),
        kelly(r["P_Team2Win"], r["Odds_Team2"])
    ),
    axis=1
)

# =========================
# PROFIT RANKING
# =========================
pred_df["EV_Max"] = pred_df[["EV_Team1", "EV_Team2"]].max(axis=1)

pred_df_sorted = pred_df.sort_values("EV_Max", ascending=False)

pred_df_sorted.to_csv(f"{DATA_DIR}/EV_Ranking.csv", index=False)

# =========================
# SAVE COMPETITION OUTPUT
# =========================
submission = pred_df[["GameID", "Team1_WinMargin"]]
template = sample_pred.drop(columns=["Team1_WinMargin"])

submission = template.merge(submission, on="GameID")
submission["Team1_WinMargin"] = submission["Team1_WinMargin"].round().astype(int)
submission.to_csv(OUTPUT_PRED_PATH, index=False)

# =========================
# TEAM RANKINGS
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

    # Pythagorean expectation (PythagWin)
    # Exponent 2.0 is common, can be tuned for sport
    pyth_exp = 2.0
    pythag_win = (points_scored ** pyth_exp) / (
        (points_scored ** pyth_exp) + (points_allowed ** pyth_exp)
    ) if (points_scored > 0 or points_allowed > 0) else 0.0

    predictive_score = 0.75 * win_pct + 0.25 * pythag_win

    team_rows.append({
        "Team": team,
        "W": wins,
        "L": losses,
        "T": ties,
        "W-L-T": f"{wins}-{losses}-{ties}",
        "PSG": round(psg, 2),
        "PAG": round(pag, 2),
        "WinPct": win_pct,
        "PythagWin": pythag_win,
        "PD": round(pdiff, 2),
        "PredictiveScore": predictive_score
    })


ranking_df = pd.DataFrame(team_rows)

# Sort by PredictiveScore, then break ties with PD (desc), PSG (desc), PAG (asc)
ranking_df = ranking_df.sort_values(
    by=["PredictiveScore", "PD", "PSG", "PAG"],
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
print("Rankings saved")

# =========================
# ZIP
# =========================
with zipfile.ZipFile(OUTPUT_ZIP_PATH, "w") as z:
    z.write(OUTPUT_PRED_PATH, "Predictions.csv")
    z.write(OUTPUT_RANK_PATH, "Rankings.xlsx")

print("🔥 Profit-optimized pipeline complete")