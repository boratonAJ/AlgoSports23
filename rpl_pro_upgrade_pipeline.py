
# =========================================================
# RPL PRO SPORTS BETTING PIPELINE (UPGRADED)
# =========================================================

import os
import zipfile
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor, LGBMClassifier

DATA_DIR = "algosports23-predictions-2025"

BASE_ELO = 2000
ELO_K = 30
FORM_WINDOW = 5
N_SIM = 3000
RANDOM_STATE = 42

rng = np.random.default_rng(RANDOM_STATE)

train_df = pd.read_csv(f"{DATA_DIR}/Train.csv")
pred_df = pd.read_csv(f"{DATA_DIR}/Predictions.csv")
sample_pred = pd.read_csv(f"{DATA_DIR}/Sample_Predictions.csv")
sample_rank = pd.read_excel(f"{DATA_DIR}/Sample_Rankings.xlsx")

train_df["margin"] = train_df["HomePts"] - train_df["AwayPts"]

def compute_elo(df):
    teams = pd.unique(df[["HomeTeam", "AwayTeam"]].values.ravel())
    elo = {t: BASE_ELO for t in teams}
    for _, r in df.iterrows():
        A, B = r["HomeTeam"], r["AwayTeam"]
        Ea = 1 / (1 + 10 ** ((elo[B] - elo[A]) / 400))
        Sa = 1 if r["margin"] > 0 else 0
        update = ELO_K * np.log(abs(r["margin"]) + 1) * (Sa - Ea)
        elo[A] += update
        elo[B] -= update
    return elo

elo = compute_elo(train_df)
teams = set(train_df["HomeTeam"]).union(set(train_df["AwayTeam"]))

train_df["elo_diff"] = train_df["HomeTeam"].map(elo) - train_df["AwayTeam"].map(elo)
train_df.fillna(0, inplace=True)

X = train_df[["elo_diff"]]
y = train_df["margin"]

ridge = Ridge()
lg = LGBMRegressor(n_estimators=800, learning_rate=0.4)

ridge.fit(X, y)
lg.fit(X, y)

pred_train = 0.3 * ridge.predict(X) + 0.7 * lg.predict(X)
sigma = np.std(y - pred_train)

def build_row(t1, t2):
    return pd.DataFrame([{
        "elo_diff": elo[t1] - elo[t2]
    }])

def predict_game(t1, t2):
    row = build_row(t1, t2)
    ml = 0.6 * lg.predict(row)[0] + 0.4 * ridge.predict(row)[0]
    sims = rng.normal(ml, sigma, N_SIM)
    return np.median(sims)

pred_df["Team1_WinMargin"] = pred_df.apply(
    lambda r: predict_game(r["Team1"], r["Team2"]), axis=1
)

pred_df["Team1_WinMargin"] = (
    np.tanh(pred_df["Team1_WinMargin"] / 18) * 35
).round().astype(int)

submission = pred_df[["GameID", "Team1_WinMargin"]]
submission = sample_pred.drop(columns=["Team1_WinMargin"]).merge(submission, on="GameID")
submission.to_csv(f"{DATA_DIR}/Predictions.csv", index=False)

team_stats = {t: {"w":0,"l":0,"t":0} for t in teams}

for _, r in pred_df.iterrows():
    t1, t2, m = r["Team1"], r["Team2"], r["Team1_WinMargin"]
    if m > 0:
        team_stats[t1]["w"] += 1
    elif m < 0:
        team_stats[t2]["w"] += 1

rows = []
for t in teams:
    score = team_stats[t]["w"]
    rows.append({"Team": t, "Score": score})

ranking_df = pd.DataFrame(rows).sort_values("Score", ascending=False)
ranking_df["Rank"] = range(1, len(ranking_df)+1)

final_rank = sample_rank[["TeamID","Team"]].merge(ranking_df,on="Team")
final_rank.to_excel(f"{DATA_DIR}/Rankings.xlsx", index=False)

with zipfile.ZipFile(f"{DATA_DIR}/Submission.zip","w") as z:
    z.write(f"{DATA_DIR}/Predictions.csv","Predictions.csv")
    z.write(f"{DATA_DIR}/Rankings.xlsx","Rankings.xlsx")

print("Pipeline saved")
