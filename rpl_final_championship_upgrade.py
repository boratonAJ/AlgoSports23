import os
import zipfile
import numpy as np
import pandas as pd

from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_squared_error

# =========================
# CONFIG
# =========================
DATA_DIR = "algosports23-predictions-2025"

BASE_ELO = 2000
ELO_K = 20
FORM_WINDOW = 5
EWM_ALPHA = 0.35
SIMS = 4000

# =========================
# LOAD
# =========================
train = pd.read_csv(f"{DATA_DIR}/Train.csv")
pred = pd.read_csv(f"{DATA_DIR}/Predictions.csv")
sample_rank = pd.read_excel(f"{DATA_DIR}/Sample_Rankings.xlsx")

train["Date"] = pd.to_datetime(train["Date"])
train = train.sort_values("Date")

train["margin"] = train["HomePts"] - train["AwayPts"]
train["win"] = (train["margin"] > 0).astype(int)

# =========================
# HOME ADVANTAGE
# =========================
home_adv = train["margin"].mean()

# =========================
# ELO
# =========================
def compute_elo(df):
    teams = pd.unique(df[["HomeTeam", "AwayTeam"]].values.ravel())
    elo = {t: BASE_ELO for t in teams}

    for _, r in df.iterrows():
        A, B = r["HomeTeam"], r["AwayTeam"]
        Ea = 1/(1+10**((elo[B]-elo[A])/400))
        Sa = 1 if r["margin"] > 0 else 0
        update = ELO_K * np.log(abs(r["margin"])+1) * (Sa - Ea)
        elo[A] += update
        elo[B] -= update

    return elo

elo = compute_elo(train)
teams = set(train["HomeTeam"]).union(set(train["AwayTeam"]))

# =========================
# TEAM OFF / DEF
# =========================
offense = train.groupby("HomeTeam")["HomePts"].mean()
defense = train.groupby("HomeTeam")["AwayPts"].mean()

# =========================
# REST DAYS
# =========================
train["prev_date_home"] = train.groupby("HomeTeam")["Date"].shift(1)
train["prev_date_away"] = train.groupby("AwayTeam")["Date"].shift(1)

train["rest_home"] = (train["Date"] - train["prev_date_home"]).dt.days.fillna(7)
train["rest_away"] = (train["Date"] - train["prev_date_away"]).dt.days.fillna(7)

# =========================
# FORM + MOMENTUM
# =========================
train["form_home"] = train.groupby("HomeTeam")["margin"].shift().rolling(5).mean()
train["form_away"] = -train.groupby("AwayTeam")["margin"].shift().rolling(5).mean()

train["ewm_home"] = train.groupby("HomeTeam")["margin"].shift().ewm(alpha=EWM_ALPHA).mean()
train["ewm_away"] = -train.groupby("AwayTeam")["margin"].shift().ewm(alpha=EWM_ALPHA).mean()

# =========================
# FEATURE BUILD
# =========================
train["elo_diff"] = train["HomeTeam"].map(elo) - train["AwayTeam"].map(elo)

train["off_diff"] = train["HomeTeam"].map(offense) - train["AwayTeam"].map(offense)
train["def_diff"] = train["HomeTeam"].map(defense) - train["AwayTeam"].map(defense)

train["rest_diff"] = train["rest_home"] - train["rest_away"]

train["form_diff"] = train["form_home"] - train["form_away"]
train["ewm_diff"] = train["ewm_home"] - train["ewm_away"]

train["momentum_diff"] = train["form_diff"] - train["ewm_diff"]

# Fill non-datetime columns with 0
non_dt_cols = train.select_dtypes(exclude=['datetime64[ns]']).columns
train[non_dt_cols] = train[non_dt_cols].fillna(0)
# Fill datetime columns with a default date
dt_cols = train.select_dtypes(include=['datetime64[ns]']).columns
train[dt_cols] = train[dt_cols].fillna(pd.Timestamp('1900-01-01'))

FEATURES = [
    "elo_diff",
    "off_diff",
    "def_diff",
    "rest_diff",
    "form_diff",
    "ewm_diff",
    "momentum_diff"
]

X = train[FEATURES]
y = train["margin"]
y_win = train["win"]

# =========================
# OOF STACK
# =========================
tscv = TimeSeriesSplit(n_splits=5)

oof_mid = np.zeros(len(X))
oof_prob = np.zeros(len(X))

for tr, val in tscv.split(X):
    X_tr, X_val = X.iloc[tr], X.iloc[val]
    y_tr = y.iloc[tr]
    y_win_tr = y_win.iloc[tr]

    reg = CatBoostRegressor(iterations=400, depth=6, verbose=False)
    reg.fit(X_tr, y_tr)

    clf = CatBoostClassifier(iterations=300, verbose=False)
    clf.fit(X_tr, y_win_tr)

    oof_mid[val] = reg.predict(X_val)
    oof_prob[val] = clf.predict_proba(X_val)[:,1]

# =========================
# CALIBRATION
# =========================
iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(oof_prob, y_win)
oof_prob = iso.predict(oof_prob)

# =========================
# META
# =========================
meta_X = pd.DataFrame({
    "mid": oof_mid,
    "prob": oof_prob,
    "elo": X["elo_diff"]
})

scaler = StandardScaler()
meta = RidgeCV(alphas=np.logspace(-3,3,10))
meta.fit(scaler.fit_transform(meta_X), y)

# =========================
# FINAL TRAIN
# =========================
reg = CatBoostRegressor(iterations=600, depth=6, verbose=False)
reg.fit(X, y)

clf = CatBoostClassifier(iterations=400, verbose=False)
clf.fit(X, y_win)

# =========================
# PREDICT
# =========================
rng = np.random.default_rng(42)

def build_row(t1, t2):
    return pd.DataFrame([{
        "elo_diff": elo[t1] - elo[t2],
        "off_diff": offense[t1] - offense[t2],
        "def_diff": defense[t1] - defense[t2],
        "rest_diff": 0,
        "form_diff": 0,
        "ewm_diff": 0,
        "momentum_diff": 0
    }])

def predict_game(t1, t2):
    Xp = build_row(t1, t2)

    mu = reg.predict(Xp)[0]

    prob = iso.predict([clf.predict_proba(Xp)[:,1][0]])[0]

    # remove home advantage (neutral)
    mu -= home_adv * 0.5

    sigma = 12

    sims = rng.normal(mu, sigma, SIMS)
    return float(np.median(sims))

# =========================
# OUTPUT
# =========================
pred["Team1_WinMargin"] = pred.apply(
    lambda r: predict_game(r["Team1"], r["Team2"]),
    axis=1
)

pred["Team1_WinMargin"] = np.tanh(pred["Team1_WinMargin"]/18)*35
pred["Team1_WinMargin"] -= pred["Team1_WinMargin"].mean()

pred.to_csv(f"{DATA_DIR}/Predictions.csv", index=False)

# =========================
# RANKINGS
# =========================
rank_df = pd.DataFrame({
    "Team": list(teams),
    "Strength": [elo[t] for t in teams]
}).sort_values("Strength", ascending=False)

rank_df["Rank"] = np.arange(1,len(rank_df)+1)

rank_final = sample_rank.merge(rank_df, on="Team")
rank_final.to_excel(f"{DATA_DIR}/Rankings.xlsx", index=False)

# =========================
# ZIP
# =========================
with zipfile.ZipFile(f"{DATA_DIR}/Submission.zip","w") as z:
    z.write(f"{DATA_DIR}/Predictions.csv","Predictions.csv")
    z.write(f"{DATA_DIR}/Rankings.xlsx","Rankings.xlsx")

print("🏆 UPGRADED MODEL COMPLETE")

# --- Evaluation: RMSE and Winning Team Accuracy ---
try:
    sample_pred = pd.read_csv(f"{DATA_DIR}/Sample_Predictions.csv")
    actual_margins = sample_pred['Team1_WinMargin'].astype(int).values
    predicted_margins = pred['Team1_WinMargin'].astype(int).values
    n = min(len(actual_margins), len(predicted_margins))
    if n > 0:
        # RMSE calculation
        rmse = np.sqrt(np.mean((predicted_margins[:n] - actual_margins[:n]) ** 2))
        print(f"RMSE (Predicted vs Actual Margins): {rmse:.2f}")
        # Winning team accuracy
        actual_winners = np.sign(actual_margins[:n])
        predicted_winners = np.sign(predicted_margins[:n])
        correct_winner = (actual_winners == predicted_winners).sum()
        accuracy = correct_winner / n * 100
        print(f"Winning Team Prediction Accuracy: {accuracy:.2f}% ({correct_winner}/{n})")
    else:
        print("Warning: Cannot evaluate RMSE or accuracy, margin counts do not match.")
except Exception as e:
    print(f"Evaluation error: {e}")     