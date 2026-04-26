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
# CONFIG (UNCHANGED DEFAULTS)
# =========================
DATA_DIR = "algosports23-predictions-2025"

BASE_ELO = 1500
ELO_K = 30
FORM_WINDOW = 5
EWM_ALPHA = 0.95
N_SIM = 3000
RANDOM_STATE = 42

SEEDS = [42, 99]
DEPTHS = [10, 12]

# =========================
# LOAD
# =========================
train = pd.read_csv(f"{DATA_DIR}/Train.csv")
pred = pd.read_csv(f"{DATA_DIR}/Predictions.csv")
sample_pred = pd.read_csv(f"{DATA_DIR}/Sample_Predictions.csv")
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
    teams = pd.unique(df[["HomeTeam","AwayTeam"]].values.ravel())
    elo = {t: BASE_ELO for t in teams}

    for _, r in df.iterrows():
        A,B = r["HomeTeam"], r["AwayTeam"]
        Ea = 1/(1+10**((elo[B]-elo[A])/400))
        Sa = 1 if r["margin"]>0 else 0
        update = ELO_K*np.log(abs(r["margin"])+1)*(Sa-Ea)
        elo[A]+=update
        elo[B]-=update
    return elo

elo = compute_elo(train)
teams = set(train["HomeTeam"]).union(set(train["AwayTeam"]))

# =========================
# OFF / DEF
# =========================
offense = train.groupby("HomeTeam")["HomePts"].mean()
defense = train.groupby("HomeTeam")["AwayPts"].mean()

# =========================
# FORM + REST
# =========================
train["prev_home"] = train.groupby("HomeTeam")["Date"].shift(1)
train["prev_away"] = train.groupby("AwayTeam")["Date"].shift(1)

train["rest_home"] = (train["Date"] - train["prev_home"]).dt.days
train["rest_away"] = (train["Date"] - train["prev_away"]).dt.days

train["form_home"] = train.groupby("HomeTeam")["margin"].shift().rolling(FORM_WINDOW).mean()
train["form_away"] = -train.groupby("AwayTeam")["margin"].shift().rolling(FORM_WINDOW).mean()

train["ewm_home"] = train.groupby("HomeTeam")["margin"].shift().ewm(alpha=EWM_ALPHA).mean()
train["ewm_away"] = -train.groupby("AwayTeam")["margin"].shift().ewm(alpha=EWM_ALPHA).mean()

# =========================
# FEATURES
# =========================
train["elo_diff"] = train["HomeTeam"].map(elo) - train["AwayTeam"].map(elo)
train["off_diff"] = train["HomeTeam"].map(offense) - train["AwayTeam"].map(offense)
train["def_diff"] = train["HomeTeam"].map(defense) - train["AwayTeam"].map(defense)
train["rest_diff"] = train["rest_home"] - train["rest_away"]
train["form_diff"] = train["form_home"] - train["form_away"]
train["ewm_diff"] = train["ewm_home"] - train["ewm_away"]
train["momentum_diff"] = train["form_diff"] - train["ewm_diff"]

FEATURES = [
    "elo_diff","off_diff","def_diff","rest_diff",
    "form_diff","ewm_diff","momentum_diff"
]

train[FEATURES] = train[FEATURES].fillna(0)

X = train[FEATURES]
y = train["margin"]
y_win = train["win"]

# =========================
# OOF STACK + SIGMA
# =========================
tscv = TimeSeriesSplit(n_splits=5)

oof_mid = np.zeros(len(X))
oof_sigma = np.zeros(len(X))
oof_prob = np.zeros(len(X))

for tr,val in tscv.split(X):

    X_tr,X_val = X.iloc[tr],X.iloc[val]
    y_tr = y.iloc[tr]

    mids, lows, highs = [],[],[]

    for seed in SEEDS:
        for d in DEPTHS:

            reg = CatBoostRegressor(
                iterations=800,
                depth=d,
                learning_rate=0.4,
                verbose=False,
                random_seed=seed
            )
            reg.fit(X_tr,y_tr)

            low = CatBoostRegressor(
                loss_function="Quantile:alpha=0.2",
                iterations=300,
                depth=d,
                verbose=False
            )
            high = CatBoostRegressor(
                loss_function="Quantile:alpha=0.8",
                iterations=300,
                depth=d,
                verbose=False
            )

            low.fit(X_tr,y_tr)
            high.fit(X_tr,y_tr)

            mids.append(reg.predict(X_val))
            lows.append(low.predict(X_val))
            highs.append(high.predict(X_val))

    mid = np.mean(mids,0)
    low = np.mean(lows,0)
    high = np.mean(highs,0)

    sigma = (high-low)/1.28

    clf = CatBoostClassifier(
        iterations=500,
        depth=10,
        learning_rate=0.4,
        verbose=False
    )
    clf.fit(X_tr,y_win.iloc[tr])

    prob = clf.predict_proba(X_val)[:,1]

    oof_mid[val] = mid
    oof_sigma[val] = sigma
    oof_prob[val] = prob

print("OOF built")

# =========================
# CALIBRATION
# =========================
iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(oof_prob, y_win)
oof_prob = iso.predict(oof_prob)

# =========================
# META MODEL
# =========================
meta_X = pd.DataFrame({
    "mid":oof_mid,
    "prob":oof_prob,
    "sigma":oof_sigma,
    "elo":X["elo_diff"]
})

scaler = StandardScaler()
meta = RidgeCV(alphas=np.logspace(-3,3,10))
meta.fit(scaler.fit_transform(meta_X), y)

print("Meta trained")

# =========================
# FINAL TRAIN
# =========================
models_mid, models_low, models_high = [],[],[]

for seed in SEEDS:
    for d in DEPTHS:

        reg = CatBoostRegressor(
            iterations=800,
            depth=d,
            learning_rate=0.4,
            verbose=False,
            random_seed=seed
        )
        reg.fit(X,y)
        models_mid.append(reg)

        low = CatBoostRegressor(
            loss_function="Quantile:alpha=0.2",
            iterations=400,
            depth=d,
            verbose=False
        )
        high = CatBoostRegressor(
            loss_function="Quantile:alpha=0.8",
            iterations=400,
            depth=d,
            verbose=False
        )

        low.fit(X,y)
        high.fit(X,y)

        models_low.append(low)
        models_high.append(high)

clf = CatBoostClassifier(
    iterations=500,
    depth=10,
    learning_rate=0.4,
    verbose=False
)
clf.fit(X,y_win)

# =========================
# PREDICT
# =========================
rng = np.random.default_rng(RANDOM_STATE)

def build_row(t1,t2):
    return pd.DataFrame([{
        "elo_diff":elo[t1]-elo[t2],
        "off_diff":offense[t1]-offense[t2],
        "def_diff":defense[t1]-defense[t2],
        "rest_diff":0,
        "form_diff":0,
        "ewm_diff":0,
        "momentum_diff":0
    }])

def predict_game(t1,t2):

    Xp = build_row(t1,t2)

    mid = np.mean([m.predict(Xp)[0] for m in models_mid])
    low = np.mean([m.predict(Xp)[0] for m in models_low])
    high = np.mean([m.predict(Xp)[0] for m in models_high])

    sigma = np.clip((high-low)/1.28,3,25)

    prob = iso.predict([clf.predict_proba(Xp)[:,1][0]])[0]

    meta_row = pd.DataFrame([{
        "mid":mid,
        "prob":prob,
        "sigma":sigma,
        "elo":Xp["elo_diff"].iloc[0]
    }])

    mu = meta.predict(scaler.transform(meta_row))[0]

    # neutral venue correction
    mu -= home_adv*0.5

    sims = rng.normal(mu,sigma,N_SIM)
    return float(np.median(sims))

# =========================
# OUTPUT
# =========================
pred["Team1_WinMargin"] = pred.apply(
    lambda r: predict_game(r["Team1"],r["Team2"]), axis=1
)

pred["Team1_WinMargin"] *= 1.3
pred["Team1_WinMargin"] = np.tanh(pred["Team1_WinMargin"]/18)*35
pred["Team1_WinMargin"] -= pred["Team1_WinMargin"].mean()
pred["Team1_WinMargin"] = pred["Team1_WinMargin"].round().astype(int)


pred.to_csv(f"{DATA_DIR}/Predictions.csv", index=False)

# =========================
# RANKINGS
# =========================
rank_df = pd.DataFrame({
    "Team":list(teams),
    "Strength":[elo[t] for t in teams]
}).sort_values("Strength",ascending=False)

rank_df["Rank"]=np.arange(1,len(rank_df)+1)
final_rankings = sample_rank[["TeamID", "Team"]].merge(
    rank_df[["Team", "Rank"]],
    on="Team",
    how="left"
)
final_rankings.to_excel(f"{DATA_DIR}/Rankings.xlsx",index=False)
print("Rankings saved")


# =========================
# ZIP
# =========================
with zipfile.ZipFile(f"{DATA_DIR}/Submission.zip","w") as z:
    z.write(f"{DATA_DIR}/Predictions.csv","Predictions.csv")
    z.write(f"{DATA_DIR}/Rankings.xlsx","Rankings.xlsx")


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
pred_train = meta.predict(scaler.transform(meta_X_train))

sigma = np.std(y - pred_train)
mae = mean_absolute_error(y, pred_train)
acc = accuracy_score(np.sign(y), np.sign(pred_train))
print(f"Sigma: {sigma:.4f}")
print(f"MAE: {mae:.4f}")
print(f"Win/Loss Accuracy: {acc:.2%}")

print("🏆 PRO SPORTS BETTING MODEL COMPLETE")