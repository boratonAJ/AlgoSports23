import os
import zipfile
import random
import numpy as np
import pandas as pd

from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_squared_error

# =========================================================
# CONFIG
# =========================================================
DATA_DIR = "algosports23-predictions-2025"

BASE_ELO = 2000
ELO_K = 20
FORM_WINDOW = 5
EWM_ALPHA = 0.35

SEEDS = [42, 99, 2024]
DEPTHS = [5, 6, 8]

TOP_N = 10
OOF_SPLITS = 5
SIMS = 4000
TUNING_TRIALS = 15


# =========================================================
# LOAD
# =========================================================
train = pd.read_csv(f"{DATA_DIR}/Train.csv")
pred = pd.read_csv(f"{DATA_DIR}/Predictions.csv")
sample_rank = pd.read_excel(f"{DATA_DIR}/Sample_Rankings.xlsx")

# Make a copy of pred for writing predictions, to avoid overwriting with the loaded CSV later
pred_out = pred.copy()

train["margin"] = train["HomePts"] - train["AwayPts"]
train["win"] = (train["margin"] > 0).astype(int)

if "Date" in train.columns:
    train["Date"] = pd.to_datetime(train["Date"], errors="coerce")
    sort_cols = [c for c in ["Date", "GameID"] if c in train.columns]
    train = train.sort_values(sort_cols).reset_index(drop=True)

# =========================================================
# ELO
# =========================================================
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

elo = compute_elo(train)
teams = set(train["HomeTeam"]).union(set(train["AwayTeam"]))

# =========================================================
# TEAM FEATURES
# =========================================================
bayes = {}
for t in teams:
    home_mean = train.loc[train["HomeTeam"] == t, "margin"].mean()
    away_mean = train.loc[train["AwayTeam"] == t, "margin"].mean()
    bayes[t] = 0.6 * (0.0 if pd.isna(home_mean) else home_mean) - 0.4 * (0.0 if pd.isna(away_mean) else away_mean)

recent = {}
for t in teams:
    vals = train.loc[(train["HomeTeam"] == t) | (train["AwayTeam"] == t), "margin"].tail(5)
    recent[t] = 0.0 if len(vals) == 0 else float(vals.mean())

# =========================================================
# FEATURE ENGINEERING
# =========================================================
train["elo_diff"] = train["HomeTeam"].map(elo) - train["AwayTeam"].map(elo)
train["bayes_diff"] = train["HomeTeam"].map(bayes) - train["AwayTeam"].map(bayes)
train["recent_diff"] = train["HomeTeam"].map(recent) - train["AwayTeam"].map(recent)

train["home_form_raw"] = train.groupby("HomeTeam")["margin"].transform(
    lambda s: s.shift(1).rolling(FORM_WINDOW, min_periods=1).mean()
)
train["away_form_raw"] = train.groupby("AwayTeam")["margin"].transform(
    lambda s: (-s).shift(1).rolling(FORM_WINDOW, min_periods=1).mean()
)
train["form_diff"] = train["home_form_raw"] - train["away_form_raw"]

train["home_ewm_raw"] = train.groupby("HomeTeam")["margin"].transform(
    lambda s: s.shift(1).ewm(alpha=EWM_ALPHA, adjust=False).mean()
)
train["away_ewm_raw"] = train.groupby("AwayTeam")["margin"].transform(
    lambda s: (-s).shift(1).ewm(alpha=EWM_ALPHA, adjust=False).mean()
)
train["ewm_diff"] = train["home_ewm_raw"] - train["away_ewm_raw"]

train["momentum_diff"] = train["form_diff"] - train["ewm_diff"]
train["adj_form_diff"] = train["form_diff"] + 0.001 * train["elo_diff"]

train = train.fillna(0.0)

ALL_FEATURES = [
    "elo_diff",
    "bayes_diff",
    "recent_diff",
    "form_diff",
    "ewm_diff",
    "momentum_diff",
    "adj_form_diff",
]

X_full = train[ALL_FEATURES].copy()
y = train["margin"].copy()
y_win = train["win"].copy()

# =========================================================
# MARGIN LIKELIHOOD TARGET
# =========================================================
MARGIN_BINS = [-1e9, -20, -10, -5, -1, 1, 5, 10, 20, 1e9]
MARGIN_LABELS = [
    "lose_20plus",
    "lose_10_20",
    "lose_5_10",
    "lose_1_5",
    "close_tie",
    "win_1_5",
    "win_5_10",
    "win_10_20",
    "win_20plus",
]

train["margin_bucket"] = pd.cut(
    train["margin"],
    bins=MARGIN_BINS,
    labels=MARGIN_LABELS,
    include_lowest=True,
    right=False
)

bucket_to_idx = {label: i for i, label in enumerate(MARGIN_LABELS)}
idx_to_bucket = {i: label for label, i in bucket_to_idx.items()}

train["margin_bucket_idx"] = train["margin_bucket"].map(bucket_to_idx).astype(int)

bucket_midpoints = np.array([
    -25.0,
    -15.0,
    -7.5,
    -3.0,
    0.0,
    3.0,
    7.5,
    15.0,
    25.0
])

print("Bucket counts:")
print(train["margin_bucket_idx"].value_counts().sort_index())

# =========================================================
# HELPERS
# =========================================================
def expand_class_probs(raw_probs: np.ndarray, seen_classes, n_classes: int) -> np.ndarray:
    """
    Expand CatBoost probabilities back to the full class space.
    raw_probs: shape (n_rows, n_seen_classes) or (n_seen_classes,)
    seen_classes: model.classes_
    returns shape (n_rows, n_classes) or (n_classes,)
    """
    raw_probs = np.asarray(raw_probs)

    if raw_probs.ndim == 1:
        full = np.zeros(n_classes, dtype=float)
        for j, cls in enumerate(seen_classes):
            full[int(cls)] = raw_probs[j]
        return full

    full = np.zeros((raw_probs.shape[0], n_classes), dtype=float)
    for j, cls in enumerate(seen_classes):
        full[:, int(cls)] = raw_probs[:, j]
    return full

# =========================================================
# AUTO TUNING
# =========================================================
def sample_params():
    return {
        "iterations": random.choice([400, 1000]),
        "depth": random.choice([8, 10]),
        "learning_rate": random.choice([0.04, 0.05]),
        "l2_leaf_reg": random.choice([3, 5])
    }

def evaluate(params):
    tscv = TimeSeriesSplit(n_splits=3)
    oof = np.zeros(len(X_full))

    for tr, val in tscv.split(X_full):
        model = CatBoostRegressor(**params, verbose=False)
        model.fit(X_full.iloc[tr], y.iloc[tr])
        oof[val] = model.predict(X_full.iloc[val])

    rmse = np.sqrt(mean_squared_error(y, oof))
    acc = (np.sign(oof) == np.sign(y)).mean()
    return rmse + 4 * (1 - acc)

best_score = float("inf")
best_params = None

for _ in range(TUNING_TRIALS):
    p = sample_params()
    s = evaluate(p)
    if s < best_score:
        best_score = s
        best_params = p

print("BEST PARAMS:", best_params)

base_reg_params = best_params.copy()
base_reg_params.pop("depth", None)

# =========================================================
# OOF STACK
# =========================================================
tscv = TimeSeriesSplit(n_splits=OOF_SPLITS)

oof_mid = np.zeros(len(X_full))
oof_prob = np.zeros(len(X_full))
oof_sigma = np.zeros(len(X_full))
oof_bin = np.zeros((len(X_full), len(MARGIN_LABELS)))

for tr, val in tscv.split(X_full):
    X_tr, X_val = X_full.iloc[tr], X_full.iloc[val]
    y_tr = y.iloc[tr]
    y_win_tr = y_win.iloc[tr]
    y_bucket_tr = train["margin_bucket_idx"].iloc[tr]

    preds_mid, preds_low, preds_high = [], [], []

    for seed in SEEDS:
        for depth in DEPTHS:
            reg = CatBoostRegressor(
                **base_reg_params,
                depth=depth,
                random_seed=seed,
                verbose=False
            )
            reg.fit(X_tr, y_tr)

            low = CatBoostRegressor(
                loss_function="Quantile:alpha=0.2",
                iterations=400,
                depth=depth,
                learning_rate=base_reg_params.get("learning_rate", 0.04),
                l2_leaf_reg=base_reg_params.get("l2_leaf_reg", 3),
                random_seed=seed,
                verbose=False
            )
            high = CatBoostRegressor(
                loss_function="Quantile:alpha=0.8",
                iterations=400,
                depth=depth,
                learning_rate=base_reg_params.get("learning_rate", 0.04),
                l2_leaf_reg=base_reg_params.get("l2_leaf_reg", 3),
                random_seed=seed,
                verbose=False
            )

            low.fit(X_tr, y_tr)
            high.fit(X_tr, y_tr)

            preds_mid.append(reg.predict(X_val))
            preds_low.append(low.predict(X_val))
            preds_high.append(high.predict(X_val))

    mid = np.mean(preds_mid, axis=0)
    low = np.mean(preds_low, axis=0)
    high = np.mean(preds_high, axis=0)
    sigma = (high - low) / 1.28

    clf = CatBoostClassifier(iterations=400, verbose=False)
    clf.fit(X_tr, y_win_tr)
    prob = clf.predict_proba(X_val)[:, 1]

    bucket_model = CatBoostClassifier(iterations=400, verbose=False)
    bucket_model.fit(X_tr, y_bucket_tr)

    raw_bucket_probs = bucket_model.predict_proba(X_val)
    bucket_probs = expand_class_probs(
        raw_bucket_probs,
        bucket_model.classes_,
        len(MARGIN_LABELS)
    )

    oof_mid[val] = mid
    oof_prob[val] = prob
    oof_sigma[val] = sigma
    oof_bin[val] = bucket_probs

print("OOF complete")

# =========================================================
# CALIBRATION + META
# =========================================================
iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(oof_prob, y_win)
oof_prob_cal = iso.predict(oof_prob)

oof_exp_margin = oof_bin @ bucket_midpoints
oof_p_team1_win = oof_bin[:, 5:].sum(axis=1)
oof_p_team2_win = oof_bin[:, :4].sum(axis=1)
oof_p_close = oof_bin[:, 4]

meta_X = pd.DataFrame({
    "mid": oof_mid,
    "prob": oof_prob_cal,
    "sigma": oof_sigma,
    "elo": X_full["elo_diff"].values,
    "exp_margin": oof_exp_margin,
    "p_team1_win": oof_p_team1_win,
    "p_team2_win": oof_p_team2_win,
    "p_close": oof_p_close,
})

meta_scaler = StandardScaler()
meta_X_scaled = meta_scaler.fit_transform(meta_X)

meta = RidgeCV(alphas=np.logspace(-3, 3, 10))
meta.fit(meta_X_scaled, y)

# =========================================================
# FINAL TRAIN
# =========================================================
selector = CatBoostRegressor(iterations=400, verbose=False)
selector.fit(X_full, y)

importances = selector.get_feature_importance()
selected_cols = X_full.columns[np.argsort(importances)[::-1][:TOP_N]].tolist()

X = X_full[selected_cols].copy()

models_mid, models_low, models_high, models_bin = [], [], [], []

for seed in SEEDS:
    for depth in DEPTHS:
        reg = CatBoostRegressor(
            **base_reg_params,
            depth=depth,
            random_seed=seed,
            verbose=False
        )
        reg.fit(X, y)
        models_mid.append(reg)

        low = CatBoostRegressor(
            loss_function="Quantile:alpha=0.2",
            iterations=400,
            depth=depth,
            learning_rate=base_reg_params.get("learning_rate", 0.04),
            l2_leaf_reg=base_reg_params.get("l2_leaf_reg", 3),
            random_seed=seed,
            verbose=False
        )
        high = CatBoostRegressor(
            loss_function="Quantile:alpha=0.8",
            iterations=400,
            depth=depth,
            learning_rate=base_reg_params.get("learning_rate", 0.04),
            l2_leaf_reg=base_reg_params.get("l2_leaf_reg", 3),
            random_seed=seed,
            verbose=False
        )

        low.fit(X, y)
        high.fit(X, y)

        models_low.append(low)
        models_high.append(high)

        bucket_model = CatBoostClassifier(
            iterations=400,
            depth=depth,
            verbose=False,
            random_seed=seed
        )
        bucket_model.fit(X, train["margin_bucket_idx"])
        models_bin.append(bucket_model)

clf = CatBoostClassifier(iterations=400, verbose=False)
clf.fit(X, y_win)

# =========================================================
# PREDICT
# =========================================================
rng = np.random.default_rng(42)

def build_row(t1, t2):
    diff = recent.get(t1, 0.0) - recent.get(t2, 0.0)
    row = pd.DataFrame([{
        "elo_diff": elo.get(t1, BASE_ELO) - elo.get(t2, BASE_ELO),
        "bayes_diff": bayes.get(t1, 0.0) - bayes.get(t2, 0.0),
        "recent_diff": diff,
        "form_diff": diff,
        "ewm_diff": diff,
        "momentum_diff": 0.0,
        "adj_form_diff": diff
    }])
    return row[selected_cols]

def predict_game(t1, t2):
    Xp = build_row(t1, t2)

    mid = np.mean([m.predict(Xp)[0] for m in models_mid])
    low = np.mean([m.predict(Xp)[0] for m in models_low])
    high = np.mean([m.predict(Xp)[0] for m in models_high])

    raw_prob = clf.predict_proba(Xp)[:, 1][0]
    prob = iso.predict([raw_prob])[0]

    all_bucket_probs = []
    for model in models_bin:
        raw_probs = model.predict_proba(Xp)[0]
        full_probs = expand_class_probs(
            raw_probs,
            model.classes_,
            len(MARGIN_LABELS)
        )
        all_bucket_probs.append(full_probs)

    bucket_probs = np.mean(all_bucket_probs, axis=0)
    exp_margin = float(bucket_probs @ bucket_midpoints)

    p_team1_win = float(bucket_probs[5:].sum())
    p_team2_win = float(bucket_probs[:4].sum())
    p_close = float(bucket_probs[4])

    sigma = float(np.clip((high - low) / 1.28, 3, 25))

    meta_row = pd.DataFrame([{
        "mid": mid,
        "prob": prob,
        "sigma": sigma,
        "elo": Xp["elo_diff"].iloc[0],
        "exp_margin": exp_margin,
        "p_team1_win": p_team1_win,
        "p_team2_win": p_team2_win,
        "p_close": p_close,
    }])

    mu = meta.predict(meta_scaler.transform(meta_row))[0]
    mu = 0.8 * mu + 0.2 * exp_margin

    sims = rng.normal(mu, sigma, SIMS)
    final_margin = float(np.median(sims))

    return {
        "margin": final_margin,
        "prob_team1_win": p_team1_win,
        "prob_team2_win": p_team2_win,
        "prob_close": p_close,
        "expected_margin_from_bins": exp_margin,
        "sigma": sigma,
        "bucket_probs": bucket_probs
    }

# =========================================================
# OUTPUT
# =========================================================

rows = []
margins = []

for _, r in pred.iterrows():
    result = predict_game(r["Team1"], r["Team2"])
    rounded_margin = int(round(result["margin"]))
    margins.append(rounded_margin)

    out_row = {
        "GameID": r["GameID"],
        "Predicted_Margin": result["margin"],
        "P_Team1_Win": result["prob_team1_win"],
        "P_Team2_Win": result["prob_team2_win"],
        "P_Close_or_Tie": result["prob_close"],
        "Expected_Margin_From_Bins": result["expected_margin_from_bins"],
        "Predicted_Sigma": result["sigma"],
    }

    for i, label in enumerate(MARGIN_LABELS):
        out_row[f"P_{label}"] = result["bucket_probs"][i]

    rows.append(out_row)

pred_out["Team1_WinMargin"] = margins
pred_out.to_csv(f"{DATA_DIR}/Predictions.csv", index=False)

likelihood_df = pd.DataFrame(rows)
likelihood_df.to_csv(f"{DATA_DIR}/Prediction_Likelihoods.csv", index=False)

# =========================================================
# RANKINGS
# =========================================================
strength_rank = {
    t: 0.5 * bayes[t] + 0.3 * ((elo[t] - BASE_ELO) / 100.0) + 0.2 * recent[t]
    for t in teams
}

ranking = pd.DataFrame({
    "Team": list(strength_rank.keys()),
    "Strength": list(strength_rank.values())
}).sort_values("Strength", ascending=False)

ranking["Rank"] = np.arange(1, len(ranking) + 1)
ranking_final = sample_rank.merge(ranking, on="Team")
ranking_final.to_excel(f"{DATA_DIR}/Rankings.xlsx", index=False)

# =========================================================
# ZIP
# =========================================================
with zipfile.ZipFile(f"{DATA_DIR}/Submission.zip", "w") as z:
    z.write(f"{DATA_DIR}/Predictions.csv", "Predictions.csv")
    z.write(f"{DATA_DIR}/Rankings.xlsx", "Rankings.xlsx")

print("🏆 FINAL CHAMPIONSHIP WITH LIKELIHOOD COMPLETE")

# --- Evaluation: RMSE and Winning Team Accuracy ---
try:
    # Use the correct variable names for loaded prediction files
    sample_pred = pd.read_csv(f"{DATA_DIR}/Sample_Predictions.csv")
    pred = pd.read_csv(f"{DATA_DIR}/Predictions.csv")
    actual_margins = sample_pred['Team1_WinMargin'].astype(int).values
    predicted_margins = pred['Team1_WinMargin'].astype(int).values
    n_games = min(len(actual_margins), len(predicted_margins))
    if n_games > 0:
        rmse = np.sqrt(np.mean((predicted_margins[:n_games] - actual_margins[:n_games]) ** 2))
        print(f"RMSE (Predicted vs Actual Margins): {rmse:.2f}")
        actual_winners = np.sign(actual_margins[:n_games])
        predicted_winners = np.sign(predicted_margins[:n_games])
        correct_winner = (actual_winners == predicted_winners).sum()
        accuracy = correct_winner / n_games * 100
        print(f"Winning Team Prediction Accuracy: {accuracy:.2f}% ({correct_winner}/{n_games})")
    else:
        print("Warning: Cannot evaluate RMSE or accuracy, margin counts do not match or are empty.")
except Exception as e:
    print(f"Evaluation error: {e}")