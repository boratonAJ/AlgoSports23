import os
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from lightgbm import LGBMClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.model_selection import KFold

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

# Model variant: "complex" (Ridge + LGBM) or "simple" (Ridge only)
MODEL_VARIANT = "simple"  # Change to "complex" to use Ridge + LGBM ensemble

BASE_ELO = 2000 # Starting ELO rating for all teams, can be tuned based on historical data or set to 1500 as a common baseline
ELO_K = 30 # ELO update factor which can be tuned based on historical performance
FORM_WINDOW = 5 # The number of recent games to consider for form calculation
EWM_ALPHA = 1 # Smoothing factor for EWM, higher means more weight on recent games
N_SIM = 3000 # Number of simulations to run for likelihood estimation, can be increased for more accuracy but will take longer
RANDOM_STATE = 42 # For reproducibility
RIDGE_ALPHA_GRID = list(range(50, 201, 10))

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
def compute_elo(df, k=15, base_elo=1500): # k is the update factor, base_elo is the starting rating for all teams
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

# =========================
# DOMAIN-SPECIFIC SPORTS FEATURES
# =========================
# Points-based features
train_df["psg_pag_diff"] = 0.0  # Will be computed per team
train_df["home_away_balance"] = 0.0

team_psg_pag = {}
team_home_adv = {}

for team in teams:
    home_games = train_df[train_df["HomeTeam"] == team]
    away_games = train_df[train_df["AwayTeam"] == team]
    
    # Points scored/allowed per game
    home_psg = home_games["HomePts"].sum() / max(len(home_games), 1)
    home_pag = home_games["AwayPts"].sum() / max(len(home_games), 1)
    away_psg = away_games["AwayPts"].sum() / max(len(away_games), 1)
    away_pag = away_games["HomePts"].sum() / max(len(away_games), 1)
    
    psg_overall = (home_psg + away_psg) / 2
    pag_overall = (home_pag + away_pag) / 2
    team_psg_pag[team] = psg_overall - pag_overall
    
    train_df.loc[train_df["HomeTeam"] == team, "psg_pag_diff"] = psg_overall - pag_overall
    train_df.loc[train_df["AwayTeam"] == team, "psg_pag_diff"] = psg_overall - pag_overall
    
    # Home court advantage indicator
    home_adv = (home_psg - away_psg) if max(len(home_games), len(away_games)) > 0 else 0
    team_home_adv[team] = home_adv
    train_df.loc[train_df["HomeTeam"] == team, "home_away_balance"] = home_adv
    train_df.loc[train_df["AwayTeam"] == team, "home_away_balance"] = -home_adv

# Drop unused columns
drop_cols = ["Date", "GameID"]

train_df = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
feature_cols = [
    "elo_diff",
    "bayes_diff",
    "recent_diff",
    "form_diff",
    "ewm_diff",
    "vol_diff",
    "elo_x_form",
    "elo_x_bayes",
    "psg_pag_diff",        # Domain: Points scored vs allowed differential
    "home_away_balance"    # Domain: Home court advantage
]
X = train_df[feature_cols]
y = train_df["margin"]


# =========================
# TRAIN MODELS
# =========================
# Hyperparameter sweep for Ridge on OOF MAE
alpha_scores = []
for alpha in RIDGE_ALPHA_GRID:
    ridge_oof_alpha = np.zeros(len(X))
    kf_alpha = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for train_idx, val_idx in kf_alpha.split(X):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold = y.iloc[train_idx]
        ridge_fold_alpha = Ridge(alpha=alpha, random_state=RANDOM_STATE)
        ridge_fold_alpha.fit(X_train_fold, y_train_fold)
        ridge_oof_alpha[val_idx] = ridge_fold_alpha.predict(X_val_fold)
    alpha_mae = mean_absolute_error(y, ridge_oof_alpha)
    alpha_scores.append((alpha, alpha_mae))

best_ridge_alpha, best_ridge_oof_mae = min(alpha_scores, key=lambda t: t[1])
print(f"Best Ridge alpha from sweep: {best_ridge_alpha} (OOF MAE={best_ridge_oof_mae:.4f})")

# Final base models trained on full data
ridge = Ridge(alpha=best_ridge_alpha, random_state=RANDOM_STATE)
lgreg = LGBMRegressor(
    objective="regression",
    n_estimators=400,
    learning_rate=0.15,
    min_child_samples=30,
    max_depth=5,
    num_leaves=16,
    subsample=0.7,
    colsample_bytree=0.7,
    random_state=RANDOM_STATE
)
cat = CatBoostRegressor(
    iterations=350,
    depth=5,
    learning_rate=0.06,
    loss_function="RMSE",
    l2_leaf_reg=10,
    random_seed=RANDOM_STATE,
    verbose=False
)

ridge.fit(X, y)
lgreg.fit(X, y)
cat.fit(X, y)

# --- Margin bin classifier ---
margin_bins = [-np.inf, -10, -3, 3, 10, np.inf]
margin_labels = [0, 1, 2, 3, 4]  # 0: BigLoss, 1: SmallLoss, 2: Draw, 3: SmallWin, 4: BigWin
train_df["margin_bin"] = pd.cut(y, bins=margin_bins, labels=margin_labels)
# clf = CatBoostClassifier(
#     iterations=400,
#     depth=6,
#     learning_rate=0.4,
#     loss_function="MultiClass",
#     verbose=False,
#     random_seed=RANDOM_STATE
# )

clf = LGBMClassifier(objective="multiclass", num_class=5)


clf.fit(X, train_df["margin_bin"])

print("Models trained (Ridge, LGBM, CatBoost, classifier)")

# =========================
# K-FOLD OUT-OF-FOLD EVALUATION
# =========================
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
oof_ridge = np.zeros(len(X))
oof_lgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))
oof_meta = np.zeros(len(X))

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold = y.iloc[train_idx]

    ridge_fold = Ridge(alpha=best_ridge_alpha, random_state=RANDOM_STATE)
    lgb_fold = LGBMRegressor(
        objective="regression",
        n_estimators=400,
        learning_rate=0.15,
        min_child_samples=30,
        max_depth=5,
        num_leaves=16,
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=RANDOM_STATE
    )
    cat_fold = CatBoostRegressor(
        iterations=350,
        depth=5,
        learning_rate=0.06,
        loss_function="RMSE",
        l2_leaf_reg=10,
        random_seed=RANDOM_STATE,
        verbose=False
    )

    ridge_fold.fit(X_train_fold, y_train_fold)
    lgb_fold.fit(X_train_fold, y_train_fold)
    cat_fold.fit(X_train_fold, y_train_fold)

    oof_ridge[val_idx] = ridge_fold.predict(X_val_fold)
    oof_lgb[val_idx] = lgb_fold.predict(X_val_fold)
    oof_cat[val_idx] = cat_fold.predict(X_val_fold)

    meta_train_X = pd.DataFrame({
        "ridge": oof_ridge[train_idx],
        "lgb": oof_lgb[train_idx],
        "cat": oof_cat[train_idx],
    })
    meta_val_X = pd.DataFrame({
        "ridge": oof_ridge[val_idx],
        "lgb": oof_lgb[val_idx],
        "cat": oof_cat[val_idx],
    })
    meta_fold = Ridge(alpha=10, random_state=RANDOM_STATE)
    meta_fold.fit(meta_train_X, y.iloc[train_idx])
    oof_meta[val_idx] = meta_fold.predict(meta_val_X)

    print(f"Fold {fold + 1}/5 OOF predictions collected")

ridge_oof_mae = mean_absolute_error(y, oof_ridge)
ridge_oof_acc = accuracy_score(np.sign(y), np.sign(oof_ridge))
meta_oof_mae = mean_absolute_error(y, oof_meta)
meta_oof_acc = accuracy_score(np.sign(y), np.sign(oof_meta))

meta_oof_features = pd.DataFrame({"ridge": oof_ridge, "lgb": oof_lgb, "cat": oof_cat})
meta_model = Ridge(alpha=10, random_state=RANDOM_STATE)
meta_model.fit(meta_oof_features, y)

print("OOF Meta-ensemble trained on Ridge+LGBM+CatBoost OOF predictions")
print(f"Ridge-only OOF MAE: {ridge_oof_mae:.4f} | OOF Accuracy: {ridge_oof_acc:.2%}")
print(f"Meta-ensemble OOF MAE: {meta_oof_mae:.4f} | OOF Accuracy: {meta_oof_acc:.2%}")

selected_model = "meta" if meta_oof_mae < ridge_oof_mae else "ridge"
oof_pred = oof_meta if selected_model == "meta" else oof_ridge
oof_mae = meta_oof_mae if selected_model == "meta" else ridge_oof_mae
oof_acc = meta_oof_acc if selected_model == "meta" else ridge_oof_acc
print(f"Selected model by OOF MAE: {selected_model}")
print(f"OOF MAE: {oof_mae:.4f} | OOF Accuracy: {oof_acc:.2%}")

# =========================
# SIGMA & METRICS
# =========================

ridge_train_pred = ridge.predict(X)
lgb_train_pred = lgreg.predict(X)
cat_train_pred = cat.predict(X)

meta_train_features = pd.DataFrame({
    "ridge": ridge_train_pred,
    "lgb": lgb_train_pred,
    "cat": cat_train_pred,
})
pred_train_meta = meta_model.predict(meta_train_features)
pred_train = pred_train_meta if selected_model == "meta" else ridge_train_pred
sigma = np.std(y - pred_train)
mae = mean_absolute_error(y, pred_train)
acc = accuracy_score(np.sign(y), np.sign(pred_train))
print(
    "Meta model coefficients "
    f"(ridge/lgb/cat): {meta_model.coef_[0]:.3f}, {meta_model.coef_[1]:.3f}, {meta_model.coef_[2]:.3f}"
)
print(f"Train metric model: {selected_model}")
print(f"Train Sigma: {sigma:.4f} | Train MAE: {mae:.4f} | Train Accuracy: {acc:.2%}")

# =========================
# VALIDATION DIAGNOSTICS CHART
# =========================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: Train vs OOF MAE
axes[0].bar(["Train", "OOF"], [mae, oof_mae], color=["blue", "orange"])
axes[0].set_ylabel("Mean Absolute Error")
axes[0].set_title("Margin MAE: Train vs OOF")
for i, v in enumerate([mae, oof_mae]):
    axes[0].text(i, v + 0.005, f"{v:.4f}", ha="center", va="bottom", fontweight="bold")

# Right: Train vs OOF Accuracy
axes[1].bar(["Train", "OOF"], [acc, oof_acc], color=["green", "red"])
axes[1].set_ylabel("Sign Accuracy")
axes[1].set_title("Win/Loss Accuracy: Train vs OOF")
axes[1].set_ylim([0, 1.0])
for i, v in enumerate([acc, oof_acc]):
    axes[1].text(i, v + 0.02, f"{v:.2%}", ha="center", va="bottom", fontweight="bold")

# Overfitting indicator
overfitting_gap_mae = mae - oof_mae
overfitting_gap_acc = acc - oof_acc
fig.suptitle(f"Generalization Check: MAE Gap={overfitting_gap_mae:.4f}, Acc Gap={overfitting_gap_acc:.2%}", 
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{DATA_DIR}/Validation_Diagnostics.png", dpi=100, bbox_inches="tight")
plt.show()
print(f"Validation diagnostics saved: {DATA_DIR}/Validation_Diagnostics.png")
if overfitting_gap_mae > 0.10 or overfitting_gap_acc > 0.05:
    print("⚠  WARNING: Significant overfitting detected. Consider regularization.")
else:
    print("✓ Model generalization looks good.")

# =========================
# OVERFITTING/UNDERFITTING DIAGNOSTIC PLOT
# =========================
plt.figure(figsize=(7, 7))
plt.scatter(y, pred_train, alpha=0.4, label="Train predictions")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label="Ideal fit")
plt.xlabel("True Margin (y)")
plt.ylabel("Predicted Margin (pred_train)")
plt.title("Train: True vs Predicted Margin")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

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
    psg_pag_diff = team_psg_pag.get(t1, 0.0) - team_psg_pag.get(t2, 0.0)
    home_away_balance = team_home_adv.get(t1, 0.0) - team_home_adv.get(t2, 0.0)

    return pd.DataFrame([{
        "elo_diff": elo_diff,
        "bayes_diff": bayes_diff,
        "recent_diff": recent_diff,
        "form_diff": form_diff,
        "ewm_diff": ewm_diff,
        "vol_diff": vol_diff,
        "elo_x_form": elo_diff * form_diff,
        "elo_x_bayes": elo_diff * bayes_diff,
        "psg_pag_diff": psg_pag_diff,
        "home_away_balance": home_away_balance
    }])

def predict_margin_and_likelihood(t1, t2):
    row = build_row(t1, t2)
    ridge_pred = ridge.predict(row)[0]
    if selected_model == "meta":
        row_preds = pd.DataFrame([{
            "ridge": ridge_pred,
            "lgb": lgreg.predict(row)[0],
            "cat": cat.predict(row)[0],
        }])
        ml = meta_model.predict(row_preds)[0]
    else:
        ml = ridge_pred
    strength = final_strength.get(t1, 0) - final_strength.get(t2, 0)
    mu = 0.5 * strength + 0.3 * ml + 0.2 * (row["elo_diff"].iloc[0] / 25)
    sims = rng.normal(mu, sigma, N_SIM)
    # Median margin prediction
    median_margin = np.median(sims)
    # --- Classifier bucket probabilities ---
    bucket_probs = clf.predict_proba(row)[0]
    # Map to labels
    labels = ["BigLoss", "SmallLoss", "Draw", "SmallWin", "BigWin"]
    likelihood_dict = dict(zip(labels, bucket_probs))
    # Derived probabilities
    p_team1_win = bucket_probs[3] + bucket_probs[4]
    p_team2_win = bucket_probs[0] + bucket_probs[1]
    p_close = bucket_probs[1] + bucket_probs[2] + bucket_probs[3]
    expected_margin = np.dot(bucket_probs, [-20, -6, 0, 6, 20])
    likelihood_dict["P_Team1Win"] = p_team1_win
    likelihood_dict["P_Team2Win"] = p_team2_win
    likelihood_dict["P_Close"] = p_close
    likelihood_dict["ExpectedMargin"] = expected_margin
    return median_margin, likelihood_dict

# =========================
# PREDICTIONS
# =========================

## Add win margin and likelihood columns (including derived) for meta-modeling, but do not save them in the final output
likelihood_cols = ["BigLoss", "SmallLoss", "Draw", "SmallWin", "BigWin", "P_Team1Win", "P_Team2Win", "P_Close", "ExpectedMargin"]
def predict_row(r):
    margin, likelihoods = predict_margin_and_likelihood(r["Team1"], r["Team2"])
    return pd.Series([margin] + [likelihoods.get(col, 0.0) for col in likelihood_cols])

meta_features = pred_df.apply(predict_row, axis=1)
pred_df["Team1_WinMargin"] = meta_features.iloc[:, 0]
for i, col in enumerate(likelihood_cols):
    pred_df[col] = meta_features.iloc[:, i + 1]

# calibration
pred_df["Team1_WinMargin"] *= 1.45 # scale factor to adjust for regression to mean. The 1.35 is derived from historical performance on validation data, can be tuned further if needed
pred_df["Team1_WinMargin"] = np.tanh(pred_df["Team1_WinMargin"] / 18) * 35
pred_df["Team1_WinMargin"] -= pred_df["Team1_WinMargin"].mean()
pred_df["Team1_WinMargin"] = pred_df["Team1_WinMargin"].round().astype(int)

# Add match confidence column (max win probability)
pred_df["Confidence"] = pred_df[["P_Team1Win", "P_Team2Win"]].max(axis=1)
pred_df["Confidence"] = pred_df["Confidence"].round(3)

# Sort matches by confidence (descending)
pred_df_sorted = pred_df.sort_values("Confidence", ascending=False).reset_index(drop=True)

# Save ranked matches by confidence to a separate CSV
pred_df_sorted[["GameID", "Team1", "Team2", "Team1_WinMargin", "Confidence"]].to_csv(f"{DATA_DIR}/Match_Confidence_Ranking.csv", index=False)
print("Match confidence ranking saved as Match_Confidence_Ranking.csv")

# =========================
# POST-PROCESSING: EV, KELLY, PROFIT SIMULATION
# =========================
# This block assumes you have columns: GameID, Team1, Team2, P_Team1Win, P_Team2Win, Odds_Team1, Odds_Team2
# If you have market odds in a separate file, load and merge here. For demo, we use dummy odds (2.0 for both teams)
post_df = pred_df.copy()
if "Odds_Team1" not in post_df.columns or "Odds_Team2" not in post_df.columns:
    post_df["Odds_Team1"] = 2.0
    post_df["Odds_Team2"] = 2.0

# Expected Value (EV) for each team
post_df["EV_Team1"] = post_df["P_Team1Win"] * (post_df["Odds_Team1"] - 1) - (1 - post_df["P_Team1Win"])
post_df["EV_Team2"] = post_df["P_Team2Win"] * (post_df["Odds_Team2"] - 1) - (1 - post_df["P_Team2Win"])

# Kelly Criterion (fraction of bankroll to bet)
# def kelly(p, b):
#     # p: probability of winning, b: decimal odds - 1
#     edge = p * (b + 1) - 1
#     if b <= 0 or p <= 0 or p >= 1:
#         return 0.0
#     f = (p * b - (1 - p)) / b # Kelly formula
#     return max(0.0, f)
def kelly(p, b, fraction=0.5, cap=0.25):
    # p: model win probability in [0,1]
    # b: net odds = decimal_odds - 1
    if b <= 0:
        return 0.0

    p = float(np.clip(p, 0.0, 1.0))
    q = 1.0 - p

    # Full Kelly
    f_full = (b * p - q) / b

    # No negative bets, then apply fractional Kelly and cap
    f = max(0.0, f_full) * fraction
    return min(f, cap)

post_df["Kelly_Team1"] = post_df.apply(lambda r: kelly(r["P_Team1Win"], r["Odds_Team1"] - 1), axis=1)
post_df["Kelly_Team2"] = post_df.apply(lambda r: kelly(r["P_Team2Win"], r["Odds_Team2"] - 1), axis=1)

# Market-aware decision: pick side with max EV, only if EV > 0
def pick_bet(row):
    if row["EV_Team1"] > 0 and row["EV_Team1"] >= row["EV_Team2"]:
        return "Team1"
    elif row["EV_Team2"] > 0 and row["EV_Team2"] > row["EV_Team1"]:
        return "Team2"
    else:
        return "No Bet"
post_df["Bet"] = post_df.apply(pick_bet, axis=1)

# Bet size: use Kelly fraction for chosen side, else 0
def bet_size(row):
    if row["Bet"] == "Team1":
        return row["Kelly_Team1"]
    elif row["Bet"] == "Team2":
        return row["Kelly_Team2"]
    else:
        return 0.0
post_df["Bet_Size"] = post_df.apply(bet_size, axis=1)

# Simulate profit for each bet (assume 1 unit bankroll per bet)
def profit(row):
    if row["Bet"] == "Team1":
        return row["P_Team1Win"] * (row["Odds_Team1"] - 1) - (1 - row["P_Team1Win"])
    elif row["Bet"] == "Team2":
        return row["P_Team2Win"] * (row["Odds_Team2"] - 1) - (1 - row["P_Team2Win"])
    else:
        return 0.0
post_df["EV_Max"] = post_df.apply(profit, axis=1)

# Sort by EV_Max (profit-based ranking)
post_df = post_df.sort_values("EV_Max", ascending=False).reset_index(drop=True)

# Save to CSV
ev_cols = [
    "GameID", "Team1", "Team2", "Team1_WinMargin", "P_Team1Win", "P_Team2Win", "Odds_Team1", "Odds_Team2",
    "EV_Team1", "EV_Team2", "Bet", "Bet_Size", "EV_Max"
]
post_df[ev_cols].to_csv(f"{DATA_DIR}/EV_Ranking.csv", index=False)
print("EV/profit-based ranking saved as EV_Ranking.csv")

# =========================
# SAVE
# =========================


# Save only GameID and Team1_WinMargin in the final Predictions.csv
submission = pred_df[["GameID", "Team1_WinMargin"]]
template = sample_pred.drop(columns=["Team1_WinMargin"])
submission = template.merge(submission, on="GameID")
submission.to_csv(OUTPUT_PRED_PATH, index=False)

print("Predictions saved")

# =========================
# RANKINGS
# =========================
# ranking_df = pd.DataFrame({
#     "Team": list(final_strength.keys()),
#     "Strength": list(final_strength.values())
# }).sort_values("Strength", ascending=False)

# ranking_df["Rank"] = range(1, len(ranking_df) + 1)

# final_rankings = sample_rank[["TeamID", "Team"]].merge(
#     ranking_df[["Team", "Rank"]],
#     on="Team",
#     how="left"
# )

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

    predictive_score = 0.80 * win_pct + 0.20 * pythag_win

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

print("Submission.zip created")