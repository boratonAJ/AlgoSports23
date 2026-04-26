# Refactored pipeline based on rpl_championship_pipeline_latest_v.py, using LGBMRegressor instead of CatBoostRegressor
import os
import zipfile
from dataclasses import dataclass
from typing import Dict, Tuple, Set, List
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

# =========================
# CONFIG
# =========================
@dataclass
class Config:
    data_dir: str = "algosports23-predictions-2025"
    train_file: str = "Train.csv"
    predictions_file: str = "Predictions.csv"
    sample_predictions_file: str = "Sample_Predictions.csv"
    sample_rankings_file: str = "Sample_Rankings.xlsx"

    output_predictions_file: str = "Predictions.csv"
    output_rankings_file: str = "Rankings.xlsx"
    output_zip_file: str = "Submission.zip"

    random_state: int = 42
    n_simulations: int = 2000

    elo_k: float = 20.0
    base_elo: float = 2000.0

    calibration_scale: float = 1.25
    tanh_divisor: float = 18.0
    tanh_multiplier: float = 35.0

    form_window: int = 5
    ewm_alpha: float = 0.35
    trend_window: int = 5

    cv_splits: int = 20

    lgbm_n_estimators: int = 1000
    lgbm_max_depth: int = 8
    lgbm_learning_rate: float = 0.05
    lgbm_reg_lambda: float = 3.0


def path_join(*parts: str) -> str:
    return os.path.join(*parts)

def load_data(cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(path_join(cfg.data_dir, cfg.train_file))
    pred_df = pd.read_csv(path_join(cfg.data_dir, cfg.predictions_file))
    sample_pred_df = pd.read_csv(path_join(cfg.data_dir, cfg.sample_predictions_file))
    sample_rank_df = pd.read_excel(path_join(cfg.data_dir, cfg.sample_rankings_file))
    print("Data loaded successfully")
    return train_df, pred_df, sample_pred_df, sample_rank_df

def preprocess_train(train_df: pd.DataFrame) -> pd.DataFrame:
    df = train_df.copy()
    df["margin"] = df["HomePts"] - df["AwayPts"]
    sort_cols = [c for c in ["Date", "GameID"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    return df

def compute_elo(df: pd.DataFrame, k: float = 20.0, base_elo: float = 2000.0) -> Dict[str, float]:
    teams = pd.unique(df[["HomeTeam", "AwayTeam"]].values.ravel())
    elo = {team: base_elo for team in teams}
    for _, row in df.iterrows():
        home_team = row["HomeTeam"]
        away_team = row["AwayTeam"]
        margin = row["margin"]
        rating_home = elo[home_team]
        rating_away = elo[away_team]
        expected_home = 1.0 / (1.0 + 10 ** ((rating_away - rating_home) / 400.0))
        actual_home = 1.0 if margin > 0 else 0.5 if margin == 0 else 0.0
        margin_factor = np.log(abs(margin) + 1.0)
        update = k * margin_factor * (actual_home - expected_home)
        elo[home_team] += update
        elo[away_team] -= update
    return elo

def compute_bayesian_style_strength(df: pd.DataFrame) -> Dict[str, float]:
    home_strength = df.groupby("HomeTeam")['margin'].mean()
    away_strength = df.groupby("AwayTeam")['margin'].mean()
    teams = set(home_strength.index).union(set(away_strength.index))
    bayes_strength: Dict[str, float] = {}
    for team in teams:
        home_val = float(home_strength.get(team, 0.0))
        away_val = float(-away_strength.get(team, 0.0))
        bayes_strength[team] = 0.6 * home_val + 0.4 * away_val
    return bayes_strength

def compute_recent_form(df: pd.DataFrame, teams: Set[str], n_games: int = 5) -> Dict[str, float]:
    home = df[["HomeTeam", "margin"]].copy()
    home.columns = ["Team", "tm"]
    away = df[["AwayTeam", "margin"]].copy()
    away.columns = ["Team", "tm"]
    away["tm"] = -away["tm"]
    all_games = pd.concat([home, away])
    recent = {}
    for t in teams:
        vals = all_games.loc[all_games["Team"] == t, "tm"].tail(n_games)
        recent[t] = vals.mean() if len(vals) > 0 else 0.0
    return recent

def compute_final_strength(teams: Set[str], elo: Dict[str, float], bayes_strength: Dict[str, float], recent_form: Dict[str, float], base_elo: float = 2000.0) -> Dict[str, float]:
    final_strength: Dict[str, float] = {}
    for team in teams:
        elo_norm = (elo.get(team, base_elo) - base_elo) / 100.0
        final_strength[team] = (
            0.5 * bayes_strength.get(team, 0.0)
            + 0.3 * elo_norm
            + 0.2 * recent_form.get(team, 0.0)
        )
    return final_strength

def add_training_features(df: pd.DataFrame, elo: Dict[str, float], form_window: int = 5, ewm_alpha: float = 0.35) -> pd.DataFrame:
    out = df.copy()
    out["elo_home"] = out["HomeTeam"].map(elo).fillna(2000.0)
    out["elo_away"] = out["AwayTeam"].map(elo).fillna(2000.0)
    out["elo_diff"] = out["elo_home"] - out["elo_away"]
    out["form_home"] = (
        out.groupby("HomeTeam")["margin"]
        .transform(lambda s: s.shift(1).rolling(form_window, min_periods=1).mean())
        .fillna(0.0)
    )
    out["away_team_margin"] = -out["margin"]
    out["form_away"] = (
        out.groupby("AwayTeam")["away_team_margin"]
        .transform(lambda s: s.shift(1).rolling(form_window, min_periods=1).mean())
        .fillna(0.0)
    )
    out["form_diff"] = out["form_home"] - out["form_away"]
    out["ewm_home"] = (
        out.groupby("HomeTeam")["margin"]
        .transform(lambda s: s.shift(1).ewm(alpha=ewm_alpha).mean())
        .fillna(0.0)
    )
    out["ewm_away"] = (
        out.groupby("AwayTeam")["away_team_margin"]
        .transform(lambda s: s.shift(1).ewm(alpha=ewm_alpha).mean())
        .fillna(0.0)
    )
    out["ewm_diff"] = out["ewm_home"] - out["ewm_away"]
    out["elo_x_form"] = out["elo_diff"] * out["form_diff"]
    out["elo_x_bayes"] = out["elo_diff"] * out["ewm_diff"]
    out = out.fillna(0.0)
    return out

def build_model_inputs(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    feature_cols = [
        "elo_diff",
        "form_diff",
        "ewm_diff",
        "elo_x_form",
        "elo_x_bayes"
    ]
    X = df[feature_cols].copy()
    y = df["margin"].copy()
    return X, y, feature_cols


# --- Define build_base_models (was missing) ---
def build_base_models(X: pd.DataFrame, y: pd.Series, cfg: Config):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_scaled, y)
    lgbm_model = LGBMRegressor(
        n_estimators=cfg.lgbm_n_estimators,
        max_depth=cfg.lgbm_max_depth,
        learning_rate=cfg.lgbm_learning_rate,
        reg_lambda=cfg.lgbm_reg_lambda,
        random_state=cfg.random_state,
        verbose=-1,
    )
    lgbm_model.fit(X, y)
    catboost_model = CatBoostRegressor(
        iterations=cfg.lgbm_n_estimators,
        depth=cfg.lgbm_max_depth,
        learning_rate=cfg.lgbm_learning_rate,
        l2_leaf_reg=cfg.lgbm_reg_lambda,
        random_seed=cfg.random_state,
        verbose=0
    )
    catboost_model.fit(X, y)
    return ridge_model, lgbm_model, catboost_model, scaler

def estimate_sigma(X: pd.DataFrame, y: pd.Series, ridge_model, lgbm_model, catboost_model, scaler):
    X_scaled = scaler.transform(X)
    preds = np.vstack([
        lgbm_model.predict(X),
        catboost_model.predict(X),
        ridge_model.predict(X_scaled)
    ])
    pred_train_mean = np.mean(preds, axis=0)
    pred_train_median = np.median(preds, axis=0)
    sigma_mean = float(np.std(y - pred_train_mean))
    sigma_median = float(np.std(y - pred_train_median))
    sigma = min(sigma_mean, sigma_median)
    print(f"Sigma (mean): {sigma_mean:.4f}, Sigma (median): {sigma_median:.4f}, Using: {sigma:.4f}")
    return sigma

def build_row(t1, t2, elo, bayes_strength, recent_form, cfg: Config):
    elo_diff = elo.get(t1, cfg.base_elo) - elo.get(t2, cfg.base_elo)
    form_diff = recent_form.get(t1, 0) - recent_form.get(t2, 0)
    ewm_diff = form_diff
    return pd.DataFrame([{
        "elo_diff": elo_diff,
        "form_diff": form_diff,
        "ewm_diff": ewm_diff,
        "elo_x_form": elo_diff * form_diff,
        "elo_x_bayes": elo_diff * ewm_diff
    }])

def predict_margin(t1, t2, ridge_model, lgbm_model, catboost_model, scaler, final_strength, elo, bayes_strength, recent_form, sigma, rng, cfg: Config):
    row = build_row(t1, t2, elo, bayes_strength, recent_form, cfg)
    row_scaled = scaler.transform(row)
    preds = [
        lgbm_model.predict(row)[0],
        catboost_model.predict(row)[0],
        ridge_model.predict(row_scaled)[0]
    ]
    ml_mean = float(np.mean(preds))
    ml_median = float(np.median(preds))
    # Use median for robustness
    ml = ml_median
    strength = final_strength.get(t1, 0) - final_strength.get(t2, 0)
    mu = 0.5 * strength + 0.3 * ml + 0.2 * (row["elo_diff"].iloc[0] / 25)
    sims = rng.normal(mu, sigma, cfg.n_simulations)
    # Use median of simulations for final prediction
    return float(np.median(sims))

def main():
    cfg = Config()
    train_df, pred_df, sample_pred_df, sample_rank_df = load_data(cfg)
    train_df = preprocess_train(train_df)
    elo = compute_elo(train_df, k=cfg.elo_k, base_elo=cfg.base_elo)
    bayes_strength = compute_bayesian_style_strength(train_df)
    teams = set(train_df["HomeTeam"]).union(set(train_df["AwayTeam"]))
    recent_form = compute_recent_form(train_df, teams=teams, n_games=cfg.form_window)
    final_strength = compute_final_strength(teams, elo, bayes_strength, recent_form, base_elo=cfg.base_elo)
    train_features_df = add_training_features(train_df, elo, form_window=cfg.form_window, ewm_alpha=cfg.ewm_alpha)
    X, y, feature_cols = build_model_inputs(train_features_df)
    ridge_model, lgbm_model, catboost_model, scaler = build_base_models(X, y, cfg)
    sigma = estimate_sigma(X, y, ridge_model, lgbm_model, catboost_model, scaler)
    rng = np.random.default_rng(cfg.random_state)
    pred_df["Team1_WinMargin"] = pred_df.apply(
        lambda r: predict_margin(r["Team1"], r["Team2"], ridge_model, lgbm_model, catboost_model, scaler, final_strength, elo, bayes_strength, recent_form, sigma, rng, cfg),
        axis=1
    )
    pred_df["Team1_WinMargin"] *= cfg.calibration_scale
    pred_df["Team1_WinMargin"] = np.tanh(pred_df["Team1_WinMargin"] / cfg.tanh_divisor) * cfg.tanh_multiplier
    pred_df["Team1_WinMargin"] -= pred_df["Team1_WinMargin"].mean()
    pred_df["Team1_WinMargin"] = pred_df["Team1_WinMargin"].round(1)
    submission = pred_df[["GameID", "Team1_WinMargin"]]
    template = sample_pred_df.drop(columns=["Team1_WinMargin"])
    submission = template.merge(submission, on="GameID")
    pred_path = path_join(cfg.data_dir, cfg.output_predictions_file)
    submission.to_csv(pred_path, index=False)
    print("Predictions saved")
    # --- Model-based round-robin ranking ---
    team_list = sorted(list(teams))
    win_counts = {team: 0 for team in team_list}
    margin_sums = {team: 0.0 for team in team_list}
    n_opponents = {team: 0 for team in team_list}
    for i, t1 in enumerate(team_list):
        for j, t2 in enumerate(team_list):
            if t1 == t2:
                continue
            margin = predict_margin(
                t1, t2, ridge_model, lgbm_model, catboost_model, scaler, final_strength, elo, bayes_strength, recent_form, sigma, rng, cfg
            )
            if margin > 0:
                win_counts[t1] += 1
            margin_sums[t1] += margin
            n_opponents[t1] += 1
    avg_margins = {team: (margin_sums[team] / n_opponents[team]) if n_opponents[team] > 0 else 0.0 for team in team_list}
    ranking_df = pd.DataFrame({
        "Team": team_list,
        "PredWins": [win_counts[t] for t in team_list],
        "AvgMargin": [avg_margins[t] for t in team_list]
    })
    ranking_df = ranking_df.sort_values(["PredWins", "AvgMargin"], ascending=[False, False]).reset_index(drop=True)
    ranking_df["Rank"] = range(1, len(ranking_df) + 1)
    final_rankings = sample_rank_df[["TeamID", "Team"]].merge(
        ranking_df[["Team", "Rank"]],
        on="Team",
        how="left"
    )
    rank_path = path_join(cfg.data_dir, cfg.output_rankings_file)
    final_rankings.to_excel(rank_path, index=False)
    print("Rankings saved (model-based)")
    zip_path = path_join(cfg.data_dir, cfg.output_zip_file)
    with zipfile.ZipFile(zip_path, "w") as z:
        z.write(pred_path, "Predictions.csv")
        z.write(rank_path, "Rankings.xlsx")
    print("Submission.zip created")
    # --- Evaluation: RMSE and Winning Team Accuracy ---
    try:
        actual_margins = sample_pred_df['Team1_WinMargin'].astype(int).values
        predicted_margins = pred_df['Team1_WinMargin'].astype(int).values
        if len(actual_margins) == 75 and len(predicted_margins) == 75:
            rmse = np.sqrt(np.mean((predicted_margins - actual_margins) ** 2))
            print(f"RMSE (Predicted vs Actual Margins): {rmse:.2f}")
            actual_winners = np.sign(actual_margins)
            predicted_winners = np.sign(predicted_margins)
            correct_winner = (actual_winners == predicted_winners).sum()
            accuracy = correct_winner / 75 * 100
            print(f"Winning Team Prediction Accuracy: {accuracy:.2f}% ({correct_winner}/75)")
        else:
            print("Warning: Cannot evaluate RMSE or accuracy, margin counts do not match.")
    except Exception as e:
        print(f"Evaluation error: {e}")

if __name__ == "__main__":
    main()
