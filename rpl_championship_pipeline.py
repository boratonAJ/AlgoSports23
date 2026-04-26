import os
import zipfile
from dataclasses import dataclass
from typing import Dict, Tuple, Set

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =========================================================
# CONFIG
# =========================================================
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
    base_elo: float = 1500.0

    calibration_scale: float = 1.25
    tanh_divisor: float = 18.0
    tanh_multiplier: float = 35.0

    form_window: int = 5
    xgb_cv_splits: int = 5


# =========================================================
# IO HELPERS
# =========================================================
def path_join(*parts: str) -> str:
    return os.path.join(*parts)


def load_data(cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(path_join(cfg.data_dir, cfg.train_file))
    pred_df = pd.read_csv(path_join(cfg.data_dir, cfg.predictions_file))
    sample_pred_df = pd.read_csv(path_join(cfg.data_dir, cfg.sample_predictions_file))
    sample_rank_df = pd.read_excel(path_join(cfg.data_dir, cfg.sample_rankings_file))
    print("Data loaded successfully.")
    return train_df, pred_df, sample_pred_df, sample_rank_df


# =========================================================
# PREPROCESSING / STRENGTH ENGINEERING
# =========================================================
def preprocess_train(train_df: pd.DataFrame) -> pd.DataFrame:
    df = train_df.copy()
    df["margin"] = df["HomePts"] - df["AwayPts"]

    # Keep chronological order for time-aware validation and leakage-safe features
    if "Date" in df.columns and "GameID" in df.columns:
        df = df.sort_values(["Date", "GameID"]).reset_index(drop=True)
    elif "Date" in df.columns:
        df = df.sort_values(["Date"]).reset_index(drop=True)
    elif "GameID" in df.columns:
        df = df.sort_values(["GameID"]).reset_index(drop=True)

    return df


def compute_elo(df: pd.DataFrame, k: float = 20.0, base_elo: float = 1500.0) -> Dict[str, float]:
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
    home_strength = df.groupby("HomeTeam")["margin"].mean()
    away_strength = df.groupby("AwayTeam")["margin"].mean()

    teams = set(home_strength.index).union(set(away_strength.index))
    bayes_strength: Dict[str, float] = {}

    for team in teams:
        home_val = float(home_strength.get(team, 0.0))
        away_val = float(-away_strength.get(team, 0.0))
        bayes_strength[team] = 0.6 * home_val + 0.4 * away_val

    return bayes_strength


def compute_recent_form(df: pd.DataFrame, teams: Set[str], n_games: int = 5) -> Dict[str, float]:
    """
    Team-centric recent form using each team's own margin perspective.
    Home team margin = margin
    Away team margin = -margin
    """
    home_games = df[["Date", "GameID", "HomeTeam", "margin"]].copy()
    home_games.columns = ["Date", "GameID", "Team", "team_margin"]

    away_games = df[["Date", "GameID", "AwayTeam", "margin"]].copy()
    away_games.columns = ["Date", "GameID", "Team", "team_margin"]
    away_games["team_margin"] = -away_games["team_margin"]

    team_games = pd.concat([home_games, away_games], ignore_index=True)
    team_games = team_games.sort_values(["Team", "Date", "GameID"]).reset_index(drop=True)

    recent_form: Dict[str, float] = {}
    for team in teams:
        vals = team_games.loc[team_games["Team"] == team, "team_margin"].tail(n_games)
        recent_form[team] = float(vals.mean()) if len(vals) > 0 else 0.0

    return recent_form


def compute_final_strength(
    teams: Set[str],
    elo: Dict[str, float],
    bayes_strength: Dict[str, float],
    recent_form: Dict[str, float],
    base_elo: float = 1500.0,
) -> Dict[str, float]:
    final_strength: Dict[str, float] = {}

    for team in teams:
        elo_norm = (elo.get(team, base_elo) - base_elo) / 100.0
        final_strength[team] = (
            0.5 * bayes_strength.get(team, 0.0)
            + 0.3 * elo_norm
            + 0.2 * recent_form.get(team, 0.0)
        )

    return final_strength


# =========================================================
# LEAKAGE-SAFE FEATURES
# =========================================================
def add_training_features(df: pd.DataFrame, elo: Dict[str, float], form_window: int = 5) -> pd.DataFrame:
    """
    Add leakage-safe training features.
    Rolling form uses only PRIOR games with shift(1).
    """
    out = df.copy()
    out = out.sort_values(["Date", "GameID"]).reset_index(drop=True)

    out["elo_home"] = out["HomeTeam"].map(elo).fillna(1500.0)
    out["elo_away"] = out["AwayTeam"].map(elo).fillna(1500.0)
    out["elo_diff"] = out["elo_home"] - out["elo_away"]

    # Home team historical form from prior home games only
    out["form_home"] = (
        out.groupby("HomeTeam")["margin"]
        .transform(lambda s: s.shift(1).rolling(form_window, min_periods=1).mean())
    )

    # Away team historical form from away-team perspective
    out["away_team_margin"] = -out["margin"]
    out["form_away"] = (
        out.groupby("AwayTeam")["away_team_margin"]
        .transform(lambda s: s.shift(1).rolling(form_window, min_periods=1).mean())
    )

    out["form_home"] = out["form_home"].fillna(0.0)
    out["form_away"] = out["form_away"].fillna(0.0)
    out["form_diff"] = out["form_home"] - out["form_away"]

    out = out.drop(columns=["away_team_margin"])
    out = out.fillna(0.0)

    return out


def build_model_inputs(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X = df[["elo_diff", "form_diff"]].copy()
    X["elo_x_form"] = X["elo_diff"] * X["form_diff"]
    y = df["margin"].copy()
    return X, y


# =========================================================
# MODEL TRAINING
# =========================================================
def train_models(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
    n_splits: int = 5,
) -> Tuple[XGBRegressor, Ridge, StandardScaler]:
    tscv = TimeSeriesSplit(n_splits=n_splits)

    xgb_param_grid = {
        "model__n_estimators": [300, 400],
        "model__max_depth": [5, 6],
        "model__learning_rate": [0.04, 0.05],
        "model__subsample": [0.9, 1.0],
        "model__colsample_bytree": [0.9, 1.0],
        "model__reg_alpha": [0.1, 1.0],
        "model__reg_lambda": [1.0, 2.0],
        "model__gamma": [0.1, 0.5],
    }

    xgb_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", XGBRegressor(random_state=random_state)),
        ]
    )

    xgb_grid = GridSearchCV(
        estimator=xgb_pipeline,
        param_grid=xgb_param_grid,
        cv=tscv,
        n_jobs=-1,
        scoring="neg_root_mean_squared_error",
        verbose=0,
    )
    xgb_grid.fit(X, y)

    best_xgb_pipeline = xgb_grid.best_estimator_
    fitted_scaler = best_xgb_pipeline.named_steps["scaler"]
    fitted_xgb_model = best_xgb_pipeline.named_steps["model"]

    # Ridge trained on full scaled data using same feature columns
    X_scaled_full = fitted_scaler.transform(X)
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_scaled_full, y)

    print("Best XGBoost params:", xgb_grid.best_params_)
    print("ML models trained with TimeSeriesSplit.")

    return fitted_xgb_model, ridge_model, fitted_scaler


def estimate_sigma(
    X: pd.DataFrame,
    y: pd.Series,
    xgb_model: XGBRegressor,
    ridge_model: Ridge,
    scaler: StandardScaler,
) -> float:
    X_scaled = scaler.transform(X)
    train_pred = 0.6 * xgb_model.predict(X_scaled) + 0.4 * ridge_model.predict(X_scaled)
    residuals = y - train_pred
    sigma = float(np.std(residuals))
    print(f"Estimated sigma: {sigma:.3f}")
    return sigma


# =========================================================
# PREDICTION
# =========================================================
def predict_margin(
    team1: str,
    team2: str,
    final_strength: Dict[str, float],
    elo: Dict[str, float],
    xgb_model: XGBRegressor,
    ridge_model: Ridge,
    scaler: StandardScaler,
    sigma: float,
    rng: np.random.Generator,
    n_sim: int = 2000,
    base_elo: float = 1500.0,
) -> float:
    strength_pred = final_strength.get(team1, 0.0) - final_strength.get(team2, 0.0)
    elo_diff = elo.get(team1, base_elo) - elo.get(team2, base_elo)

    # Future games do not know realized form, so use neutral proxy
    form_diff = 0.0
    elo_x_form = elo_diff * form_diff

    X_input = pd.DataFrame(
        [{
            "elo_diff": elo_diff,
            "form_diff": form_diff,
            "elo_x_form": elo_x_form,
        }]
    )
    X_input_scaled = scaler.transform(X_input)

    xgb_pred = xgb_model.predict(X_input_scaled)[0]
    ridge_pred = ridge_model.predict(X_input_scaled)[0]
    ml_pred = 0.6 * xgb_pred + 0.4 * ridge_pred

    mu = (
        0.5 * strength_pred
        + 0.3 * ml_pred
        + 0.2 * (elo_diff / 25.0)
    )

    sims = rng.normal(loc=mu, scale=sigma, size=n_sim)
    return float(np.median(sims))


def generate_predictions(
    pred_df: pd.DataFrame,
    final_strength: Dict[str, float],
    elo: Dict[str, float],
    xgb_model: XGBRegressor,
    ridge_model: Ridge,
    scaler: StandardScaler,
    sigma: float,
    cfg: Config,
) -> pd.DataFrame:
    out = pred_df.copy()
    rng = np.random.default_rng(cfg.random_state)

    out["Team1_WinMargin"] = out.apply(
        lambda row: predict_margin(
            team1=row["Team1"],
            team2=row["Team2"],
            final_strength=final_strength,
            elo=elo,
            xgb_model=xgb_model,
            ridge_model=ridge_model,
            scaler=scaler,
            sigma=sigma,
            rng=rng,
            n_sim=cfg.n_simulations,
            base_elo=cfg.base_elo,
        ),
        axis=1,
    )

    # Final calibration
    out["Team1_WinMargin"] *= cfg.calibration_scale
    out["Team1_WinMargin"] = np.tanh(out["Team1_WinMargin"] / cfg.tanh_divisor) * cfg.tanh_multiplier
    out["Team1_WinMargin"] -= out["Team1_WinMargin"].mean()
    out["Team1_WinMargin"] = out["Team1_WinMargin"].round(1)

    return out


# =========================================================
# OUTPUT BUILDERS
# =========================================================
def build_submission(pred_df: pd.DataFrame, sample_pred_df: pd.DataFrame) -> pd.DataFrame:
    submission = pred_df[["GameID", "Team1_WinMargin"]].copy()

    template = sample_pred_df.copy()
    if "Team1_WinMargin" in template.columns:
        template = template.drop(columns=["Team1_WinMargin"])

    submission = template.merge(submission, on="GameID", how="inner")
    return submission


def build_rankings(final_strength: Dict[str, float], sample_rank_df: pd.DataFrame) -> pd.DataFrame:
    ranking_df = pd.DataFrame(
        {"Team": list(final_strength.keys()), "Strength": list(final_strength.values())}
    )

    ranking_df = ranking_df.sort_values("Strength", ascending=False).reset_index(drop=True)
    ranking_df["Rank"] = np.arange(1, len(ranking_df) + 1)

    final_rankings = sample_rank_df[["TeamID", "Team"]].merge(
        ranking_df[["Team", "Rank"]],
        on="Team",
        how="left",
    )

    return final_rankings


def save_outputs(
    submission_df: pd.DataFrame,
    rankings_df: pd.DataFrame,
    cfg: Config,
) -> None:
    pred_path = path_join(cfg.data_dir, cfg.output_predictions_file)
    rank_path = path_join(cfg.data_dir, cfg.output_rankings_file)
    zip_path = path_join(cfg.data_dir, cfg.output_zip_file)

    submission_df.to_csv(pred_path, index=False)
    print(f"{pred_path} saved")

    rankings_df.to_excel(rank_path, index=False)
    print(f"{rank_path} saved")

    for file_name in [cfg.output_predictions_file, cfg.output_rankings_file]:
        file_path = path_join(cfg.data_dir, file_name)
        if os.path.exists(file_path):
            print(f"{file_path} exists and is ready for submission.")
        else:
            raise FileNotFoundError(f"Missing expected output file: {file_path}")

    with zipfile.ZipFile(zip_path, "w") as zipf:
        zipf.write(pred_path, arcname=cfg.output_predictions_file)
        zipf.write(rank_path, arcname=cfg.output_rankings_file)

    print(f"{zip_path} created successfully")


# =========================================================
# MAIN PIPELINE
# =========================================================
def main() -> None:
    cfg = Config()

    # Load
    train_df, pred_df, sample_pred_df, sample_rank_df = load_data(cfg)

    # Preprocess
    train_df = preprocess_train(train_df)

    # Strength models
    elo = compute_elo(train_df, k=cfg.elo_k, base_elo=cfg.base_elo)
    print("Elo computed.")

    bayes_strength = compute_bayesian_style_strength(train_df)
    teams = set(train_df["HomeTeam"]).union(set(train_df["AwayTeam"]))
    recent_form = compute_recent_form(train_df, teams=teams, n_games=cfg.form_window)

    final_strength = compute_final_strength(
        teams=teams,
        elo=elo,
        bayes_strength=bayes_strength,
        recent_form=recent_form,
        base_elo=cfg.base_elo,
    )

    # Leakage-safe feature engineering
    train_features_df = add_training_features(train_df, elo=elo, form_window=cfg.form_window)
    X, y = build_model_inputs(train_features_df)

    # Train models with time-aware validation
    xgb_model, ridge_model, scaler = train_models(
        X=X,
        y=y,
        random_state=cfg.random_state,
        n_splits=cfg.xgb_cv_splits,
    )

    sigma = estimate_sigma(
        X=X,
        y=y,
        xgb_model=xgb_model,
        ridge_model=ridge_model,
        scaler=scaler,
    )

    # Predict derby matches
    pred_with_margins = generate_predictions(
        pred_df=pred_df,
        final_strength=final_strength,
        elo=elo,
        xgb_model=xgb_model,
        ridge_model=ridge_model,
        scaler=scaler,
        sigma=sigma,
        cfg=cfg,
    )

    # Build outputs
    submission_df = build_submission(pred_with_margins, sample_pred_df)
    rankings_df = build_rankings(final_strength, sample_rank_df)

    # Save
    save_outputs(submission_df, rankings_df, cfg)


if __name__ == "__main__":
    main()