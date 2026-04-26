import os
import zipfile
from dataclasses import dataclass
from typing import Dict, Tuple, Set, List
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


# =========================================================
# CONFIG
# =========================================================
@dataclass
class Config:
    data_dir: str = "../algosports23-predictions-2025"
    train_file: str = "Train.csv"
    predictions_file: str = "Predictions.csv"
    sample_predictions_file: str = "Sample_Predictions.csv"
    sample_rankings_file: str = "Sample_Rankings.xlsx"

    output_predictions_file: str = "Predictions.csv"
    output_rankings_file: str = "Rankings.xlsx"
    output_zip_file: str = "Submission.zip"

    random_state: int = 42
    n_simulations: int = 3000

    elo_k: float = 20.0
    base_elo: float = 1500.0

    calibration_scale: float = 1.10
    tanh_divisor: float = 15.0
    tanh_multiplier: float = 34.0

    form_window: int = 5
    ewm_alpha: float = 0.35
    trend_window: int = 5

    cv_splits: int = 5

    cat_iterations: int = 600
    cat_depth: int = 6
    cat_learning_rate: float = 0.1
    cat_l2_leaf_reg: float = 3.0


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
# PREPROCESSING
# =========================================================
def preprocess_train(train_df: pd.DataFrame) -> pd.DataFrame:
    df = train_df.copy()
    df["margin"] = df["HomePts"] - df["AwayPts"]

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    sort_cols = [c for c in ["Date", "GameID"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    return df


# =========================================================
# CORE TEAM STRENGTH
# =========================================================
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
# TIME-SERIES FEATURE HELPERS
# =========================================================
def _rolling_slope(values: pd.Series, window: int) -> pd.Series:
    def slope_fn(arr: np.ndarray) -> float:
        n = len(arr)
        if n < 2:
            return 0.0
        x = np.arange(n, dtype=float)
        y = np.asarray(arr, dtype=float)
        x_mean = x.mean()
        y_mean = y.mean()
        denom = np.sum((x - x_mean) ** 2)
        if denom == 0:
            return 0.0
        return float(np.sum((x - x_mean) * (y - y_mean)) / denom)

    return values.rolling(window, min_periods=2).apply(slope_fn, raw=True)


def add_training_features(
    df: pd.DataFrame,
    elo: Dict[str, float],
    form_window: int = 5,
    ewm_alpha: float = 0.35,
    trend_window: int = 5,
) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values(["Date", "GameID"]).reset_index(drop=True)

    out["elo_home"] = out["HomeTeam"].map(elo).fillna(1500.0)
    out["elo_away"] = out["AwayTeam"].map(elo).fillna(1500.0)
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

    out["ts_ewm_home"] = (
        out.groupby("HomeTeam")["margin"]
        .transform(lambda s: s.shift(1).ewm(alpha=ewm_alpha, adjust=False).mean())
        .fillna(0.0)
    )
    out["ts_ewm_away"] = (
        out.groupby("AwayTeam")["away_team_margin"]
        .transform(lambda s: s.shift(1).ewm(alpha=ewm_alpha, adjust=False).mean())
        .fillna(0.0)
    )
    out["ts_ewm_diff"] = out["ts_ewm_home"] - out["ts_ewm_away"]

    out["ts_trend_home"] = (
        out.groupby("HomeTeam")["margin"]
        .transform(lambda s: _rolling_slope(s.shift(1), trend_window))
        .fillna(0.0)
    )
    out["ts_trend_away"] = (
        out.groupby("AwayTeam")["away_team_margin"]
        .transform(lambda s: _rolling_slope(s.shift(1), trend_window))
        .fillna(0.0)
    )
    out["ts_trend_diff"] = out["ts_trend_home"] - out["ts_trend_away"]

    out["elo_x_form"] = out["elo_diff"] * out["form_diff"]
    out["elo_x_ts_ewm"] = out["elo_diff"] * out["ts_ewm_diff"]
    out["form_x_trend"] = out["form_diff"] * out["ts_trend_diff"]

    out = out.drop(columns=["away_team_margin"])
    out = out.fillna(0.0)

    return out


def build_model_inputs(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    all_features = [
        "elo_diff",
        "form_diff",
        "ts_ewm_diff",
        "ts_trend_diff",
        "elo_x_form",
        "elo_x_ts_ewm",
        "form_x_trend",
    ]
    ts_features = [
        "elo_diff",
        "ts_ewm_diff",
        "ts_trend_diff",
        "elo_x_ts_ewm",
    ]
    X = df[all_features].copy()
    y = df["margin"].copy()
    return X, y, all_features, ts_features


# =========================================================
# BASE MODEL TRAINING
# =========================================================
def build_base_models(
    X: pd.DataFrame,
    y: pd.Series,
    ts_feature_names: List[str],
    cfg: Config,
):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_scaled, y)

    cat_model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=cfg.cat_iterations,
        depth=cfg.cat_depth,
        learning_rate=cfg.cat_learning_rate,
        l2_leaf_reg=cfg.cat_l2_leaf_reg,
        random_seed=cfg.random_state,
        verbose=False,
    )
    cat_model.fit(X, y)

    ts_ridge_model = Ridge(alpha=2.0)
    ts_ridge_model.fit(X[ts_feature_names], y)

    return ridge_model, cat_model, ts_ridge_model, scaler


# =========================================================
# OOF STACKING
# =========================================================
def build_strength_baseline(X: pd.DataFrame) -> np.ndarray:
    return (
        0.50 * (X["elo_diff"].values / 25.0)
        + 0.30 * X["form_diff"].values
        + 0.20 * X["ts_ewm_diff"].values
    )


def generate_oof_base_predictions(
    X: pd.DataFrame,
    y: pd.Series,
    ts_feature_names: List[str],
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tscv = TimeSeriesSplit(n_splits=cfg.cv_splits)

    oof_strength = np.zeros(len(X))
    oof_ridge = np.zeros(len(X))
    oof_cat = np.zeros(len(X))
    oof_ts = np.zeros(len(X))

    full_strength_baseline = build_strength_baseline(X)

    for fold, (train_idx, valid_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train = y.iloc[train_idx]

        # Ridge
        fold_scaler = StandardScaler()
        X_train_scaled = fold_scaler.fit_transform(X_train)
        X_valid_scaled = fold_scaler.transform(X_valid)

        ridge_model = Ridge(alpha=1.0)
        ridge_model.fit(X_train_scaled, y_train)
        oof_ridge[valid_idx] = ridge_model.predict(X_valid_scaled)

        # CatBoost
        cat_model = CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="RMSE",
            iterations=cfg.cat_iterations,
            depth=cfg.cat_depth,
            learning_rate=cfg.cat_learning_rate,
            l2_leaf_reg=cfg.cat_l2_leaf_reg,
            random_seed=cfg.random_state + fold,
            verbose=False,
        )
        cat_model.fit(X_train, y_train)
        oof_cat[valid_idx] = cat_model.predict(X_valid)

        # TS Ridge
        ts_model = Ridge(alpha=2.0)
        ts_model.fit(X_train[ts_feature_names], y_train)
        oof_ts[valid_idx] = ts_model.predict(X_valid[ts_feature_names])

        # Strength baseline
        oof_strength[valid_idx] = full_strength_baseline[valid_idx]

    return oof_strength, oof_ridge, oof_cat, oof_ts


def _winsorize_meta_features(meta_X: pd.DataFrame) -> pd.DataFrame:
    out = meta_X.copy()
    for col in out.columns:
        lower = out[col].quantile(0.01)
        upper = out[col].quantile(0.99)
        out[col] = out[col].clip(lower, upper)
    return out


def train_meta_model(
    oof_strength: np.ndarray,
    oof_ridge: np.ndarray,
    oof_cat: np.ndarray,
    oof_ts: np.ndarray,
    y: pd.Series,
    cv_splits: int = 5,
):
    meta_X = pd.DataFrame({
        "strength": oof_strength,
        "ridge": oof_ridge,
        "cat": oof_cat,
        "ts": oof_ts,
    })

    meta_X = _winsorize_meta_features(meta_X)

    meta_scaler = StandardScaler()
    meta_X_scaled = meta_scaler.fit_transform(meta_X)

    tscv = TimeSeriesSplit(n_splits=cv_splits)

    meta_model = RidgeCV(
        alphas=np.logspace(-3, 3, 25),
        cv=tscv,
        scoring="neg_root_mean_squared_error"
    )
    meta_model.fit(meta_X_scaled, y)

    coef_sum = np.abs(meta_model.coef_).sum()
    if coef_sum > 0:
        weights = meta_model.coef_ / coef_sum
        print(
            "Stable meta weights:",
            {
                "strength": round(weights[0], 4),
                "ridge": round(weights[1], 4),
                "cat": round(weights[2], 4),
                "ts": round(weights[3], 4),
            },
        )
    print(f"Chosen RidgeCV alpha: {meta_model.alpha_}")

    return meta_model, meta_scaler


def estimate_sigma_from_oof(
    oof_strength: np.ndarray,
    oof_ridge: np.ndarray,
    oof_cat: np.ndarray,
    oof_ts: np.ndarray,
    y: pd.Series,
    meta_model,
    meta_scaler: StandardScaler,
) -> float:
    meta_X = pd.DataFrame({
        "strength": oof_strength,
        "ridge": oof_ridge,
        "cat": oof_cat,
        "ts": oof_ts,
    })

    meta_X = _winsorize_meta_features(meta_X)
    meta_X_scaled = meta_scaler.transform(meta_X)
    oof_pred = meta_model.predict(meta_X_scaled)

    residuals = y - oof_pred
    sigma = float(np.std(residuals))
    print(f"Estimated sigma from stable OOF stack: {sigma:.3f}")
    return sigma


# =========================================================
# FUTURE MATCH FEATURE BUILDING
# =========================================================
def build_future_feature_row(
    team1: str,
    team2: str,
    elo: Dict[str, float],
    recent_form: Dict[str, float],
    all_feature_names: List[str],
    base_elo: float = 1500.0,
) -> pd.DataFrame:
    elo_diff = elo.get(team1, base_elo) - elo.get(team2, base_elo)
    form_diff = recent_form.get(team1, 0.0) - recent_form.get(team2, 0.0)

    ts_ewm_diff = form_diff
    ts_trend_diff = 0.0

    row = pd.DataFrame([{
        "elo_diff": elo_diff,
        "form_diff": form_diff,
        "ts_ewm_diff": ts_ewm_diff,
        "ts_trend_diff": ts_trend_diff,
        "elo_x_form": elo_diff * form_diff,
        "elo_x_ts_ewm": elo_diff * ts_ewm_diff,
        "form_x_trend": form_diff * ts_trend_diff,
    }])[all_feature_names]

    return row


# =========================================================
# PREDICTION
# =========================================================
def predict_margin(
    team1: str,
    team2: str,
    final_strength: Dict[str, float],
    elo: Dict[str, float],
    recent_form: Dict[str, float],
    ridge_model: Ridge,
    cat_model: CatBoostRegressor,
    ts_ridge_model: Ridge,
    scaler: StandardScaler,
    meta_model,
    meta_scaler: StandardScaler,
    sigma: float,
    rng: np.random.Generator,
    all_feature_names: List[str],
    ts_feature_names: List[str],
    n_sim: int = 3000,
    base_elo: float = 1500.0,
) -> float:
    strength_pred = final_strength.get(team1, 0.0) - final_strength.get(team2, 0.0)

    row = build_future_feature_row(
        team1=team1,
        team2=team2,
        elo=elo,
        recent_form=recent_form,
        all_feature_names=all_feature_names,
        base_elo=base_elo,
    )

    row_scaled = scaler.transform(row)

    pred_strength = (
        0.50 * (row["elo_diff"].iloc[0] / 25.0)
        + 0.30 * row["form_diff"].iloc[0]
        + 0.20 * row["ts_ewm_diff"].iloc[0]
    )

    pred_ridge = ridge_model.predict(row_scaled)[0]
    pred_cat = cat_model.predict(row)[0]
    pred_ts = ts_ridge_model.predict(row[ts_feature_names])[0]

    meta_row = pd.DataFrame([{
        "strength": pred_strength,
        "ridge": pred_ridge,
        "cat": pred_cat,
        "ts": pred_ts,
    }])

    meta_row = _winsorize_meta_features(meta_row)
    meta_row_scaled = meta_scaler.transform(meta_row)

    stacked_pred = meta_model.predict(meta_row_scaled)[0]

    mu = 0.85 * stacked_pred + 0.15 * strength_pred

    sims = rng.normal(loc=mu, scale=sigma, size=n_sim)
    return float(np.median(sims))


def generate_predictions(
    pred_df: pd.DataFrame,
    final_strength: Dict[str, float],
    elo: Dict[str, float],
    recent_form: Dict[str, float],
    ridge_model: Ridge,
    cat_model: CatBoostRegressor,
    ts_ridge_model: Ridge,
    scaler: StandardScaler,
    meta_model,
    meta_scaler: StandardScaler,
    sigma: float,
    all_feature_names: List[str],
    ts_feature_names: List[str],
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
            recent_form=recent_form,
            ridge_model=ridge_model,
            cat_model=cat_model,
            ts_ridge_model=ts_ridge_model,
            scaler=scaler,
            meta_model=meta_model,
            meta_scaler=meta_scaler,
            sigma=sigma,
            rng=rng,
            all_feature_names=all_feature_names,
            ts_feature_names=ts_feature_names,
            n_sim=cfg.n_simulations,
            base_elo=cfg.base_elo,
        ),
        axis=1,
    )

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

    with zipfile.ZipFile(zip_path, "w") as zipf:
        zipf.write(pred_path, arcname=cfg.output_predictions_file)
        zipf.write(rank_path, arcname=cfg.output_rankings_file)

    print(f"{zip_path} created successfully")


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    cfg = Config()

    train_df, pred_df, sample_pred_df, sample_rank_df = load_data(cfg)
    train_df = preprocess_train(train_df)

    elo = compute_elo(train_df, k=cfg.elo_k, base_elo=cfg.base_elo)
    print("Elo computed.")

    teams = set(train_df["HomeTeam"]).union(set(train_df["AwayTeam"]))
    bayes_strength = compute_bayesian_style_strength(train_df)
    recent_form = compute_recent_form(train_df, teams=teams, n_games=cfg.form_window)

    final_strength = compute_final_strength(
        teams=teams,
        elo=elo,
        bayes_strength=bayes_strength,
        recent_form=recent_form,
        base_elo=cfg.base_elo,
    )

    train_features_df = add_training_features(
        train_df,
        elo=elo,
        form_window=cfg.form_window,
        ewm_alpha=cfg.ewm_alpha,
        trend_window=cfg.trend_window,
    )

    X, y, all_feature_names, ts_feature_names = build_model_inputs(train_features_df)

    oof_strength, oof_ridge, oof_cat, oof_ts = generate_oof_base_predictions(
        X=X,
        y=y,
        ts_feature_names=ts_feature_names,
        cfg=cfg,
    )

    meta_model, meta_scaler = train_meta_model(
        oof_strength=oof_strength,
        oof_ridge=oof_ridge,
        oof_cat=oof_cat,
        oof_ts=oof_ts,
        y=y,
        cv_splits=cfg.cv_splits,
    )

    sigma = estimate_sigma_from_oof(
        oof_strength=oof_strength,
        oof_ridge=oof_ridge,
        oof_cat=oof_cat,
        oof_ts=oof_ts,
        y=y,
        meta_model=meta_model,
        meta_scaler=meta_scaler,
    )

    ridge_model, cat_model, ts_ridge_model, scaler = build_base_models(
        X=X,
        y=y,
        ts_feature_names=ts_feature_names,
        cfg=cfg,
    )

    pred_with_margins = generate_predictions(
        pred_df=pred_df,
        final_strength=final_strength,
        elo=elo,
        recent_form=recent_form,
        ridge_model=ridge_model,
        cat_model=cat_model,
        ts_ridge_model=ts_ridge_model,
        scaler=scaler,
        meta_model=meta_model,
        meta_scaler=meta_scaler,
        sigma=sigma,
        all_feature_names=all_feature_names,
        ts_feature_names=ts_feature_names,
        cfg=cfg,
    )

    submission_df = build_submission(pred_with_margins, sample_pred_df)
    rankings_df = build_rankings(final_strength, sample_rank_df)

    save_outputs(submission_df, rankings_df, cfg)
    print("Pipeline completed successfully.")

    # --- Evaluation: RMSE and Winning Team Accuracy ---
    try:
        actual_margins = sample_pred_df['Team1_WinMargin'].astype(int).values
        predicted_margins = pred_df['Team1_WinMargin'].astype(int).values
        if len(actual_margins) == 75 and len(predicted_margins) == 75:
            # RMSE calculation
            rmse = np.sqrt(np.mean((predicted_margins - actual_margins) ** 2))
            print(f"RMSE (Predicted vs Actual Margins): {rmse:.2f}")
            # Winning team accuracy
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