"""Two-step SME credit-risk training pipeline .

STEP 1 — Stratified 5-fold CV: leak-proof per-fold StandardScaler on
          continuous features only; cost-sensitive class weights (no SMOTE);
          Recall / AUC-ROC / F1; headless ROC figure.

STEP 2 — Final fit on 100% of the data: global scaler + native serialization
          of XGBoost, CatBoost, and LightGBM for SHAP / causal scripts.

Target: MIS_Status (0 = Paid in Full, 1 = Default).
"""

from __future__ import annotations

import os
import sys
import traceback
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: required on Colab / Kaggle / CI

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    f1_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths (resolved from this file so the script is cwd-independent)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "processed_sme_final.csv"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
ARTIFACTS_DIR = PROJECT_ROOT / "models" / "artifacts"

TARGET_COL = "MIS_Status"
N_SPLITS = 5
RANDOM_STATE = 42
CARDINALITY_THRESHOLD = 10  # nunique > 10 => continuous; else leave unscaled
N_ROC_POINTS = 100
POSITIVE_LABEL = 1  # Default

MODEL_ORDER = [
    "Logistic Regression",
    "Random Forest",
    "XGBoost",
    "LightGBM",
    "CatBoost",
]
CHAMPION_MODELS = ("XGBoost", "LightGBM", "CatBoost")

# Standard published-style tree baselines (not nested-CV tuned).
XGB_BASELINE = dict(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1,
    reg_lambda=1.0,
    tree_method="hist",
    eval_metric="logloss",
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbosity=0,
)
LGBM_BASELINE = dict(
    n_estimators=300,
    num_leaves=31,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbosity=-1,
)
CATBOOST_BASELINE = dict(
    iterations=300,
    depth=6,
    learning_rate=0.05,
    l2_leaf_reg=3.0,
    random_seed=RANDOM_STATE,
    verbose=False,
    allow_writing_files=False,
)
RF_BASELINE = dict(
    n_estimators=200,
    max_depth=20,
    min_samples_leaf=2,
    max_features="sqrt",
    n_jobs=-1,
    random_state=RANDOM_STATE,
)


def ensure_output_dirs() -> None:
    """Create artifact folders if a fresh clone / cloud runtime is empty."""
    for directory in (FIGURES_DIR, RESULTS_DIR, ARTIFACTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
        print(f"  [ok] {directory.relative_to(PROJECT_ROOT)}")


def load_dataset(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    if not path.exists():
        raise FileNotFoundError(
            f"Preprocessed dataset not found: {path}\n"
            "Expected Week-1 output at data/processed/processed_sme_final.csv"
        )

    print(f"Loading {path.relative_to(PROJECT_ROOT)} ...")
    df = pd.read_csv(path, low_memory=False)
    if TARGET_COL not in df.columns:
        raise KeyError(f"Target column '{TARGET_COL}' is missing from {path.name}")

    y = df[TARGET_COL].astype(int)
    X = df.drop(columns=[TARGET_COL]).apply(pd.to_numeric, errors="coerce")
    if X.empty:
        raise ValueError("Feature matrix is empty after dropping the target.")
    if X.isna().any().any():
        n_missing = int(X.isna().sum().sum())
        print(f"  Warning: {n_missing:,} missing feature values; median-imputing.")
        X = X.fillna(X.median(numeric_only=True))

    n_neg, n_pos = int((y == 0).sum()), int((y == 1).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Target is degenerate (only one class present).")

    print(f"  Rows: {len(X):,} | Features: {X.shape[1]}")
    print(f"  Paid in Full (0): {n_neg:,} | Default (1): {n_pos:,}")
    print(f"  Empirical default rate: {n_pos / len(y):.4f}")
    print(f"  Global scale_pos_weight (n0/n1): {n_neg / n_pos:.4f}")
    return X, y


def identify_continuous_columns(frame: pd.DataFrame) -> list[str]:
    """High-cardinality numeric features (nunique > 10). Binary dummies are excluded."""
    return [
        col
        for col in frame.columns
        if int(frame[col].nunique(dropna=False)) > CARDINALITY_THRESHOLD
        and col not in ["NAICS"]  # we do not want to scale the NAICS code
    ]


def apply_scaler_to_continuous(
    X_train: pd.DataFrame,
    X_apply: pd.DataFrame,
    continuous_cols: list[str],
    scaler: StandardScaler | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Fit scaler on training continuous columns only; transform train and apply sets.

    Binary / low-cardinality columns are copied through unchanged.
    """
    X_train_out = X_train.copy()
    X_apply_out = X_apply.copy()
    if not continuous_cols:
        print("  Warning: no continuous columns identified; skipping StandardScaler.")
        return X_train_out, X_apply_out, scaler or StandardScaler()

    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(X_train[continuous_cols])

    X_train_out[continuous_cols] = scaler.transform(X_train[continuous_cols])
    X_apply_out[continuous_cols] = scaler.transform(X_apply[continuous_cols])
    return X_train_out, X_apply_out, scaler


def scale_pos_weight_from_labels(y_train: pd.Series | np.ndarray) -> float:
    y_arr = np.asarray(y_train)
    n_neg = int((y_arr == 0).sum())
    n_pos = int((y_arr == 1).sum())
    if n_pos == 0:
        raise ValueError("Training split contains no default (class 1) labels.")
    return n_neg / n_pos


def build_models(scale_pos_weight: float) -> dict:
    """Cost-sensitive estimators. SMOTE is intentionally not used.

    Resampling would distort predicted default probabilities required by
    expected-loss and SHAP analyses.
    """
    return {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            class_weight="balanced",
            **RF_BASELINE,
        ),
        "XGBoost": XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            **XGB_BASELINE,
        ),
        "LightGBM": LGBMClassifier(
            scale_pos_weight=scale_pos_weight,
            **LGBM_BASELINE,
        ),
        "CatBoost": CatBoostClassifier(
            auto_class_weights="Balanced",
            **CATBOOST_BASELINE,
        ),
    }


def evaluate_fold(y_true, y_pred, y_proba) -> dict[str, float]:
    return {
        "recall": float(recall_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)),
    }


def interpolate_tpr(y_true, y_proba, mean_fpr: np.ndarray) -> np.ndarray:
    fpr, tpr, _ = roc_curve(y_true, y_proba, pos_label=POSITIVE_LABEL)
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    interp_tpr[-1] = 1.0
    return interp_tpr


def run_cross_validation(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, dict, np.ndarray]:
    """STEP 1: leak-proof stratified evaluation. Does not persist models."""
    print("\n" + "=" * 78)
    print("STEP 1  Stratified 5-Fold Cross-Validation (evaluation only)")
    print("=" * 78)
    print("Scaler: StandardScaler fit on TRAINING continuous columns of each fold.")
    print("Imbalance: native class weights / scale_pos_weight — SMOTE is not used.\n")

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    mean_fpr = np.linspace(0, 1, N_ROC_POINTS)
    records: list[dict] = []
    tpr_store = {name: [] for name in MODEL_ORDER}

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        continuous_cols = identify_continuous_columns(X_train)
        print(f"Fold {fold}/{N_SPLITS}")
        print(f"  Continuous columns scaled ({len(continuous_cols)}): {continuous_cols}")
        print(
            f"  Left unscaled (binary/low-cardinality): "
            f"{[c for c in X_train.columns if c not in continuous_cols]}"
        )

        X_train_s, X_val_s, _ = apply_scaler_to_continuous(X_train, X_val, continuous_cols)

        spw = scale_pos_weight_from_labels(y_train)
        print(f"  Train n0/n1 scale_pos_weight = {spw:.4f}")

        models = build_models(spw)
        for name in MODEL_ORDER:
            model = models[name]
            print(f"  Fitting {name} ...", flush=True)
            model.fit(X_train_s, y_train)
            y_proba = model.predict_proba(X_val_s)[:, 1]
            y_pred = (y_proba >= 0.5).astype(int)
            metrics = evaluate_fold(y_val, y_pred, y_proba)
            print(
                f"    {name:<20} Recall={metrics['recall']:.4f}  "
                f"AUC-ROC={metrics['auc_roc']:.4f}  F1={metrics['f1']:.4f}"
            )
            records.append({"stage": "cv", "fold": fold, "model": name, **metrics})
            tpr_store[name].append(interpolate_tpr(y_val, y_proba, mean_fpr))
        print()

    fold_df = pd.DataFrame(records)
    return fold_df, tpr_store, mean_fpr


def write_metrics_benchmark(fold_df: pd.DataFrame) -> pd.DataFrame:
    """Persist fold-level rows and mean/std summary rows in one CSV."""
    metric_cols = ["recall", "auc_roc", "f1"]
    summary_rows: list[dict] = []

    print("\n" + "=" * 78)
    print("MEAN ± STD  |  positive class = Default (MIS_Status = 1)")
    print("=" * 78)
    print(f"{'Model':<22} {'Recall':>20} {'AUC-ROC':>20} {'F1-Score':>20}")
    print("-" * 78)

    for model in MODEL_ORDER:
        subset = fold_df.loc[fold_df["model"] == model, metric_cols]
        means = subset.mean()
        stds = subset.std(ddof=1)
        print(
            f"{model:<22} "
            f"{means['recall']:.4f} ± {stds['recall']:.4f}   "
            f"{means['auc_roc']:.4f} ± {stds['auc_roc']:.4f}   "
            f"{means['f1']:.4f} ± {stds['f1']:.4f}"
        )
        summary_rows.append(
            {"stage": "cv", "fold": "mean", "model": model, **means.to_dict()}
        )
        summary_rows.append(
            {"stage": "cv", "fold": "std", "model": model, **stds.to_dict()}
        )

    print("=" * 78)
    print("Accuracy is omitted as a headline metric (governance Rule 2).")
    print("These CV estimates are the paper's unbiased performance claims.")
    print("STEP 2 models are fit on 100% of the data and must not be scored as test metrics.\n")

    combined = pd.concat([fold_df, pd.DataFrame(summary_rows)], ignore_index=True)
    out_path = RESULTS_DIR / "metrics_benchmark.csv"
    combined.to_csv(out_path, index=False)
    print(f"Wrote {out_path.relative_to(PROJECT_ROOT)}")
    return combined


def plot_cv_roc(tpr_store: dict[str, list[np.ndarray]], mean_fpr: np.ndarray) -> Path:
    out_path = FIGURES_DIR / "roc_curves_comparison.png"
    plt.figure(figsize=(9, 7))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Chance")

    for name in MODEL_ORDER:
        tprs = np.vstack(tpr_store[name])
        mean_tpr = tprs.mean(axis=0)
        mean_auc = auc(mean_fpr, mean_tpr)
        std_tpr = tprs.std(axis=0)
        plt.plot(mean_fpr, mean_tpr, linewidth=2, label=f"{name} (mean AUC={mean_auc:.3f})")
        plt.fill_between(
            mean_fpr,
            np.clip(mean_tpr - std_tpr, 0, 1),
            np.clip(mean_tpr + std_tpr, 0, 1),
            alpha=0.12,
        )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("Stratified 5-Fold CV ROC Curves — SME Credit Risk")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out_path.relative_to(PROJECT_ROOT)}")
    return out_path


def save_champion_native(name: str, model, dest: Path) -> None:
    """Write each booster in its native audit format.

    LGBMClassifier (sklearn API) does not expose ``save_model``; the fitted
    booster does. XGBoost and CatBoost sklearn wrappers do expose it.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if name == "LightGBM":
        booster = getattr(model, "booster_", None)
        if booster is None:
            raise AttributeError(
                "LightGBM model has no booster_ after fit; cannot serialize."
            )
        booster.save_model(str(dest))
        return
    if not hasattr(model, "save_model"):
        raise AttributeError(f"{type(model).__name__} has no save_model().")
    model.save_model(str(dest))


def run_final_fit(X: pd.DataFrame, y: pd.Series) -> None:
    """STEP 2: global scaler + champion serialization. Not used for reported metrics."""
    print("\n" + "=" * 78)
    print("STEP 2  Final fit on 100% of the sample (serialization only)")
    print("=" * 78)

    continuous_cols = identify_continuous_columns(X)
    print(f"Global continuous columns ({len(continuous_cols)}): {continuous_cols}")

    scaler = StandardScaler()
    if not continuous_cols:
        raise RuntimeError("Cannot fit a global scaler: no continuous columns found.")
    scaler.fit(X[continuous_cols])

    X_scaled = X.copy()
    X_scaled[continuous_cols] = scaler.transform(X[continuous_cols])

    scaler_path = ARTIFACTS_DIR / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"Saved global StandardScaler -> {scaler_path.relative_to(PROJECT_ROOT)}")
    print(f"  scaler.feature_names_in_ = {list(scaler.feature_names_in_)}")

    spw = scale_pos_weight_from_labels(y)
    print(f"Full-sample scale_pos_weight = {spw:.4f}")

    champions = {
        "XGBoost": XGBClassifier(scale_pos_weight=spw, **XGB_BASELINE),
        "LightGBM": LGBMClassifier(scale_pos_weight=spw, **LGBM_BASELINE),
        "CatBoost": CatBoostClassifier(auto_class_weights="Balanced", **CATBOOST_BASELINE),
    }
    save_paths = {
        "XGBoost": ARTIFACTS_DIR / "xgboost_best.json",
        "CatBoost": ARTIFACTS_DIR / "catboost_best.bin",
        "LightGBM": ARTIFACTS_DIR / "lightgbm_best.txt",
    }

    for name in CHAMPION_MODELS:
        print(f"Fitting final {name} on all {len(X_scaled):,} rows ...", flush=True)
        model = champions[name]
        model.fit(X_scaled, y)
        dest = save_paths[name]
        save_champion_native(name, model, dest)
        print(f"  Serialized {name} -> {dest.relative_to(PROJECT_ROOT)}")

    print("\nSTEP 2 complete. Downstream SHAP/DoWhy scripts must:")
    print("  1. Load models/artifacts/scaler.joblib")
    print("  2. Transform only scaler.feature_names_in_ (continuous columns)")
    print("  3. Leave one-hot / low-cardinality columns unscaled")


def main() -> int:
    os.chdir(PROJECT_ROOT)
    print("SME Credit Risk — two-step trainer")
    print(f"Project root: {PROJECT_ROOT}\n")
    print("Creating output directories if missing:")
    ensure_output_dirs()

    X, y = load_dataset(DATA_PATH)
    fold_df, tpr_store, mean_fpr = run_cross_validation(X, y)
    write_metrics_benchmark(fold_df)
    plot_cv_roc(tpr_store, mean_fpr)
    run_final_fit(X, y)

    print("\nPipeline finished successfully.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nTraining pipeline failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1) from exc
