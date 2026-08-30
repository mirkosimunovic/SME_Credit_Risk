"""SME credit-risk model benchmarking.

Stratified 5-fold CV with per-fold StandardScaler (no leakage) and
cost-sensitive class weights (no SMOTE). Reports Recall, AUC-ROC, and F1.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "processed_sme_final.csv"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"

TARGET_COL = "MIS_Status"
N_SPLITS = 5
RANDOM_STATE = 42
N_ESTIMATORS = 100
N_ROC_POINTS = 100

MODEL_ORDER = [
    "Logistic Regression",
    "Random Forest",
    "XGBoost",
    "LightGBM",
    "CatBoost",
]


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Preprocessed dataset not found: {path}")

    df = pd.read_csv(path, low_memory=False)
    if TARGET_COL not in df.columns:
        raise KeyError(f"Target column '{TARGET_COL}' missing from {path}")

    y = df[TARGET_COL].astype(int).to_numpy()
    X = df.drop(columns=[TARGET_COL]).apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        X = X.fillna(X.median(numeric_only=True))
    X = X.to_numpy(dtype=np.float64)

    n_neg, n_pos = int((y == 0).sum()), int((y == 1).sum())
    print(f"Loaded {path.relative_to(PROJECT_ROOT)}")
    print(f"  Shape: {X.shape[0]:,} rows x {X.shape[1]} features")
    print(f"  Target MIS_Status — Paid in Full (0): {n_neg:,} | Default (1): {n_pos:,}")
    print(f"  Default rate: {n_pos / len(y):.4f}")
    return X, y


def build_models(scale_pos_weight: float) -> dict:
    """Instantiate cost-sensitive classifiers for the current training fold."""
    return {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=N_ESTIMATORS,
            scale_pos_weight=scale_pos_weight,
            tree_method="hist",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbosity=0,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=N_ESTIMATORS,
            is_unbalance=True,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbosity=-1,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=N_ESTIMATORS,
            auto_class_weights="Balanced",
            random_seed=RANDOM_STATE,
            verbose=False,
            allow_writing_files=False,
        ),
    }


def evaluate_fold(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> dict[str, float]:
    return {
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }


def interpolate_tpr(y_true: np.ndarray, y_proba: np.ndarray, mean_fpr: np.ndarray) -> np.ndarray:
    fpr, tpr, _ = roc_curve(y_true, y_proba, pos_label=1)
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    interp_tpr[-1] = 1.0
    return interp_tpr


def print_fold_metrics(fold: int, model_name: str, metrics: dict[str, float]) -> None:
    print(
        f"  Fold {fold} | {model_name:<20} "
        f"Recall={metrics['recall']:.4f}  "
        f"AUC-ROC={metrics['auc_roc']:.4f}  "
        f"F1={metrics['f1']:.4f}"
    )


def summarize_and_save(records: list[dict]) -> pd.DataFrame:
    results = pd.DataFrame(records)
    mean_rows = (
        results.groupby("model", sort=False)[["recall", "auc_roc", "f1"]]
        .mean()
        .reset_index()
    )
    std_rows = (
        results.groupby("model", sort=False)[["recall", "auc_roc", "f1"]]
        .std()
        .reset_index()
    )

    print("\n" + "=" * 72)
    print("MEAN ± STD ACROSS 5 STRATIFIED FOLDS  (positive class = Default)")
    print("=" * 72)
    print(f"{'Model':<22} {'Recall':>18} {'AUC-ROC':>18} {'F1-Score':>18}")
    print("-" * 72)
    for model in MODEL_ORDER:
        m = mean_rows.loc[mean_rows["model"] == model].iloc[0]
        s = std_rows.loc[std_rows["model"] == model].iloc[0]
        print(
            f"{model:<22} "
            f"{m['recall']:.4f} ± {s['recall']:.4f}   "
            f"{m['auc_roc']:.4f} ± {s['auc_roc']:.4f}   "
            f"{m['f1']:.4f} ± {s['f1']:.4f}"
        )
    print("=" * 72)
    print("Accuracy is intentionally omitted per project governance (Rule 2).")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fold_path = RESULTS_DIR / "metrics_benchmark.csv"
    mean_path = RESULTS_DIR / "metrics_benchmark_mean.csv"
    results.to_csv(fold_path, index=False)
    mean_rows.to_csv(mean_path, index=False)
    print(f"Saved fold metrics: {fold_path.relative_to(PROJECT_ROOT)}")
    print(f"Saved mean metrics: {mean_path.relative_to(PROJECT_ROOT)}")
    return mean_rows


def plot_cv_roc(tpr_store: dict[str, list[np.ndarray]], mean_fpr: np.ndarray) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "roc_curves_comparison.png"

    plt.figure(figsize=(9, 7))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Chance")

    for name in MODEL_ORDER:
        tprs = np.vstack(tpr_store[name])
        mean_tpr = tprs.mean(axis=0)
        mean_auc = auc(mean_fpr, mean_tpr)
        std_tpr = tprs.std(axis=0)
        plt.plot(mean_fpr, mean_tpr, linewidth=2, label=f"{name} (AUC={mean_auc:.3f})")
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
    print(f"Saved ROC comparison: {out_path.relative_to(PROJECT_ROOT)}")
    return out_path


def run_benchmark() -> None:
    os.chdir(PROJECT_ROOT)
    X, y = load_dataset(DATA_PATH)

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    mean_fpr = np.linspace(0, 1, N_ROC_POINTS)
    records: list[dict] = []
    tpr_store = {name: [] for name in MODEL_ORDER}

    print("\nStarting stratified 5-fold CV (scaler fit on training fold only).")
    print("Imbalance handling: cost-sensitive weights — SMOTE is not used.\n")

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        scale_pos_weight = n_neg / max(n_pos, 1)
        print(f"Fold {fold}/{N_SPLITS}  scale_pos_weight={scale_pos_weight:.4f}")

        models = build_models(scale_pos_weight)
        for name in MODEL_ORDER:
            model = models[name]
            model.fit(X_train_scaled, y_train)
            y_proba = model.predict_proba(X_val_scaled)[:, 1]
            y_pred = (y_proba >= 0.5).astype(int)

            metrics = evaluate_fold(y_val, y_pred, y_proba)
            print_fold_metrics(fold, name, metrics)
            records.append({"fold": fold, "model": name, **metrics})
            tpr_store[name].append(interpolate_tpr(y_val, y_proba, mean_fpr))
        print()

    summarize_and_save(records)
    plot_cv_roc(tpr_store, mean_fpr)


if __name__ == "__main__":
    try:
        run_benchmark()
    except Exception as exc:
        print(f"Training pipeline failed: {exc}", file=sys.stderr)
        raise
