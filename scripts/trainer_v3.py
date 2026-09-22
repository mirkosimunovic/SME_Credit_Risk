"""Two-step SME credit-risk trainer v3 (unweighted ROC-AUC baseline).

Uses chronological splits from scripts/preprocess_v3.py:

    data/processed/X_train_v3.csv, y_train_v3.csv
    data/processed/X_oot_v3.csv,   y_oot_v3.csv

Tuning objective is roc_auc. Champions are fit with unweighted binary
cross-entropy (no sample_weight, no scale_pos_weight). Financial-weight
helpers remain as optional post-hoc utilities.

Target: MIS_Status (0 = Paid in Full, 1 = Default).
"""

from __future__ import annotations

import json
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
from fknni import FastKNNImputer
from sklearn.metrics import (
    auc,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
ARTIFACTS_DIR = PROJECT_ROOT / "models" / "artifacts"

TARGET_COL = "MIS_Status"
N_SPLITS = 5
RANDOM_STATE = 42
CARDINALITY_THRESHOLD = 10  # nunique > 10 => continuous (scaled); else leave raw
LIFT_FRACTION = 0.10
POSITIVE_LABEL = 1
ASSUMED_INTEREST_RATE = 0.06

MODEL_ORDER = ["XGBoost", "LightGBM", "CatBoost"]
N_TUNE_ITER = 12
TUNE_CV = 2
TUNE_SUBSAMPLE = 40_000
BOOTSTRAP_ITERS = 1_000
VIF_WARN = 5.0


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


USE_GPU = _cuda_available()
# XGBoost 2+/3: gpu_hist was removed. GPU is tree_method='hist' + device='cuda'.
XGB_DEVICE = "cuda" if USE_GPU else "cpu"
LGBM_DEVICE = "gpu" if USE_GPU else "cpu"
CAT_TASK = "GPU" if USE_GPU else "CPU"


def xgb_fixed(seed: int = RANDOM_STATE) -> dict:
    return dict(
        eval_metric="logloss",
        tree_method="hist",
        device=XGB_DEVICE,
        n_jobs=-1,
        random_state=seed,
        verbosity=0,
    )


def lgbm_fixed(seed: int = RANDOM_STATE) -> dict:
    return dict(
        device=LGBM_DEVICE,
        n_jobs=-1,
        random_state=seed,
        verbosity=-1,
    )


def cat_fixed(seed: int = RANDOM_STATE) -> dict:
    return dict(
        task_type=CAT_TASK,
        verbose=False,
        allow_writing_files=False,
        random_seed=seed,
    )


XGB_SEARCH = {
    "n_estimators": [150, 300, 500],
    "max_depth": [4, 6, 8],
    "learning_rate": [0.03, 0.05, 0.1],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "min_child_weight": [1, 5],
    "reg_lambda": [1.0, 5.0],
}
LGBM_SEARCH = {
    "n_estimators": [150, 300, 500],
    "num_leaves": [15, 31, 63],
    "learning_rate": [0.03, 0.05, 0.1],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "min_child_samples": [10, 20, 50],
}
CAT_SEARCH = {
    "iterations": [150, 300, 500],
    "depth": [4, 6, 8],
    "learning_rate": [0.03, 0.05, 0.1],
    "l2_leaf_reg": [1.0, 3.0, 7.0],
}


def ensure_dirs() -> None:
    for directory in (FIGURES_DIR, RESULTS_DIR, ARTIFACTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
        print(f"  [ok] {directory.relative_to(PROJECT_ROOT)}")


def load_xy(split: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load one chronological split. Target is always MIS_Status."""
    x_path = PROCESSED_DIR / f"X_{split}_v3.csv"
    y_path = PROCESSED_DIR / f"y_{split}_v3.csv"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"Missing {split} v3 split. Run scripts/preprocess_v3.py first.\n"
            f"  expected {x_path}\n  expected {y_path}"
        )
    X = pd.read_csv(x_path, low_memory=False)
    y = pd.read_csv(y_path, low_memory=False)
    if TARGET_COL not in y.columns:
        y = y.iloc[:, 0]
    else:
        y = y[TARGET_COL]
    y = y.astype(int)
    if len(X) != len(y):
        raise ValueError(f"{split}: X has {len(X):,} rows but y has {len(y):,}.")
    print(
        f"  Loaded {split}: X={X.shape}  defaults={int(y.sum()):,}  "
        f"rate={float(y.mean()):.6f}"
    )
    return X, y


def encode_non_numeric(X_fit: pd.DataFrame, X_apply: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Map leftover string columns (State, RevLineCr, LowDoc, ...) to integers.

    Codes are learned on X_fit only. Unseen / missing tokens become NaN so the
    fold imputer — not this encoder — fills them. Numeric columns pass through.
    """
    X_tr = X_fit.copy()
    X_ap = X_apply.copy()
    encodings: dict[str, dict] = {}

    for col in X_tr.columns:
        if pd.api.types.is_numeric_dtype(X_tr[col]):
            X_tr[col] = pd.to_numeric(X_tr[col], errors="coerce")
            X_ap[col] = pd.to_numeric(X_ap[col], errors="coerce")
            continue

        as_num_tr = pd.to_numeric(X_tr[col], errors="coerce")
        as_num_ap = pd.to_numeric(X_ap[col], errors="coerce")
        mostly_numeric = as_num_tr.notna().mean() >= 0.95
        if mostly_numeric:
            X_tr[col] = as_num_tr
            X_ap[col] = as_num_ap
            continue

        tokens_tr = X_tr[col].where(X_tr[col].notna(), np.nan).astype("string")
        tokens_ap = X_ap[col].where(X_ap[col].notna(), np.nan).astype("string")
        vocab = pd.Index(tokens_tr.dropna().unique())
        mapping = {str(v): i for i, v in enumerate(vocab)}
        encodings[col] = mapping
        X_tr[col] = tokens_tr.map(mapping).astype(float)
        X_ap[col] = tokens_ap.map(mapping).astype(float)

    return X_tr, X_ap, encodings


def encode_with_mapping(
    X: pd.DataFrame, encodings: dict, feature_columns: list[str]
) -> pd.DataFrame:
    """Apply encodings learned on a training slice to a new frame."""
    out = pd.DataFrame(index=X.index)
    for col in feature_columns:
        if col not in X.columns:
            out[col] = np.nan
            continue
        series = X[col]
        if col in encodings:
            tokens = series.where(series.notna(), np.nan).astype("string")
            out[col] = tokens.map(encodings[col]).astype(float)
        else:
            out[col] = pd.to_numeric(series, errors="coerce")
    return out


def identify_continuous(frame: pd.DataFrame) -> list[str]:
    """High-cardinality numeric features. Binary / low-card flags stay unscaled."""
    return [
        col
        for col in frame.columns
        if int(frame[col].nunique(dropna=False)) > CARDINALITY_THRESHOLD
    ]


def make_imputer() -> FastKNNImputer:
    """fknni FastKNNImputer. Only fit_transform exists — do not call fit/transform."""
    return FastKNNImputer(n_neighbors=5, strategy="mean")


def _to_numpy(frame: pd.DataFrame) -> np.ndarray:
    return frame.to_numpy(dtype=np.float64, copy=True)


def _knn_fit_transform(X: np.ndarray) -> np.ndarray:
    """Run FastKNNImputer.fit_transform and coerce CuPy output back to NumPy."""
    imputed = make_imputer().fit_transform(X)
    if hasattr(imputed, "get"):
        imputed = imputed.get()
    return np.asarray(imputed, dtype=np.float64)


def impute_training_slice(X_enc: pd.DataFrame) -> pd.DataFrame:
    """Impute the training slice in isolation (no validation / OOT rows)."""
    print(
        f"    FastKNNImputer.fit_transform on training slice ({len(X_enc):,} rows) ...",
        flush=True,
    )
    arr = _knn_fit_transform(_to_numpy(X_enc))
    return pd.DataFrame(arr, columns=X_enc.columns, index=X_enc.index)


def impute_from_reference(X_ref_imp: pd.DataFrame, X_apply_enc: pd.DataFrame) -> pd.DataFrame:
    """Fill apply-set NaNs using a complete training matrix as the neighbor pool."""
    n_ref = len(X_ref_imp)
    X_apply_aligned = X_apply_enc.reindex(columns=list(X_ref_imp.columns))
    stacked = np.vstack([_to_numpy(X_ref_imp), _to_numpy(X_apply_aligned)])
    print(
        f"    FastKNNImputer.fit_transform on stacked reference+apply "
        f"({stacked.shape[0]:,} rows) ...",
        flush=True,
    )
    stacked_imp = _knn_fit_transform(stacked)
    return pd.DataFrame(
        stacked_imp[n_ref:],
        columns=X_ref_imp.columns,
        index=X_apply_enc.index,
    )


class StackedFastKNNImputer:
    """Serializable stand-in for FastKNNImputer, which has no transform()."""

    def __init__(self, n_neighbors: int = 5, strategy: str = "mean"):
        self.n_neighbors = n_neighbors
        self.strategy = strategy
        self.reference_: pd.DataFrame | None = None
        self.encodings_: dict = {}
        self.feature_columns_: list[str] = []

    def fit_reference(self, X_fit_enc: pd.DataFrame, encodings: dict) -> pd.DataFrame:
        self.encodings_ = encodings
        self.feature_columns_ = list(X_fit_enc.columns)
        self.reference_ = impute_training_slice(X_fit_enc)
        return self.reference_

    def transform(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        if self.reference_ is None:
            raise RuntimeError("StackedFastKNNImputer has no training reference.")
        X_enc = encode_with_mapping(X_raw, self.encodings_, self.feature_columns_)
        return impute_from_reference(self.reference_, X_enc)

    def transform_encoded(self, X_enc: pd.DataFrame) -> pd.DataFrame:
        if self.reference_ is None:
            raise RuntimeError("StackedFastKNNImputer has no training reference.")
        return impute_from_reference(self.reference_, X_enc)


def impute_and_scale(
    X_fit: pd.DataFrame,
    X_apply: pd.DataFrame,
    imputer: StackedFastKNNImputer | None = None,
    scaler: StandardScaler | None = None,
    continuous_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, StackedFastKNNImputer, StandardScaler, list[str]]:
    """Leak-proof numeric prep: encode -> impute train -> impute apply -> scale."""
    X_fit_enc, X_apply_enc, encodings = encode_non_numeric(X_fit, X_apply)

    if imputer is None or imputer.reference_ is None:
        imputer = StackedFastKNNImputer()
        X_fit_imp = imputer.fit_reference(X_fit_enc, encodings)
    else:
        X_fit_imp = imputer.reference_

    X_apply_imp = imputer.transform_encoded(X_apply_enc)

    if continuous_cols is None:
        continuous_cols = identify_continuous(X_fit_imp)
    print(f"    Continuous columns scaled ({len(continuous_cols)}): {continuous_cols}")

    if scaler is None:
        scaler = StandardScaler()
        if continuous_cols:
            scaler.fit(X_fit_imp[continuous_cols])

    X_fit_out = X_fit_imp.copy()
    X_apply_out = X_apply_imp.copy()
    if continuous_cols:
        X_fit_out[continuous_cols] = scaler.transform(X_fit_imp[continuous_cols])
        X_apply_out[continuous_cols] = scaler.transform(X_apply_imp[continuous_cols])
    return X_fit_out, X_apply_out, imputer, scaler, continuous_cols


def _principal_amount(X: pd.DataFrame) -> np.ndarray:
    """Gross approval in dollars. preprocess_v3 drops GrAppv; invert log1p if needed."""
    if "GrAppv" in X.columns:
        g = pd.to_numeric(X["GrAppv"], errors="coerce").to_numpy(dtype=float)
    elif "Log_GrAppv" in X.columns:
        logg = pd.to_numeric(X["Log_GrAppv"], errors="coerce").to_numpy(dtype=float)
        g = np.expm1(logg)
    else:
        raise KeyError("Need GrAppv or Log_GrAppv to compute financial weights.")
    return np.where(np.isfinite(g) & (g > 0), g, np.nan)


def _term_years(X: pd.DataFrame) -> np.ndarray:
    if "Term_Years" not in X.columns:
        return np.full(len(X), np.nan)
    t = pd.to_numeric(X["Term_Years"], errors="coerce").to_numpy(dtype=float)
    return np.where(np.isfinite(t) & (t > 0), t, np.nan)


def _guarantee_ratio(X: pd.DataFrame) -> np.ndarray:
    if "Guarantee_Ratio" not in X.columns:
        return np.zeros(len(X), dtype=float)
    g = pd.to_numeric(X["Guarantee_Ratio"], errors="coerce").to_numpy(dtype=float)
    g = np.where(np.isfinite(g), g, 0.0)
    return np.clip(g, 0.0, 1.0)


def loan_cashflows(
    X: pd.DataFrame, assumed_interest_rate: float = ASSUMED_INTEREST_RATE
) -> tuple[np.ndarray, np.ndarray]:
    """Per-loan opportunity profit (performing) and unsecured loss (default)."""
    principal = _principal_amount(X)
    term = _term_years(X)
    guar = _guarantee_ratio(X)
    profit = principal * assumed_interest_rate * term
    loss = principal * (1.0 - guar)
    profit = np.where(np.isfinite(profit) & (profit >= 0), profit, np.nan)
    loss = np.where(np.isfinite(loss) & (loss >= 0), loss, np.nan)
    return profit, loss


def calculate_financial_weights(
    X: pd.DataFrame, y: pd.Series, assumed_interest_rate: float = ASSUMED_INTEREST_RATE
) -> np.ndarray:
    """Asymmetric cost weights, mean-normalized to 1.0.

    Default (y=1): un-guaranteed principal GrAppv * (1 - Guarantee_Ratio).
    Paid-in-full (y=0): forgone interest GrAppv * rate * Term_Years.
    Must be called on *unscaled* features (dollar units, not z-scores).
    """
    y_arr = np.asarray(y, dtype=int)
    profit, loss = loan_cashflows(X, assumed_interest_rate)
    raw = np.where(y_arr == 1, loss, profit)
    raw = np.where(np.isfinite(raw) & (raw > 0), raw, np.nan)
    raw = np.where(np.isnan(raw), 1.0, raw)
    mean = float(np.mean(raw))
    if not np.isfinite(mean) or mean <= 0:
        return np.ones(len(y_arr), dtype=float)
    return (raw / mean).astype(float)


def expected_portfolio_profit(
    X: pd.DataFrame,
    y_true,
    y_pred,
    assumed_interest_rate: float = ASSUMED_INTEREST_RATE,
) -> float:
    """P&L of originated loans (predicted PIF) at a 0.5 cutoff.

    TN: collect assumed interest. FN: lose un-guaranteed principal.
    Denied loans (predicted default) contribute 0.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    profit, loss = loan_cashflows(X, assumed_interest_rate)
    profit = np.nan_to_num(profit, nan=0.0)
    loss = np.nan_to_num(loss, nan=0.0)
    originated = y_pred == 0
    tn = originated & (y_true == 0)
    fn = originated & (y_true == 1)
    return float(profit[tn].sum() - loss[fn].sum())


def report_vif(X_scaled: pd.DataFrame, continuous_cols: list[str], label: str) -> list[dict]:
    """Variance inflation on scaled continuous columns. Warn if VIF > 5."""
    if not continuous_cols:
        print(f"  VIF [{label}]: no continuous columns.")
        return []
    mat = X_scaled[continuous_cols].to_numpy(dtype=np.float64)
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    rows = []
    print(f"  VIF [{label}] on {len(continuous_cols)} continuous features:")
    for i, col in enumerate(continuous_cols):
        try:
            vif = float(variance_inflation_factor(mat, i))
        except Exception:
            vif = float("nan")
        rows.append({"feature": col, "vif": vif})
        flag = "  WARNING VIF>5" if (np.isfinite(vif) and vif > VIF_WARN) else "  "
        print(f"  {flag}  {col:<24} VIF={vif:.3f}")
    return rows


def _tune_subsample(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    n = min(TUNE_SUBSAMPLE, len(X))
    if n == len(X):
        return X, y
    Xs, _, ys, _ = train_test_split(
        X, y, train_size=n, stratify=y, random_state=RANDOM_STATE
    )
    return Xs, ys


def tune_model(name: str, X: pd.DataFrame, y: pd.Series) -> dict:
    """Unweighted RandomizedSearchCV maximizing ROC-AUC."""
    Xs, ys = _tune_subsample(X, y)
    print(
        f"    Tuning {name} (n_iter={N_TUNE_ITER}, cv={TUNE_CV}, "
        f"subsample={len(Xs):,}, gpu={USE_GPU}, scoring=roc_auc) ...",
        flush=True,
    )
    if name == "XGBoost":
        est = XGBClassifier(**xgb_fixed())
        grid = XGB_SEARCH
    elif name == "LightGBM":
        est = LGBMClassifier(**lgbm_fixed())
        grid = LGBM_SEARCH
    else:
        est = CatBoostClassifier(**cat_fixed())
        grid = CAT_SEARCH
    search = RandomizedSearchCV(
        est,
        grid,
        n_iter=N_TUNE_ITER,
        cv=TUNE_CV,
        scoring="roc_auc",
        n_jobs=1,
        random_state=RANDOM_STATE,
        refit=True,
        verbose=0,
    )
    search.fit(Xs, ys)
    print(
        f"    {name} best ROC-AUC={search.best_score_:.4f}  "
        f"params={search.best_params_}"
    )
    return dict(search.best_params_)


def instantiate_tuned(name: str, params: dict, seed: int = RANDOM_STATE):
    if name == "XGBoost":
        return XGBClassifier(**xgb_fixed(seed), **params)
    if name == "LightGBM":
        return LGBMClassifier(**lgbm_fixed(seed), **params)
    return CatBoostClassifier(**cat_fixed(seed), **params)


def bootstrap_auc_ci(
    y_true, y_proba, n_iterations: int = BOOTSTRAP_ITERS, seed: int = RANDOM_STATE
) -> dict:
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    n = len(y_true)
    roc_scores, pr_scores = [], []
    for _ in range(n_iterations):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_proba[idx]
        if yt.min() == yt.max():
            continue
        roc_scores.append(roc_auc_score(yt, yp))
        pr_scores.append(average_precision_score(yt, yp))
    roc = np.asarray(roc_scores)
    pr = np.asarray(pr_scores)
    return {
        "auc_roc_ci95_low": float(np.percentile(roc, 2.5)),
        "auc_roc_ci95_high": float(np.percentile(roc, 97.5)),
        "auc_pr_ci95_low": float(np.percentile(pr, 2.5)),
        "auc_pr_ci95_high": float(np.percentile(pr, 97.5)),
        "n_bootstrap": int(len(roc)),
    }


def lift_at_fraction(y_true, y_proba, fraction: float = LIFT_FRACTION) -> float:
    """Default rate in the top-`fraction` scored loans / base default rate."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    n = len(y_true)
    if n == 0:
        return float("nan")
    base = float(y_true.mean())
    if base <= 0:
        return float("nan")
    n_top = max(1, int(np.ceil(n * fraction)))
    order = np.argsort(y_proba)[::-1][:n_top]
    return float(y_true[order].mean() / base)


def evaluate(y_true, y_proba) -> dict[str, float]:
    y_pred = (np.asarray(y_proba) >= 0.5).astype(int)
    return {
        "auc_roc": float(roc_auc_score(y_true, y_proba)),
        "auc_pr": float(average_precision_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "lift_at_10": lift_at_fraction(y_true, y_proba, LIFT_FRACTION),
    }


def print_metrics(label: str, metrics: dict[str, float]) -> None:
    print(
        f"    {label:<22} "
        f"AUC-ROC={metrics['auc_roc']:.4f}  "
        f"AUC-PR={metrics['auc_pr']:.4f}  "
        f"F1={metrics['f1']:.4f}  "
        f"Recall={metrics['recall']:.4f}  "
        f"BalAcc={metrics['balanced_accuracy']:.4f}  "
        f"Lift@10%={metrics['lift_at_10']:.3f}"
    )


def save_champion(name: str, model, dest: Path) -> None:
    """Native audit formats. LGBMClassifier.save_model lives on booster_."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if name == "LightGBM":
        booster = getattr(model, "booster_", None)
        if booster is None:
            raise AttributeError("LightGBM model has no booster_ after fit.")
        booster.save_model(str(dest))
        return
    model.save_model(str(dest))


# ===========================================================================
# STEP 1 — leak-proof CV on the training partition
# ===========================================================================
def run_cross_validation(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    print("\n" + "=" * 78)
    print("STEP 1  Stratified 5-fold CV on X_train / y_train")
    print("=" * 78)
    print("Imputer: FastKNNImputer.fit_transform on the train fold, then on stacked val.")
    print("         (Library has no fit/transform; val neighbors come from imputed train.)")
    print("Scaler:  StandardScaler on continuous columns (nunique > 10) — train fold only.")
    print("Fit:     unweighted binary cross-entropy. No sample_weight / scale_pos_weight.")
    print("Tuning:  RandomizedSearchCV scoring=roc_auc.")
    print("OOT is held out of this entire loop.\n")

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    records: list[dict] = []

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y), start=1):
        X_tr, X_va = X.iloc[tr_idx].copy(), X.iloc[va_idx].copy()
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        print(f"Fold {fold}/{N_SPLITS}  n_train={len(X_tr):,}  n_val={len(X_va):,}")

        X_tr_p, X_va_p, _, _, continuous_cols = impute_and_scale(X_tr, X_va)
        report_vif(X_tr_p, continuous_cols, label=f"fold {fold} train")
        for name in MODEL_ORDER:
            print(f"  Fitting {name} ...", flush=True)
            best = tune_model(name, X_tr_p, y_tr)
            model = instantiate_tuned(name, best)
            model.fit(X_tr_p, y_tr)
            y_proba = model.predict_proba(X_va_p)[:, 1]
            metrics = evaluate(y_va, y_proba)
            print_metrics(name, metrics)
            records.append({"stage": "cv", "fold": fold, "model": name, **metrics})
        print()

    fold_df = pd.DataFrame(records)
    metric_cols = [
        "auc_roc",
        "auc_pr",
        "f1",
        "recall",
        "balanced_accuracy",
        "lift_at_10",
    ]

    print("=" * 78)
    print("MEAN ± STD across 5 stratified folds  (positive class = Default)")
    print("=" * 78)
    header = f"{'Model':<22}" + "".join(f"{m:>22}" for m in metric_cols)
    print(header)
    print("-" * len(header))

    summary_rows: list[dict] = []
    for name in MODEL_ORDER:
        subset = fold_df.loc[fold_df["model"] == name, metric_cols]
        means, stds = subset.mean(), subset.std(ddof=1)
        cells = []
        for m in metric_cols:
            cells.append(f"{means[m]:.4f}±{stds[m]:.4f}".rjust(22))
        print(f"{name:<22}" + "".join(cells))
        summary_rows.append({"stage": "cv", "fold": "mean", "model": name, **means.to_dict()})
        summary_rows.append({"stage": "cv", "fold": "std", "model": name, **stds.to_dict()})

    print("=" * 78)
    print("Accuracy is omitted as a headline metric (rare-event default).")
    print("These CV numbers are the in-time performance claims. STEP 3 is OOT.\n")

    combined = pd.concat([fold_df, pd.DataFrame(summary_rows)], ignore_index=True)
    out_path = RESULTS_DIR / "metrics_benchmark_v3.csv"
    combined.to_csv(out_path, index=False)
    print(f"Wrote {out_path.relative_to(PROJECT_ROOT)}")
    return combined


# ===========================================================================
# STEP 2 — global artifacts on 100% of X_train
# ===========================================================================
def run_final_fit(
    X_train: pd.DataFrame, y_train: pd.Series, X_oot: pd.DataFrame
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    print("\n" + "=" * 78)
    print("STEP 2  Global imputer / scaler / champion fit on 100% of X_train")
    print("=" * 78)
    print("OOT is imputed against the frozen imputed-train reference. It is not used to fit it.\n")

    X_train_p, X_oot_p, imputer, scaler, continuous_cols = impute_and_scale(X_train, X_oot)
    report_vif(X_train_p, continuous_cols, label="final train")

    imputer_path = ARTIFACTS_DIR / "imputer_v3.joblib"
    scaler_path = ARTIFACTS_DIR / "scaler_v3.joblib"
    joblib.dump(imputer, imputer_path)
    joblib.dump(scaler, scaler_path)
    print(f"  Saved {imputer_path.relative_to(PROJECT_ROOT)}  (StackedFastKNNImputer)")
    print(f"  Saved {scaler_path.relative_to(PROJECT_ROOT)}")
    print(f"  scaler.feature_names_in_ = {list(getattr(scaler, 'feature_names_in_', continuous_cols))}")

    best_params: dict[str, dict] = {}
    save_paths = {
        "CatBoost": ARTIFACTS_DIR / "catboost_best_v3.bin",
        "LightGBM": ARTIFACTS_DIR / "lightgbm_best_v3.txt",
    }
    fitted = {}
    for name in MODEL_ORDER:
        print(f"  Tuning + fitting final {name} on {len(X_train_p):,} training rows ...", flush=True)
        best = tune_model(name, X_train_p, y_train)
        best_params[name] = best
        if name == "XGBoost":
            raw_path = ARTIFACTS_DIR / "xgboost_raw_v3.json"
            cal_path = ARTIFACTS_DIR / "xgboost_calibrated_v3.joblib"
            # Track A — uncalibrated trees for SHAP / TreeExplainer.
            raw = instantiate_tuned(name, best)
            raw.fit(X_train_p, y_train)
            save_champion(name, raw, raw_path)
            print(f"    Serialized raw trees -> {raw_path.relative_to(PROJECT_ROOT)}")
            # Track B — isotonic CV calibration on train folds only (OOT never seen).
            cal = CalibratedClassifierCV(
                estimator=instantiate_tuned(name, best),
                method="isotonic",
                cv=5,
            )
            print("    Fitting CalibratedClassifierCV(method='isotonic', cv=5) ...", flush=True)
            cal.fit(X_train_p, y_train)
            joblib.dump(cal, cal_path)
            print(f"    Serialized calibrated -> {cal_path.relative_to(PROJECT_ROOT)}")
            fitted[name] = cal
            continue
        model = instantiate_tuned(name, best)
        model.fit(X_train_p, y_train)
        save_champion(name, model, save_paths[name])
        print(f"    Serialized -> {save_paths[name].relative_to(PROJECT_ROOT)}")
        fitted[name] = model

    hp_path = RESULTS_DIR / "best_hyperparams_v3.json"
    with open(hp_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "gpu": USE_GPU,
                "tuning_objective": "roc_auc",
                "params": best_params,
            },
            fh,
            indent=2,
        )
    print(f"  Saved {hp_path.relative_to(PROJECT_ROOT)}")
    return fitted, X_train_p, X_oot_p


# ===========================================================================
# STEP 3 — chronological OOT evaluation
# ===========================================================================
def run_oot_evaluation(
    models: dict, X_oot_p: pd.DataFrame, y_oot: pd.Series, X_oot_raw: pd.DataFrame
) -> pd.DataFrame:
    print("\n" + "=" * 78)
    print("STEP 3  Out-of-time evaluation on X_oot / y_oot")
    print("=" * 78)
    print("This split is newer than every training row. It is the paper's temporal test.\n")

    records: list[dict] = []
    proba_store: dict[str, np.ndarray] = {}
    for name in MODEL_ORDER:
        print(f"  Scoring {name} on OOT ({len(X_oot_p):,} rows) ...", flush=True)
        y_proba = models[name].predict_proba(X_oot_p)[:, 1]
        metrics = evaluate(y_oot, y_proba)
        print_metrics(f"OOT {name}", metrics)
        print(f"    Bootstrap {BOOTSTRAP_ITERS} AUC CIs for {name} ...", flush=True)
        ci = bootstrap_auc_ci(y_oot, y_proba)
        print(
            f"    AUC-ROC 95% CI=[{ci['auc_roc_ci95_low']:.4f}, {ci['auc_roc_ci95_high']:.4f}]  "
            f"AUC-PR 95% CI=[{ci['auc_pr_ci95_low']:.4f}, {ci['auc_pr_ci95_high']:.4f}]"
        )
        records.append({"stage": "oot", "fold": "oot", "model": name, **metrics, **ci})
        proba_store[name] = y_proba

    oot_df = pd.DataFrame(records)
    bench_path = RESULTS_DIR / "metrics_benchmark_v3.csv"
    if bench_path.exists():
        prior = pd.read_csv(bench_path)
        pd.concat([prior, oot_df], ignore_index=True).to_csv(bench_path, index=False)
    else:
        oot_df.to_csv(bench_path, index=False)
    print(f"  Appended OOT rows to {bench_path.relative_to(PROJECT_ROOT)}")

    out_path = FIGURES_DIR / "roc_curves_comparison_v3.png"
    plt.figure(figsize=(9, 7))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Chance")
    for name in MODEL_ORDER:
        fpr, tpr, _ = roc_curve(y_oot, proba_store[name], pos_label=POSITIVE_LABEL)
        plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("Out-of-Time ROC — SME Credit Risk v3 (unweighted ROC-AUC)")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Wrote {out_path.relative_to(PROJECT_ROOT)}")
    return oot_df


def main() -> int:
    os.chdir(PROJECT_ROOT)
    print("SME Credit Risk v3 — unweighted ROC-AUC baseline")
    print(f"Project root: {PROJECT_ROOT}  GPU={USE_GPU} (xgb device={XGB_DEVICE})\n")
    print("Creating output directories if missing:")
    ensure_dirs()

    print("\nLoading chronological splits (NaNs still present):")
    X_train, y_train = load_xy("train")
    X_oot, y_oot = load_xy("oot")

    run_cross_validation(X_train, y_train)
    models, _, X_oot_p = run_final_fit(X_train, y_train, X_oot)
    run_oot_evaluation(models, X_oot_p, y_oot, X_oot)

    print("\nPipeline finished successfully.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nTraining pipeline failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1) from exc
