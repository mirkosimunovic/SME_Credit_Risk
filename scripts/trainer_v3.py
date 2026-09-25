"""Two-step SME credit-risk trainer v3 (unweighted ROC-AUC baseline).

Uses chronological splits from scripts/preprocess_v3.py:

    data/processed/X_train_v3.csv, y_train_v3.csv
    data/processed/X_oot_v3.csv,   y_oot_v3.csv

Tuning objective is roc_auc. Champions are fit with unweighted binary
cross-entropy (no sample_weight, no scale_pos_weight). The OOT benchmark
selects a single champion; CatBoost is trained with native categoricals.

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
MISSING_CAT = "__MISSING__"

MODEL_ORDER = ["XGBoost", "LightGBM", "CatBoost"]
CHAMPION_METRIC = "auc_roc"
N_TUNE_ITER = 12
TUNE_CV = 2
TUNE_SUBSAMPLE = 40_000
BOOTSTRAP_ITERS = 1_000
VIF_WARN = 5.0
DEFAULT_RATE_WARN_PP = 0.02
NATIVE_EXT = {"XGBoost": ".json", "LightGBM": ".txt", "CatBoost": ".bin"}


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
        if int(pd.to_numeric(frame[col], errors="coerce").nunique(dropna=True))
        > CARDINALITY_THRESHOLD
    ]


def identify_categorical_columns(X_fit: pd.DataFrame) -> list[str]:
    """String leftovers plus low-cardinality columns (CatBoost native cats)."""
    cats: list[str] = []
    for col in X_fit.columns:
        series = X_fit[col]
        if not pd.api.types.is_numeric_dtype(series):
            as_num = pd.to_numeric(series, errors="coerce")
            if float(as_num.notna().mean()) < 0.95:
                cats.append(col)
                continue
            nunique = int(as_num.nunique(dropna=True))
        else:
            nunique = int(pd.to_numeric(series, errors="coerce").nunique(dropna=True))
        if nunique <= CARDINALITY_THRESHOLD:
            cats.append(col)
    return cats


def _as_cat_tokens(series: pd.Series) -> pd.Series:
    """Train-fold-safe tokenisation. Numeric cats become integer strings when whole."""
    if pd.api.types.is_numeric_dtype(series):
        num = pd.to_numeric(series, errors="coerce")
        out = pd.Series(pd.NA, index=series.index, dtype="string")
        ok = num.notna()
        if ok.any():
            vals = num.loc[ok].to_numpy(dtype=float)
            rounded = np.rint(vals)
            whole = np.isclose(vals, rounded, equal_nan=False)
            tokens = np.where(whole, rounded.astype(np.int64).astype(str), np.array([f"{v:.6g}" for v in vals]))
            out.loc[ok] = tokens
        return out
    tokens = series.where(series.notna(), np.nan).astype("string").str.strip()
    return tokens.replace({"": pd.NA, "<NA>": pd.NA, "nan": pd.NA, "None": pd.NA, "NaN": pd.NA})


def fit_category_modes(X_fit: pd.DataFrame, cat_cols: list[str]) -> dict[str, str]:
    """Mode per categorical column, learned on the training slice only."""
    modes: dict[str, str] = {}
    for col in cat_cols:
        tokens = _as_cat_tokens(X_fit[col]).dropna()
        modes[col] = str(tokens.mode().iloc[0]) if len(tokens) else MISSING_CAT
    return modes


def apply_category_columns(
    X: pd.DataFrame, cat_cols: list[str], modes: dict[str, str]
) -> pd.DataFrame:
    """Fill missing cats with train modes; leave unseen tokens as their own level."""
    out = pd.DataFrame(index=X.index)
    for col in cat_cols:
        if col not in X.columns:
            out[col] = modes.get(col, MISSING_CAT)
            continue
        tokens = _as_cat_tokens(X[col])
        mode = modes.get(col, MISSING_CAT)
        out[col] = tokens.fillna(mode).astype(str)
    return out


def prepare_catboost_frame(
    X_raw: pd.DataFrame,
    X_numeric: pd.DataFrame,
    cat_cols: list[str],
    modes: dict[str, str],
) -> pd.DataFrame:
    """Numeric/scaled features plus native string categoricals (mode-imputed)."""
    out = X_numeric.copy()
    if not cat_cols:
        return out
    cat_frame = apply_category_columns(X_raw.reindex(index=X_numeric.index), cat_cols, modes)
    for col in cat_cols:
        out[col] = cat_frame[col].astype(str)
    return out


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


def _knn_scale_stats(X: pd.DataFrame, continuous_cols: list[str]) -> dict:
    """Train-only mean/std for the KNN distance metric (NaNs ignored)."""
    stats: dict = {"cols": list(continuous_cols), "mean": {}, "scale": {}}
    for col in continuous_cols:
        if col not in X.columns:
            continue
        s = pd.to_numeric(X[col], errors="coerce")
        mu = float(s.mean())
        sd = float(s.std(ddof=0))
        stats["mean"][col] = 0.0 if not np.isfinite(mu) else mu
        stats["scale"][col] = 1.0 if (not np.isfinite(sd) or sd < 1e-12) else sd
    return stats


def _apply_knn_scale(X: pd.DataFrame, stats: dict | None, invert: bool = False) -> pd.DataFrame:
    if not stats or not stats.get("cols"):
        return X.copy()
    out = X.copy()
    for col in stats["cols"]:
        if col not in out.columns:
            continue
        mu = float(stats["mean"][col])
        sd = float(stats["scale"][col])
        vals = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float)
        out[col] = vals * sd + mu if invert else (vals - mu) / sd
    return out


def _print_knn_imputation_audit(X_before: pd.DataFrame, X_after: pd.DataFrame, continuous_cols: list[str]) -> dict:
    n = len(X_before)
    missing = X_before.isna().sum()
    rows = []
    print("    KNN imputation audit (continuous columns, raw units after invert):")
    for col in continuous_cols:
        n_miss = int(missing.get(col, 0))
        rate = n_miss / n if n else 0.0
        rec = {"column": col, "n_missing": n_miss, "missing_rate": rate}
        rows.append(rec)
        print(f"      {col:<24} missing={n_miss:,}  rate={rate:.4%}")
    audit = {"n_rows": int(n), "columns": rows}
    if not continuous_cols or missing.reindex(continuous_cols).fillna(0).sum() == 0:
        print("      No missing continuous cells; scaled KNN distance has no fill to compare.")
        return audit
    most = str(missing.reindex(continuous_cols).fillna(0).idxmax())
    mask = X_before[most].isna()
    observed = pd.to_numeric(X_before.loc[~mask, most], errors="coerce").dropna()
    imputed = pd.to_numeric(X_after.loc[mask, most], errors="coerce").dropna()
    print(f"    Sanity check — most-missing continuous column: {most}")
    if len(observed) and len(imputed):
        print(
            f"      observed n={len(observed):,} mean={float(observed.mean()):.4f} "
            f"median={float(observed.median()):.4f}"
        )
        print(
            f"      imputed  n={len(imputed):,} mean={float(imputed.mean()):.4f} "
            f"median={float(imputed.median()):.4f}"
        )
        audit["sanity_column"] = most
        audit["observed_mean"] = float(observed.mean())
        audit["observed_median"] = float(observed.median())
        audit["imputed_mean"] = float(imputed.mean())
        audit["imputed_median"] = float(imputed.median())
        audit["n_imputed"] = int(len(imputed))
    return audit


def impute_training_slice(
    X_enc: pd.DataFrame, knn_stats: dict | None = None, audit: bool = False
) -> pd.DataFrame:
    """Impute the training slice in isolation (no validation / OOT rows)."""
    print(
        f"    FastKNNImputer.fit_transform on training slice ({len(X_enc):,} rows) ...",
        flush=True,
    )
    Xs = _apply_knn_scale(X_enc, knn_stats, invert=False)
    arr = _knn_fit_transform(_to_numpy(Xs))
    imputed = pd.DataFrame(arr, columns=X_enc.columns, index=X_enc.index)
    imputed = _apply_knn_scale(imputed, knn_stats, invert=True)
    if audit and knn_stats:
        _print_knn_imputation_audit(X_enc, imputed, list(knn_stats.get("cols", [])))
    return imputed


def impute_from_reference(
    X_ref_imp: pd.DataFrame,
    X_apply_enc: pd.DataFrame,
    knn_stats: dict | None = None,
) -> pd.DataFrame:
    """Fill apply-set NaNs using a complete training matrix as the neighbor pool."""
    n_ref = len(X_ref_imp)
    X_apply_aligned = X_apply_enc.reindex(columns=list(X_ref_imp.columns))
    ref_s = _apply_knn_scale(X_ref_imp, knn_stats, invert=False)
    apply_s = _apply_knn_scale(X_apply_aligned, knn_stats, invert=False)
    stacked = np.vstack([_to_numpy(ref_s), _to_numpy(apply_s)])
    print(
        f"    FastKNNImputer.fit_transform on stacked reference+apply "
        f"({stacked.shape[0]:,} rows) ...",
        flush=True,
    )
    stacked_imp = _knn_fit_transform(stacked)
    apply_imp = pd.DataFrame(
        stacked_imp[n_ref:],
        columns=X_ref_imp.columns,
        index=X_apply_enc.index,
    )
    return _apply_knn_scale(apply_imp, knn_stats, invert=True)


class StackedFastKNNImputer:
    """Serializable stand-in for FastKNNImputer, which has no transform()."""

    def __init__(self, n_neighbors: int = 5, strategy: str = "mean"):
        self.n_neighbors = n_neighbors
        self.strategy = strategy
        self.reference_: pd.DataFrame | None = None
        self.encodings_: dict = {}
        self.feature_columns_: list[str] = []
        self.knn_scale_stats_: dict | None = None
        self.cat_columns_: list[str] = []
        self.cat_modes_: dict[str, str] = {}

    def fit_reference(
        self,
        X_fit_enc: pd.DataFrame,
        encodings: dict,
        audit: bool = False,
    ) -> pd.DataFrame:
        self.encodings_ = encodings
        self.feature_columns_ = list(X_fit_enc.columns)
        continuous = identify_continuous(X_fit_enc)
        self.knn_scale_stats_ = _knn_scale_stats(X_fit_enc, continuous)
        self.reference_ = impute_training_slice(
            X_fit_enc, self.knn_scale_stats_, audit=audit
        )
        return self.reference_

    def transform(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        if self.reference_ is None:
            raise RuntimeError("StackedFastKNNImputer has no training reference.")
        X_enc = encode_with_mapping(X_raw, self.encodings_, self.feature_columns_)
        return impute_from_reference(self.reference_, X_enc, self.knn_scale_stats_)

    def transform_encoded(self, X_enc: pd.DataFrame) -> pd.DataFrame:
        if self.reference_ is None:
            raise RuntimeError("StackedFastKNNImputer has no training reference.")
        return impute_from_reference(self.reference_, X_enc, self.knn_scale_stats_)


def impute_and_scale(
    X_fit: pd.DataFrame,
    X_apply: pd.DataFrame,
    imputer: StackedFastKNNImputer | None = None,
    scaler: StandardScaler | None = None,
    continuous_cols: list[str] | None = None,
    audit: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, StackedFastKNNImputer, StandardScaler, list[str]]:
    """Leak-proof numeric prep: encode -> scaled-KNN impute train -> impute apply -> scale."""
    X_fit_enc, X_apply_enc, encodings = encode_non_numeric(X_fit, X_apply)

    if imputer is None or imputer.reference_ is None:
        imputer = StackedFastKNNImputer()
        X_fit_imp = imputer.fit_reference(X_fit_enc, encodings, audit=audit)
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


def tune_model(name: str, X: pd.DataFrame, y: pd.Series, cat_features: list[str] | None = None) -> dict:
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
        kwargs = dict(cat_fixed())
        if cat_features:
            kwargs["cat_features"] = list(cat_features)
        est = CatBoostClassifier(**kwargs)
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


def instantiate_tuned(
    name: str,
    params: dict,
    seed: int = RANDOM_STATE,
    cat_features: list[str] | None = None,
):
    if name == "XGBoost":
        return XGBClassifier(**xgb_fixed(seed), **params)
    if name == "LightGBM":
        return LGBMClassifier(**lgbm_fixed(seed), **params)
    kwargs = dict(cat_fixed(seed))
    if cat_features:
        kwargs["cat_features"] = list(cat_features)
    return CatBoostClassifier(**{**kwargs, **params})


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


def _model_frame(name: str, X_raw: pd.DataFrame, X_num: pd.DataFrame, cat_cols, cat_modes):
    if name == "CatBoost":
        return prepare_catboost_frame(X_raw, X_num, cat_cols, cat_modes)
    return X_num


def select_champion(bench_path: Path, metric: str = CHAMPION_METRIC) -> tuple[str, pd.DataFrame]:
    """Pick the best model on OOT `metric`, falling back to CV-mean if OOT is absent."""
    if not bench_path.exists():
        raise FileNotFoundError(f"Missing benchmark CSV: {bench_path}")
    df = pd.read_csv(bench_path)
    if metric not in df.columns:
        raise KeyError(f"{bench_path.name} has no column '{metric}'.")
    oot = df[(df["stage"].astype(str) == "oot") & (df["fold"].astype(str) == "oot")].copy()
    if len(oot):
        table = oot
        source = "OOT"
    else:
        table = df[(df["stage"].astype(str) == "cv") & (df["fold"].astype(str) == "mean")].copy()
        source = "CV-mean"
        if table.empty:
            raise ValueError(f"No OOT or CV-mean rows in {bench_path.name}.")
    table = table.drop_duplicates(subset=["model"], keep="last")
    table = table.sort_values(metric, ascending=False)
    winner = str(table.iloc[0]["model"])
    win_score = float(table.iloc[0][metric])
    runner_up = str(table.iloc[1]["model"]) if len(table) > 1 else None
    margin = float(win_score - float(table.iloc[1][metric])) if runner_up else float("nan")
    print("\n" + "=" * 78)
    print(f"CHAMPION SELECTION  source={source}  metric={metric}")
    print("=" * 78)
    show_cols = [c for c in ["model", "auc_roc", "auc_pr", "f1", "recall", "balanced_accuracy"] if c in table.columns]
    print(table[show_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    extra = f"  margin vs {runner_up} = {margin:+.4f}" if runner_up else ""
    print(f"  Winner: {winner}  {metric}={win_score:.4f}{extra}")
    return winner, table


def export_champion_artifacts(
    winner: str,
    fitted: dict,
    train_frames: dict[str, pd.DataFrame],
    y_train: pd.Series,
    best_params: dict[str, dict],
    comparison: pd.DataFrame,
    cat_cols: list[str],
) -> dict:
    """Write model-agnostic raw + calibrated champion files plus metadata."""
    raw_model = fitted[winner]
    ext = NATIVE_EXT[winner]
    raw_path = ARTIFACTS_DIR / f"champion_raw_v3{ext}"
    cal_path = ARTIFACTS_DIR / "champion_calibrated_v3.joblib"
    meta_path = ARTIFACTS_DIR / "champion_metadata_v3.json"
    save_champion(winner, raw_model, raw_path)
    X_train_m = train_frames[winner]
    cats = list(cat_cols) if winner == "CatBoost" else None
    print(f"  Fitting CalibratedClassifierCV on champion={winner} (train only, cv=5) ...")
    cal = CalibratedClassifierCV(
        estimator=instantiate_tuned(winner, best_params[winner], cat_features=cats),
        method="isotonic",
        cv=5,
    )
    cal.fit(X_train_m, y_train)
    joblib.dump(cal, cal_path)
    ranked = comparison.sort_values(CHAMPION_METRIC, ascending=False)
    win_score = float(ranked.iloc[0][CHAMPION_METRIC])
    runner = None
    margin = None
    if len(ranked) > 1:
        runner = str(ranked.iloc[1]["model"])
        margin = float(win_score - float(ranked.iloc[1][CHAMPION_METRIC]))
    meta = {
        "model": winner,
        "champion_metric": CHAMPION_METRIC,
        "champion_metric_value": win_score,
        "runner_up": runner,
        "margin_over_runner_up": margin,
        "raw_path": str(raw_path.relative_to(PROJECT_ROOT)),
        "calibrated_path": str(cal_path.relative_to(PROJECT_ROOT)),
        "native_format": ext.lstrip("."),
        "cat_features": list(cat_cols) if winner == "CatBoost" else [],
        "comparison": ranked[["model", CHAMPION_METRIC]].to_dict(orient="records"),
    }
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"  Saved {raw_path.relative_to(PROJECT_ROOT)}")
    print(f"  Saved {cal_path.relative_to(PROJECT_ROOT)}")
    print(f"  Saved {meta_path.relative_to(PROJECT_ROOT)}")
    return meta


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
        cat_cols = identify_categorical_columns(X_tr)
        cat_modes = fit_category_modes(X_tr, cat_cols)
        print(f"    CatBoost cat_features ({len(cat_cols)}): {cat_cols}")
        report_vif(X_tr_p, continuous_cols, label=f"fold {fold} train")
        for name in MODEL_ORDER:
            print(f"  Fitting {name} ...", flush=True)
            X_tr_m = _model_frame(name, X_tr, X_tr_p, cat_cols, cat_modes)
            X_va_m = _model_frame(name, X_va, X_va_p, cat_cols, cat_modes)
            cats = cat_cols if name == "CatBoost" else None
            best = tune_model(name, X_tr_m, y_tr, cat_features=cats)
            model = instantiate_tuned(name, best, cat_features=cats)
            model.fit(X_tr_m, y_tr)
            y_proba = model.predict_proba(X_va_m)[:, 1]
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
) -> tuple[dict, dict, dict, dict, list[str]]:
    print("\n" + "=" * 78)
    print("STEP 2  Global imputer / scaler / champion fit on 100% of X_train")
    print("=" * 78)
    print("OOT is imputed against the frozen imputed-train reference. It is not used to fit it.\n")

    X_train_p, X_oot_p, imputer, scaler, continuous_cols = impute_and_scale(
        X_train, X_oot, audit=True
    )
    cat_cols = identify_categorical_columns(X_train)
    cat_modes = fit_category_modes(X_train, cat_cols)
    imputer.cat_columns_ = list(cat_cols)
    imputer.cat_modes_ = dict(cat_modes)
    print(f"  CatBoost cat_features ({len(cat_cols)}): {cat_cols}")
    report_vif(X_train_p, continuous_cols, label="final train")

    imputer_path = ARTIFACTS_DIR / "imputer_v3.joblib"
    scaler_path = ARTIFACTS_DIR / "scaler_v3.joblib"
    joblib.dump(imputer, imputer_path)
    joblib.dump(scaler, scaler_path)
    print(f"  Saved {imputer_path.relative_to(PROJECT_ROOT)}  (StackedFastKNNImputer)")
    print(f"  Saved {scaler_path.relative_to(PROJECT_ROOT)}")
    print(f"  scaler.feature_names_in_ = {list(getattr(scaler, 'feature_names_in_', continuous_cols))}")

    train_frames = {name: _model_frame(name, X_train, X_train_p, cat_cols, cat_modes) for name in MODEL_ORDER}
    oot_frames = {name: _model_frame(name, X_oot, X_oot_p, cat_cols, cat_modes) for name in MODEL_ORDER}

    best_params: dict[str, dict] = {}
    save_paths = {
        "XGBoost": ARTIFACTS_DIR / "xgboost_best_v3.json",
        "CatBoost": ARTIFACTS_DIR / "catboost_best_v3.bin",
        "LightGBM": ARTIFACTS_DIR / "lightgbm_best_v3.txt",
    }
    fitted = {}
    for name in MODEL_ORDER:
        print(f"  Tuning + fitting final {name} on {len(train_frames[name]):,} training rows ...", flush=True)
        cats = cat_cols if name == "CatBoost" else None
        best = tune_model(name, train_frames[name], y_train, cat_features=cats)
        best_params[name] = best
        model = instantiate_tuned(name, best, cat_features=cats)
        model.fit(train_frames[name], y_train)
        save_champion(name, model, save_paths[name])
        print(f"    Serialized -> {save_paths[name].relative_to(PROJECT_ROOT)}")
        fitted[name] = model

    hp_path = RESULTS_DIR / "best_hyperparams_v3.json"
    with open(hp_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "gpu": USE_GPU,
                "tuning_objective": "roc_auc",
                "champion_metric": CHAMPION_METRIC,
                "cat_features": cat_cols,
                "params": best_params,
            },
            fh,
            indent=2,
        )
    print(f"  Saved {hp_path.relative_to(PROJECT_ROOT)}")
    return fitted, train_frames, oot_frames, best_params, cat_cols


# ===========================================================================
# STEP 3 — chronological OOT evaluation
# ===========================================================================
def run_oot_evaluation(
    models: dict, oot_frames: dict[str, pd.DataFrame], y_oot: pd.Series
) -> pd.DataFrame:
    print("\n" + "=" * 78)
    print("STEP 3  Out-of-time evaluation on X_oot / y_oot")
    print("=" * 78)
    print("This split is newer than every training row. It is the paper's temporal test.\n")

    records: list[dict] = []
    proba_store: dict[str, np.ndarray] = {}
    for name in MODEL_ORDER:
        X_oot_m = oot_frames[name]
        print(f"  Scoring {name} on OOT ({len(X_oot_m):,} rows) ...", flush=True)
        y_proba = models[name].predict_proba(X_oot_m)[:, 1]
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
    models, train_frames, oot_frames, best_params, cat_cols = run_final_fit(
        X_train, y_train, X_oot
    )
    run_oot_evaluation(models, oot_frames, y_oot)
    winner, comparison = select_champion(RESULTS_DIR / "metrics_benchmark_v3.csv")
    export_champion_artifacts(
        winner, models, train_frames, y_train, best_params, comparison, cat_cols
    )

    print("\nPipeline finished successfully.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nTraining pipeline failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1) from exc
