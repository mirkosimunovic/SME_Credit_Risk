"""Two-step SME credit-risk trainer (JRFM-oriented).

Uses the chronological, un-imputed splits from scripts/preprocess.py:

    data/processed/X_train.csv, y_train.csv
    data/processed/X_oot.csv,   y_oot.csv

STEP 1 — Stratified 5-fold CV on the training partition only.
          FastKNNImputer only exposes fit_transform (no sklearn fit/transform).
          Train-fold rows are imputed in isolation; val rows are stacked under
          that complete reference so neighbors come from train. Never fit on OOT.
          Metrics: Recall, AUC-ROC, AUC-PR, F1, Lift@10%.

STEP 2 — Impute 100% of X_train, persist that complete matrix as the imputer
          artifact (the fknni object cannot transform new rows). Fit scaler and
          champions on the imputed/scaled train set; serialize native artifacts.

STEP 3 — Score the chronological OOT holdout with those frozen artifacts
          and write a headless ROC comparison.

Target: MIS_Status (0 = Paid in Full, 1 = Default).
SMOTE is not used; class imbalance is handled with scale_pos_weight.
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
from fknni import FastKNNImputer
from sklearn.metrics import (
    auc,
    average_precision_score,
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

MODEL_ORDER = ["XGBoost", "LightGBM", "CatBoost"]

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


def ensure_dirs() -> None:
    for directory in (FIGURES_DIR, RESULTS_DIR, ARTIFACTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
        print(f"  [ok] {directory.relative_to(PROJECT_ROOT)}")


def load_xy(split: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load one chronological split. Target is always MIS_Status."""
    x_path = PROCESSED_DIR / f"X_{split}.csv"
    y_path = PROCESSED_DIR / f"y_{split}.csv"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"Missing {split} split. Run scripts/preprocess.py first.\n"
            f"  expected {x_path}\n  expected {y_path}"
        )
    X = pd.read_csv(x_path, low_memory=False)
    y = pd.read_csv(y_path, low_memory=False)
    if TARGET_COL not in y.columns:
        # single-column file with no header, or a different name
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
        # Already numeric (including columns that are float with NaNs).
        if pd.api.types.is_numeric_dtype(X_tr[col]):
            X_tr[col] = pd.to_numeric(X_tr[col], errors="coerce")
            X_ap[col] = pd.to_numeric(X_ap[col], errors="coerce")
            continue

        # Try a silent numeric cast first (e.g. '1.0' stored as object).
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
    """Fill apply-set NaNs using a complete training matrix as the neighbor pool.

    FastKNNImputer cannot transform new rows. We stack the already-imputed
    reference on top of the encoded apply set and run fit_transform. Reference
    rows have no NaNs, so they should stay fixed; apply rows borrow neighbors
    predominantly from that frozen train block.
    """
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
    """Serializable stand-in for FastKNNImputer, which has no transform().

    Stores the complete imputed training matrix plus the train-only encodings.
    New data is encoded with those maps, stacked under the reference, and
    imputed with a fresh fit_transform.
    """

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
    """Leak-proof numeric prep: encode -> impute train -> impute apply -> scale.

    FastKNNImputer has no fit/transform, so the wrapper imputes X_fit alone,
    then imputes X_apply against that complete reference.
    """
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


def scale_pos_weight(y: pd.Series | np.ndarray) -> float:
    """XGBoost / LightGBM / CatBoost convention: n_neg / n_pos.

    This is the inverse of the positive prevalence. Using n_pos / n_neg would
    under-weight the default class and collapse recall on a rare event.
    """
    y_arr = np.asarray(y)
    n_neg = int((y_arr == 0).sum())
    n_pos = int((y_arr == 1).sum())
    if n_pos == 0:
        raise ValueError("Training slice contains no default (class 1) labels.")
    return n_neg / n_pos


def build_models(spw: float) -> dict:
    return {
        "XGBoost": XGBClassifier(scale_pos_weight=spw, **XGB_BASELINE),
        "LightGBM": LGBMClassifier(scale_pos_weight=spw, **LGBM_BASELINE),
        "CatBoost": CatBoostClassifier(scale_pos_weight=spw, **CATBOOST_BASELINE),
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
        "recall": float(recall_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, y_proba)),
        "auc_pr": float(average_precision_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)),
        "lift_at_10": lift_at_fraction(y_true, y_proba, LIFT_FRACTION),
    }


def print_metrics(label: str, metrics: dict[str, float]) -> None:
    print(
        f"    {label:<22} "
        f"Recall={metrics['recall']:.4f}  "
        f"AUC-ROC={metrics['auc_roc']:.4f}  "
        f"AUC-PR={metrics['auc_pr']:.4f}  "
        f"F1={metrics['f1']:.4f}  "
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
    print("Imbalance: scale_pos_weight = n_neg / n_pos on the training fold.")
    print("OOT is held out of this entire loop.\n")

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    records: list[dict] = []

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y), start=1):
        X_tr, X_va = X.iloc[tr_idx].copy(), X.iloc[va_idx].copy()
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        spw = scale_pos_weight(y_tr)
        print(f"Fold {fold}/{N_SPLITS}  n_train={len(X_tr):,}  n_val={len(X_va):,}  scale_pos_weight={spw:.2f}")

        X_tr_p, X_va_p, *_ = impute_and_scale(X_tr, X_va)
        models = build_models(spw)
        for name in MODEL_ORDER:
            print(f"  Fitting {name} ...", flush=True)
            model = models[name]
            model.fit(X_tr_p, y_tr)
            y_proba = model.predict_proba(X_va_p)[:, 1]
            metrics = evaluate(y_va, y_proba)
            print_metrics(name, metrics)
            records.append({"stage": "cv", "fold": fold, "model": name, **metrics})
        print()

    fold_df = pd.DataFrame(records)
    metric_cols = ["recall", "auc_roc", "auc_pr", "f1", "lift_at_10"]

    print("=" * 78)
    print("MEAN ± STD across 5 stratified folds  (positive class = Default)")
    print("=" * 78)
    header = f"{'Model':<22}" + "".join(f"{m:>16}" for m in metric_cols)
    print(header)
    print("-" * len(header))

    summary_rows: list[dict] = []
    for name in MODEL_ORDER:
        subset = fold_df.loc[fold_df["model"] == name, metric_cols]
        means, stds = subset.mean(), subset.std(ddof=1)
        print(
            f"{name:<22}"
            + "".join(f"{means[m]:.4f}±{stds[m]:.4f}".rjust(16) for m in metric_cols)
        )
        summary_rows.append({"stage": "cv", "fold": "mean", "model": name, **means.to_dict()})
        summary_rows.append({"stage": "cv", "fold": "std", "model": name, **stds.to_dict()})

    print("=" * 78)
    print("Accuracy is omitted as a headline metric (rare-event default).")
    print("These CV numbers are the in-time performance claims. STEP 3 is OOT.\n")

    combined = pd.concat([fold_df, pd.DataFrame(summary_rows)], ignore_index=True)
    out_path = RESULTS_DIR / "metrics_benchmark.csv"
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

    imputer_path = ARTIFACTS_DIR / "imputer.joblib"
    scaler_path = ARTIFACTS_DIR / "scaler.joblib"
    # FastKNNImputer itself cannot transform new rows. Persist the wrapper that
    # holds the complete training matrix + encodings (call .transform on new X).
    joblib.dump(imputer, imputer_path)
    joblib.dump(scaler, scaler_path)
    print(f"  Saved {imputer_path.relative_to(PROJECT_ROOT)}  (StackedFastKNNImputer)")
    print(f"  Saved {scaler_path.relative_to(PROJECT_ROOT)}")
    print(f"  scaler.feature_names_in_ = {list(getattr(scaler, 'feature_names_in_', continuous_cols))}")

    spw = scale_pos_weight(y_train)
    print(f"  Full-train scale_pos_weight = {spw:.2f}")

    champions = build_models(spw)
    save_paths = {
        "XGBoost": ARTIFACTS_DIR / "xgboost_best.json",
        "CatBoost": ARTIFACTS_DIR / "catboost_best.bin",
        "LightGBM": ARTIFACTS_DIR / "lightgbm_best.txt",
    }
    fitted = {}
    for name in MODEL_ORDER:
        print(f"  Fitting final {name} on {len(X_train_p):,} training rows ...", flush=True)
        model = champions[name]
        model.fit(X_train_p, y_train)
        save_champion(name, model, save_paths[name])
        print(f"    Serialized -> {save_paths[name].relative_to(PROJECT_ROOT)}")
        fitted[name] = model

    return fitted, X_train_p, X_oot_p


# ===========================================================================
# STEP 3 — chronological OOT evaluation
# ===========================================================================
def run_oot_evaluation(models: dict, X_oot_p: pd.DataFrame, y_oot: pd.Series) -> pd.DataFrame:
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
        records.append({"stage": "oot", "fold": "oot", "model": name, **metrics})
        proba_store[name] = y_proba

    oot_df = pd.DataFrame(records)
    bench_path = RESULTS_DIR / "metrics_benchmark.csv"
    if bench_path.exists():
        prior = pd.read_csv(bench_path)
        pd.concat([prior, oot_df], ignore_index=True).to_csv(bench_path, index=False)
    else:
        oot_df.to_csv(bench_path, index=False)
    print(f"  Appended OOT rows to {bench_path.relative_to(PROJECT_ROOT)}")

    # Headless OOT ROC — one curve per champion.
    out_path = FIGURES_DIR / "roc_curves_comparison.png"
    plt.figure(figsize=(9, 7))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Chance")
    for name in MODEL_ORDER:
        fpr, tpr, _ = roc_curve(y_oot, proba_store[name], pos_label=POSITIVE_LABEL)
        plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("Out-of-Time ROC — SME Credit Risk (chronological holdout)")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Wrote {out_path.relative_to(PROJECT_ROOT)}")
    return oot_df


def main() -> int:
    os.chdir(PROJECT_ROOT)
    print("SME Credit Risk — two-step trainer + OOT evaluation")
    print(f"Project root: {PROJECT_ROOT}\n")
    print("Creating output directories if missing:")
    ensure_dirs()

    print("\nLoading chronological splits (NaNs still present):")
    X_train, y_train = load_xy("train")
    X_oot, y_oot = load_xy("oot")

    run_cross_validation(X_train, y_train)
    models, _, X_oot_p = run_final_fit(X_train, y_train, X_oot)
    run_oot_evaluation(models, X_oot_p, y_oot)

    print("\nPipeline finished successfully.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nTraining pipeline failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1) from exc
