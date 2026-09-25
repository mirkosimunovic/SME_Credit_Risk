"""Week 3 v3 — SHAP + Kendall's W for the unweighted ROC-AUC champion.

Loads frozen trainer_v3 artifacts (StackedFastKNNImputer + StandardScaler),
explains a stratified OOT subsample, then retrains the selected champion
under 5 seeds without sample weights.

Target: MIS_Status (0 = Paid in Full, 1 = Default).
"""

from __future__ import annotations

import json
import os
import pickle
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
import shap
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
ARTIFACTS_DIR = PROJECT_ROOT / "models" / "artifacts"

TARGET_COL = "MIS_Status"
SHAP_SAMPLE_SIZE = 5_000
CARDINALITY_THRESHOLD = 10
STABILITY_SEEDS = [42, 7, 13, 21, 99]
TOP_K = 10
MISSING_CAT = "__MISSING__"


# ===========================================================================
# 1. Custom imputer namespace (required for joblib to unpickle the wrapper)
# ===========================================================================
# trainer_v3.py dumped a StackedFastKNNImputer. Pickle restores attributes
# (reference_, encodings_, …) onto THIS class. Methods always come from the
# loading module, so they cannot be `pass` — transform must actually impute.
def _to_numpy(frame: pd.DataFrame) -> np.ndarray:
    return frame.to_numpy(dtype=np.float64, copy=True)


def encode_with_mapping(
    X: pd.DataFrame, encodings: dict, feature_columns: list[str]
) -> pd.DataFrame:
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


def impute_from_reference(
    X_ref_imp: pd.DataFrame,
    X_apply_enc: pd.DataFrame,
    knn_stats: dict | None = None,
) -> pd.DataFrame:
    """Fill new-row NaNs using the frozen complete training matrix as neighbors."""
    from fknni import FastKNNImputer

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
    imputed = FastKNNImputer(n_neighbors=5, strategy="mean").fit_transform(stacked)
    if hasattr(imputed, "get"):
        imputed = imputed.get()
    stacked_imp = np.asarray(imputed, dtype=np.float64)
    apply_imp = pd.DataFrame(
        stacked_imp[n_ref:],
        columns=X_ref_imp.columns,
        index=X_apply_enc.index,
    )
    return _apply_knn_scale(apply_imp, knn_stats, invert=True)


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


def _as_cat_tokens(series: pd.Series) -> pd.Series:
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


def apply_category_columns(X: pd.DataFrame, cat_cols: list[str], modes: dict[str, str]) -> pd.DataFrame:
    out = pd.DataFrame(index=X.index)
    for col in cat_cols:
        if col not in X.columns:
            out[col] = modes.get(col, MISSING_CAT)
            continue
        tokens = _as_cat_tokens(X[col])
        out[col] = tokens.fillna(modes.get(col, MISSING_CAT)).astype(str)
    return out


def prepare_catboost_frame(X_raw, X_numeric, cat_cols, modes) -> pd.DataFrame:
    out = X_numeric.copy()
    if not cat_cols:
        return out
    cat_frame = apply_category_columns(X_raw.reindex(index=X_numeric.index), cat_cols, modes)
    for col in cat_cols:
        out[col] = cat_frame[col].astype(str)
    return out


class StackedFastKNNImputer:
    def __init__(self, n_neighbors: int = 5, strategy: str = "mean"):
        self.n_neighbors = n_neighbors
        self.strategy = strategy
        self.reference_: pd.DataFrame | None = None
        self.encodings_: dict = {}
        self.feature_columns_: list[str] = []
        self.knn_scale_stats_: dict | None = None
        self.cat_columns_: list[str] = []
        self.cat_modes_: dict = {}

    def fit_reference(self, X_fit_enc, encodings):
        # Training already happened in trainer_v3.py. The pickled object carries
        # reference_ / encodings_. Re-fitting here would rebuild the neighbor pool.
        self.encodings_ = encodings
        self.feature_columns_ = list(X_fit_enc.columns)
        return self.reference_

    def transform(self, X_raw):
        if self.reference_ is None:
            raise RuntimeError("StackedFastKNNImputer has no training reference.")
        X_enc = encode_with_mapping(X_raw, self.encodings_, self.feature_columns_)
        return impute_from_reference(self.reference_, X_enc, getattr(self, "knn_scale_stats_", None))

    def transform_encoded(self, X_enc):
        if self.reference_ is None:
            raise RuntimeError("StackedFastKNNImputer has no training reference.")
        return impute_from_reference(self.reference_, X_enc, getattr(self, "knn_scale_stats_", None))


def _register_imputer_for_unpickle() -> None:
    """Map pickle module names so joblib can find StackedFastKNNImputer."""
    this = sys.modules[__name__]
    sys.modules.setdefault("trainer_v3", this)
    sys.modules.setdefault("scripts.trainer_v3", this)
    sys.modules.setdefault("trainer_v2", this)
    sys.modules.setdefault("scripts.trainer_v2", this)
    sys.modules.setdefault("trainer", this)
    sys.modules.setdefault("scripts.trainer", this)
    sys.modules.setdefault("__main__", this)


def load_imputer(path: Path) -> StackedFastKNNImputer:
    _register_imputer_for_unpickle()
    try:
        obj = joblib.load(path)
    except Exception as exc:
        print(f"  joblib.load failed ({exc}); retrying with class remap ...")

        class _RemapUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if name == "StackedFastKNNImputer":
                    return StackedFastKNNImputer
                return super().find_class(module, name)

        with open(path, "rb") as fh:
            obj = _RemapUnpickler(fh).load()
    if not hasattr(obj, "reference_") or obj.reference_ is None:
        raise RuntimeError(f"{path} did not contain a fitted StackedFastKNNImputer.reference_.")
    return obj


# ===========================================================================
# Helpers
# ===========================================================================
def load_champion_metadata() -> dict:
    meta_path = ARTIFACTS_DIR / "champion_metadata_v3.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            "Missing models/artifacts/champion_metadata_v3.json. Run scripts/trainer_v3.py first."
        )
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    print(f"  Champion metadata: model={meta.get('model')}  metric={meta.get('champion_metric')}")
    return meta


def load_tuned_params(model_name: str) -> dict:
    hp_path = RESULTS_DIR / "best_hyperparams_v3.json"
    if not hp_path.exists():
        raise FileNotFoundError(
            "Missing outputs/results/best_hyperparams_v3.json. Run scripts/trainer_v3.py first."
        )
    with open(hp_path, encoding="utf-8") as fh:
        payload = json.load(fh)
    params = payload.get("params", {}).get(model_name)
    if not isinstance(params, dict) or not params:
        raise KeyError(f"best_hyperparams_v3.json has no params.{model_name} block.")
    params = dict(params)
    params.pop("random_state", None)
    params.pop("scale_pos_weight", None)
    print(f"  Loaded tuned {model_name} params from {hp_path.relative_to(PROJECT_ROOT)}")
    return params


def instantiate_stability_model(name: str, params: dict, seed: int, cat_features: list[str] | None):
    if name == "XGBoost":
        return XGBClassifier(
            random_state=seed,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            verbosity=0,
            **params,
        )
    if name == "LightGBM":
        return LGBMClassifier(random_state=seed, verbosity=-1, n_jobs=-1, **params)
    kwargs = dict(params)
    if cat_features:
        kwargs["cat_features"] = list(cat_features)
    return CatBoostClassifier(
        random_seed=seed, verbose=False, allow_writing_files=False, **kwargs
    )


def load_raw_champion(meta: dict):
    name = str(meta["model"])
    path = PROJECT_ROOT / meta["raw_path"]
    if not path.exists():
        raise FileNotFoundError(f"Missing champion raw artifact: {path}")
    print(f"  Loading raw champion {path.relative_to(PROJECT_ROOT)} ({name})")
    if name == "XGBoost":
        model = XGBClassifier()
        model.load_model(str(path))
        return model
    if name == "LightGBM":
        import lightgbm as lgb

        return lgb.Booster(model_file=str(path))
    model = CatBoostClassifier()
    model.load_model(str(path))
    return model


def load_xy(split: str) -> tuple[pd.DataFrame, pd.Series]:
    x_path = PROCESSED_DIR / f"X_{split}_v3.csv"
    y_path = PROCESSED_DIR / f"y_{split}_v3.csv"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Missing {split} v3 split under {PROCESSED_DIR}")
    X = pd.read_csv(x_path, low_memory=False)
    y = pd.read_csv(y_path, low_memory=False)
    y = y[TARGET_COL] if TARGET_COL in y.columns else y.iloc[:, 0]
    y = y.astype(int)
    print(f"  Loaded {split}: X={X.shape}  default rate={float(y.mean()):.6f}")
    return X, y


def apply_frozen_scaler(X_imp: pd.DataFrame, scaler) -> pd.DataFrame:
    """Scale the same continuous columns the trainer fitted on (feature_names_in_)."""
    X_out = X_imp.copy()
    if hasattr(scaler, "feature_names_in_"):
        cols = [c for c in scaler.feature_names_in_ if c in X_out.columns]
    else:
        cols = [
            c
            for c in X_out.columns
            if int(X_out[c].nunique(dropna=False)) > CARDINALITY_THRESHOLD
        ]
    print(f"    Scaling {len(cols)} continuous columns: {cols}")
    if cols:
        X_out[cols] = scaler.transform(X_out[cols])
    return X_out


def shap_matrix(shap_out, n_rows: int, n_features: int) -> np.ndarray:
    """Normalize TreeExplainer output to (n_samples, n_features) for class 1."""
    if hasattr(shap_out, "values"):
        vals = np.asarray(shap_out.values)
    elif isinstance(shap_out, list):
        vals = np.asarray(shap_out[1] if len(shap_out) == 2 else shap_out[-1])
    else:
        vals = np.asarray(shap_out)
    if vals.ndim == 3:
        vals = vals[:, :, -1]
    if vals.shape != (n_rows, n_features):
        raise ValueError(f"Unexpected SHAP shape {vals.shape}; expected {(n_rows, n_features)}")
    return vals


def kendalls_w(rank_matrix: np.ndarray) -> dict:
    """Kendall's coefficient of concordance W with tie correction.

    rank_matrix: (n_items, n_raters). Rank 1 = most important.
    W = 1 → identical rankings; W = 0 → no concordance.
    """
    ranks = np.asarray(rank_matrix, dtype=float)
    n_items, n_raters = ranks.shape
    if n_items < 2 or n_raters < 2:
        raise ValueError("Kendall's W needs at least 2 items and 2 raters.")

    row_sums = ranks.sum(axis=1)
    s_stat = float(np.sum((row_sums - row_sums.mean()) ** 2))

    tie_term = 0.0
    for j in range(n_raters):
        _, counts = np.unique(ranks[:, j], return_counts=True)
        tie_term += float(np.sum(counts**3 - counts))

    denom = (n_raters**2) * (n_items**3 - n_items) - n_raters * tie_term
    if denom <= 0:
        w = 1.0 if s_stat == 0 else 0.0
    else:
        w = 12.0 * s_stat / denom

    chi2 = n_raters * (n_items - 1) * w
    df = n_items - 1
    return {
        "kendall_w": float(w),
        "friedman_chi2": float(chi2),
        "df": int(df),
        "n_items": int(n_items),
        "n_raters": int(n_raters),
        "S": s_stat,
        "tie_correction_T": tie_term,
    }


def stratified_oot_sample(
    X: pd.DataFrame, y: pd.Series, n: int, seed: int = 42
) -> tuple[pd.DataFrame, pd.Series]:
    n = min(n, len(X))
    if n == len(X):
        return X.copy(), y.copy()
    X_s, _, y_s, _ = train_test_split(
        X, y, train_size=n, stratify=y, random_state=seed
    )
    return X_s.reset_index(drop=True), y_s.reset_index(drop=True)


# ===========================================================================
# Pipeline
# ===========================================================================
def main() -> int:
    os.chdir(PROJECT_ROOT)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("Week 3 v3  SHAP + Kendall's W (ROC-AUC champion)")
    print("=" * 78)

    # -----------------------------------------------------------------------
    # 2. Load frozen assets and transform OOT (and train) consistently
    # -----------------------------------------------------------------------
    print("\n[2] Loading frozen imputer / scaler and chronological splits")
    meta = load_champion_metadata()
    model_name = str(meta["model"])
    tuned_params = load_tuned_params(model_name)
    cat_features = list(meta.get("cat_features") or [])
    imputer_path = ARTIFACTS_DIR / "imputer_v3.joblib"
    scaler_path = ARTIFACTS_DIR / "scaler_v3.joblib"
    raw_path = PROJECT_ROOT / meta["raw_path"]
    for p in (imputer_path, scaler_path, raw_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing artifact: {p}")

    print(f"  Loading {imputer_path.relative_to(PROJECT_ROOT)} ...")
    imputer = load_imputer(imputer_path)
    print(
        f"  Imputer reference shape={imputer.reference_.shape}  "
        f"encoded columns={len(imputer.feature_columns_)}"
    )
    print(f"  Loading {scaler_path.relative_to(PROJECT_ROOT)} ...")
    scaler = joblib.load(scaler_path)
    cat_cols = list(getattr(imputer, "cat_columns_", None) or cat_features)
    cat_modes = dict(getattr(imputer, "cat_modes_", None) or {})

    print("  Loading OOT features / targets ...")
    X_oot_raw, y_oot = load_xy("oot")
    print("  Transforming OOT (impute against frozen train reference, then scale) ...")
    X_oot_imp = imputer.transform(X_oot_raw)
    X_oot_p = apply_frozen_scaler(X_oot_imp, scaler)

    print("  Loading training features / targets ...")
    X_train_raw, y_train = load_xy("train")
    print("  Using imputer.reference_ as imputed X_train (champion training matrix) ...")
    X_train_imp = imputer.reference_.copy()
    if list(X_train_imp.columns) != list(X_oot_p.columns):
        X_train_imp = X_train_imp.reindex(columns=X_oot_p.columns)
    X_train_p = apply_frozen_scaler(X_train_imp, scaler)
    if len(X_train_p) != len(y_train):
        print(
            "  Warning: imputed-train rows != y_train length. "
            "Falling back to imputer.transform(X_train)."
        )
        print("  Transforming train ...")
        X_train_imp = imputer.transform(X_train_raw)
        X_train_p = apply_frozen_scaler(X_train_imp, scaler)

    if model_name == "CatBoost":
        X_oot_m = prepare_catboost_frame(X_oot_raw, X_oot_p, cat_cols, cat_modes)
        X_train_m = prepare_catboost_frame(X_train_raw, X_train_p, cat_cols, cat_modes)
    else:
        X_oot_m, X_train_m = X_oot_p, X_train_p

    print("  Stability retraining is unweighted (ROC-AUC champion; no sample_weight).")

    # -----------------------------------------------------------------------
    # 3. Champion SHAP on a stratified 5,000-row OOT sample
    # -----------------------------------------------------------------------
    print("\n[3] Champion SHAP (beeswarm + mean-|SHAP| bar) on stratified OOT sample")
    print(f"  Drawing stratified sample of {SHAP_SAMPLE_SIZE:,} OOT rows ...")
    X_shap, y_shap = stratified_oot_sample(X_oot_m, y_oot, SHAP_SAMPLE_SIZE, seed=42)
    print(f"  Sample shape={X_shap.shape}  sample default rate={float(y_shap.mean()):.6f}")

    champion = load_raw_champion(meta)

    print("  Initializing shap.TreeExplainer on the champion ...")
    explainer = shap.TreeExplainer(champion)
    print("  Computing SHAP values for the OOT sample ...")
    shap_raw = explainer.shap_values(X_shap)
    shap_vals = shap_matrix(shap_raw, n_rows=len(X_shap), n_features=X_shap.shape[1])
    mean_abs = np.abs(shap_vals).mean(axis=0)
    feature_names = list(X_shap.columns)
    order = np.argsort(-mean_abs)
    top_features = [feature_names[i] for i in order[:TOP_K]]
    print("  Top-10 features by mean |SHAP| (champion):")
    for rank, idx in enumerate(order[:TOP_K], start=1):
        print(f"    {rank:2d}. {feature_names[idx]:<24}  mean|SHAP|={mean_abs[idx]:.6f}")

    beeswarm_path = FIGURES_DIR / "shap_summary_v3.png"
    print(f"  Writing beeswarm -> {beeswarm_path.relative_to(PROJECT_ROOT)}")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_vals, X_shap, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(beeswarm_path, dpi=150, bbox_inches="tight")
    plt.close("all")

    bar_path = FIGURES_DIR / "shap_importance_v3.png"
    print(f"  Writing importance bar -> {bar_path.relative_to(PROJECT_ROOT)}")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_vals, X_shap, plot_type="bar", show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close("all")

    # -----------------------------------------------------------------------
    # 4. Stability: 5 seeds, same OOT sample, Kendall's W on top-10 ranks
    # -----------------------------------------------------------------------
    print("\n[4] Explanation stability — retrain champion under 5 seeds")
    print(f"  Algorithm: {model_name}")
    print(f"  Seeds: {STABILITY_SEEDS}")
    print("  Hyperparameters match trainer_v3; unweighted fit; only random_state changes.")

    ranks_global = []  # each row: ranks of top_features among ALL features (1 = highest |SHAP|)
    ranks_within = []  # ranks among the top-10 only
    mean_abs_by_seed: dict[str, dict[str, float]] = {}

    for i, seed in enumerate(STABILITY_SEEDS, start=1):
        print(f"  Computing SHAP for Seed {i}/{len(STABILITY_SEEDS)} (random_state={seed}) ...")
        model = instantiate_stability_model(
            model_name, tuned_params, seed, cat_cols if model_name == "CatBoost" else None
        )
        model.fit(X_train_m, y_train)
        seed_explainer = shap.TreeExplainer(model)
        seed_shap = shap_matrix(
            seed_explainer.shap_values(X_shap),
            n_rows=len(X_shap),
            n_features=X_shap.shape[1],
        )
        seed_mean_abs = np.abs(seed_shap).mean(axis=0)
        global_rank = pd.Series(seed_mean_abs, index=feature_names).rank(
            ascending=False, method="average"
        )
        ranks_global.append([float(global_rank[f]) for f in top_features])
        within = pd.Series(seed_mean_abs, index=feature_names).loc[top_features]
        within_rank = within.rank(ascending=False, method="average")
        ranks_within.append(within_rank.to_numpy(dtype=float))
        mean_abs_by_seed[str(seed)] = {
            f: float(seed_mean_abs[feature_names.index(f)]) for f in top_features
        }
        print(
            f"    Seed {seed} top-3: "
            + ", ".join(
                f"{f} (rank {global_rank[f]:.1f})"
                for f in pd.Series(seed_mean_abs, index=feature_names)
                .sort_values(ascending=False)
                .head(3)
                .index
            )
        )

    print("  Calculating Kendall's W ...")
    rank_mat = np.column_stack(ranks_within)  # (10 features, 5 seeds)
    w_stats = kendalls_w(rank_mat)
    rank_var = {
        feat: float(np.var(rank_mat[i, :], ddof=1))
        for i, feat in enumerate(top_features)
    }
    mean_rank = {
        feat: float(np.mean(rank_mat[i, :]))
        for i, feat in enumerate(top_features)
    }

    report = {
        "champion_model": meta.get("raw_path"),
        "champion_algorithm": model_name,
        "champion_hyperparams": dict(tuned_params),
        "stability_fit": "unweighted (no sample_weight)",
        "shap_sample_size": int(len(X_shap)),
        "shap_sample_default_rate": float(y_shap.mean()),
        "seeds": STABILITY_SEEDS,
        "top_10_features_champion": top_features,
        "champion_mean_abs_shap": {
            feature_names[i]: float(mean_abs[i]) for i in order[:TOP_K]
        },
        "kendall_w": w_stats["kendall_w"],
        "kendall_w_formula": (
            "W = 12S / (m^2 (n^3 - n) - m T), "
            "n=10 features, m=5 seeds, T=tie correction"
        ),
        "friedman_chi2": w_stats["friedman_chi2"],
        "df": w_stats["df"],
        "rank_variance_top10": rank_var,
        "mean_rank_top10": mean_rank,
        "ranks_within_top10_by_seed": {
            str(seed): {
                feat: float(rank_mat[j, k])
                for j, feat in enumerate(top_features)
            }
            for k, seed in enumerate(STABILITY_SEEDS)
        },
        "global_ranks_of_top10_by_seed": {
            str(seed): {
                feat: ranks_global[k][j]
                for j, feat in enumerate(top_features)
            }
            for k, seed in enumerate(STABILITY_SEEDS)
        },
        "mean_abs_shap_top10_by_seed": mean_abs_by_seed,
        "interpretation": (
            "W approaching 1.0 indicates stable adverse-action feature ranks "
            "across random seeds (EU AI Act / CFPB explanation stability)."
        ),
    }

    out_json = RESULTS_DIR / "stability_report_v3.json"
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"  Wrote {out_json.relative_to(PROJECT_ROOT)}")
    print(f"  Kendall's W (top {TOP_K} features, {len(STABILITY_SEEDS)} seeds) = {w_stats['kendall_w']:.4f}")
    print(f"  Friedman chi-square = {w_stats['friedman_chi2']:.3f}  (df={w_stats['df']})")

    print("\n" + "=" * 78)
    print("Week 3 explainer v3 finished.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nExplainer pipeline failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1) from exc
