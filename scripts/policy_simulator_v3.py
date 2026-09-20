"""OOT policy simulator v3: XGBoost baseline vs naive ML vs LinearDML CATE.

Bridges trainer_v3.py (XGBoost champion) and causal_inference_v2.py Model A
(Term_Years treatment; Guarantee_Ratio as a common cause). Does not run GCM.

Intervention: extend Term_Years by 5 for OOT loans with baseline default
probability in [0.50, 0.65] (marginal rejections).
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import traceback
import warnings
from pathlib import Path

import joblib
import networkx as nx
import numpy as np
import pandas as pd
from dowhy import CausalModel
from fknni import FastKNNImputer
from lightgbm import LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
ARTIFACTS_DIR = PROJECT_ROOT / "models" / "artifacts"

TARGET = "MIS_Status"
TREATMENT_TERM = "Term_Years"
TREATMENT_GUAR = "Guarantee_Ratio"
INTEREST_RATE = 0.06
APPROVE_CUT = 0.50
MARGINAL_LO = 0.50
MARGINAL_HI = 0.65
TERM_DELTA = 5.0
P_CLIP = 1e-5
BOOTSTRAP_ITERS = 1_000
RANDOM_STATE = 42

# Exact Model A confounder list from causal_inference_v2.py.
CONFOUNDERS = [
    "Log_GrAppv",
    "NoEmp",
    "NewExist_Clean",
    "State_Points",
    "NAICS_Sector_Points",
    "UrbanRural",
]

LGBM_KW = dict(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.05,
    verbosity=-1,
    n_jobs=-1,
    random_state=RANDOM_STATE,
)
LGBM_SEARCH = {
    "n_estimators": [80, 150, 300],
    "max_depth": [3, 6, 8],
    "learning_rate": [0.03, 0.05, 0.1],
    "num_leaves": [15, 31, 63],
}
_TUNED_NUISANCE: dict | None = None


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_xy(split: str) -> tuple[pd.DataFrame, pd.Series]:
    x_path = PROCESSED_DIR / f"X_{split}_v3.csv"
    y_path = PROCESSED_DIR / f"y_{split}_v3.csv"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"Missing {split} v3 split. Run scripts/preprocess_v3.py first."
        )
    X = pd.read_csv(x_path, low_memory=False)
    y = pd.read_csv(y_path, low_memory=False)
    y = y[TARGET] if TARGET in y.columns else y.iloc[:, 0]
    y = y.astype(int)
    if len(X) != len(y):
        raise ValueError(f"{split}: X/y length mismatch.")
    print(f"  Loaded {split}: X={X.shape}  default rate={float(y.mean()):.6f}")
    return X, y


# ---------------------------------------------------------------------------
# Frozen trainer_v3 imputer (pickle namespace remap)
# ---------------------------------------------------------------------------
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


def impute_from_reference(X_ref_imp: pd.DataFrame, X_apply_enc: pd.DataFrame) -> pd.DataFrame:
    n_ref = len(X_ref_imp)
    X_apply_aligned = X_apply_enc.reindex(columns=list(X_ref_imp.columns))
    stacked = np.vstack([_to_numpy(X_ref_imp), _to_numpy(X_apply_aligned)])
    print(
        f"    FastKNNImputer.fit_transform on stacked reference+apply "
        f"({stacked.shape[0]:,} rows) ...",
        flush=True,
    )
    imputed = FastKNNImputer(n_neighbors=5, strategy="mean").fit_transform(stacked)
    if hasattr(imputed, "get"):
        imputed = imputed.get()
    stacked_imp = np.asarray(imputed, dtype=np.float64)
    return pd.DataFrame(
        stacked_imp[n_ref:],
        columns=X_ref_imp.columns,
        index=X_apply_enc.index,
    )


class StackedFastKNNImputer:
    def __init__(self, n_neighbors: int = 5, strategy: str = "mean"):
        self.n_neighbors = n_neighbors
        self.strategy = strategy
        self.reference_: pd.DataFrame | None = None
        self.encodings_: dict = {}
        self.feature_columns_: list[str] = []

    def transform(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        if self.reference_ is None:
            raise RuntimeError("StackedFastKNNImputer has no training reference.")
        X_enc = encode_with_mapping(X_raw, self.encodings_, self.feature_columns_)
        return impute_from_reference(self.reference_, X_enc)


def _register_imputer_for_unpickle() -> None:
    this = sys.modules[__name__]
    for name in (
        "trainer_v3",
        "scripts.trainer_v3",
        "trainer_v2",
        "scripts.trainer_v2",
        "trainer",
        "scripts.trainer",
        "__main__",
    ):
        sys.modules.setdefault(name, this)


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
        raise RuntimeError(f"{path} did not contain a fitted StackedFastKNNImputer.")
    return obj


def apply_frozen_scaler(X_imp: pd.DataFrame, scaler) -> pd.DataFrame:
    X_out = X_imp.copy()
    if hasattr(scaler, "feature_names_in_"):
        cols = [c for c in scaler.feature_names_in_ if c in X_out.columns]
    else:
        cols = []
    if cols:
        X_out[cols] = scaler.transform(X_out[cols])
    return X_out


def transform_for_xgb(X_raw: pd.DataFrame, imputer, scaler) -> pd.DataFrame:
    return apply_frozen_scaler(imputer.transform(X_raw), scaler)


# ---------------------------------------------------------------------------
# Cashflows (unscaled features; GrAppv recovered from log1p if needed)
# ---------------------------------------------------------------------------
def clip_p(p) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), P_CLIP, 1.0 - P_CLIP)


def principal_amount(X: pd.DataFrame) -> np.ndarray:
    if "GrAppv" in X.columns:
        g = pd.to_numeric(X["GrAppv"], errors="coerce").to_numpy(dtype=float)
    elif "Log_GrAppv" in X.columns:
        g = np.expm1(pd.to_numeric(X["Log_GrAppv"], errors="coerce").to_numpy(dtype=float))
    else:
        raise KeyError("Need GrAppv or Log_GrAppv.")
    return np.where(np.isfinite(g) & (g > 0), g, 0.0)


def term_years(X: pd.DataFrame) -> np.ndarray:
    t = pd.to_numeric(X[TREATMENT_TERM], errors="coerce").to_numpy(dtype=float)
    return np.where(np.isfinite(t) & (t > 0), t, 0.0)


def guarantee_ratio(X: pd.DataFrame) -> np.ndarray:
    g = pd.to_numeric(X[TREATMENT_GUAR], errors="coerce").to_numpy(dtype=float)
    g = np.where(np.isfinite(g), g, 0.0)
    return np.clip(g, 0.0, 1.0)


def expected_value(p, principal, term, guar, rate: float = INTEREST_RATE) -> np.ndarray:
    """Per-loan EV if originated. Rejected loans should be zeroed by the caller."""
    p = clip_p(p)
    principal = np.asarray(principal, dtype=float)
    term = np.asarray(term, dtype=float)
    guar = np.asarray(guar, dtype=float)
    interest = principal * rate * term
    loss = principal * (1.0 - guar)
    return (1.0 - p) * interest - p * loss


def portfolio_from_proba(p, principal, term, guar) -> tuple[np.ndarray, np.ndarray]:
    """Approve if p < 0.50. Returns (approved_mask, per-loan EV with rejects = 0)."""
    p = clip_p(p)
    approved = p < APPROVE_CUT
    ev = expected_value(p, principal, term, guar)
    ev = np.where(approved, ev, 0.0)
    return approved, ev


# ---------------------------------------------------------------------------
# LinearDML Model A (causal_inference_v2.py) — fit only, no GCM
# ---------------------------------------------------------------------------
def graph_to_gml(graph: nx.DiGraph) -> str:
    return "\n".join(nx.generate_gml(graph))


def identification_graph(treatment: str, common_causes: list[str]) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_nodes_from(common_causes + [treatment, TARGET])
    for x in common_causes:
        g.add_edge(x, treatment)
        g.add_edge(x, TARGET)
    g.add_edge(treatment, TARGET)
    return g


def lgbm_regressor() -> LGBMRegressor:
    params = dict(LGBM_KW)
    if _TUNED_NUISANCE:
        params.update(_TUNED_NUISANCE)
    return LGBMRegressor(**params)


def tune_nuisance_lgbm(X: pd.DataFrame, y: pd.Series) -> dict:
    global _TUNED_NUISANCE
    if _TUNED_NUISANCE is not None:
        return _TUNED_NUISANCE
    n = min(30_000, len(X))
    if n < len(X):
        Xs, _, ys, _ = train_test_split(
            X, y, train_size=n, stratify=None, random_state=RANDOM_STATE
        )
    else:
        Xs, ys = X, y
    print(f"  Tuning LGBMRegressor nuisance model on {len(Xs):,} rows ...")
    search = RandomizedSearchCV(
        LGBMRegressor(**LGBM_KW),
        LGBM_SEARCH,
        n_iter=10,
        cv=2,
        scoring="neg_mean_squared_error",
        n_jobs=1,
        random_state=RANDOM_STATE,
        refit=True,
        verbose=0,
    )
    search.fit(Xs, ys)
    _TUNED_NUISANCE = dict(search.best_params_)
    print(f"  Nuisance best params: {_TUNED_NUISANCE}")
    return _TUNED_NUISANCE


def complete_case_train(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    confounders = [c for c in CONFOUNDERS if c in X.columns]
    missing = [c for c in CONFOUNDERS if c not in X.columns]
    if missing:
        raise KeyError(f"Required confounders missing from X_train_v3: {missing}")
    cols = confounders + [TREATMENT_TERM, TREATMENT_GUAR, TARGET]
    df = X.copy()
    df[TARGET] = np.asarray(y, dtype=int)
    out = df[cols].apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
    print(f"  Complete-case train for LinearDML: {len(out):,} / {len(X):,}")
    if len(out) < 500:
        raise ValueError("Too few complete-case rows to fit LinearDML.")
    return out


def fit_linear_dml_model_a(df: pd.DataFrame):
    """Model A: Term_Years treatment; Guarantee_Ratio in the backdoor set."""
    confounders = [c for c in CONFOUNDERS if c in df.columns]
    common_causes = confounders + [TREATMENT_GUAR]
    tune_nuisance_lgbm(df[common_causes + [TREATMENT_TERM]], df[TARGET])
    gml = graph_to_gml(identification_graph(TREATMENT_TERM, common_causes))
    model = CausalModel(
        data=df,
        treatment=TREATMENT_TERM,
        outcome=TARGET,
        graph=gml,
    )
    estimand = model.identify_effect(proceed_when_unidentifiable=True)
    print("  Fitting LinearDML (Model A: Term_Years; no GCM) ...")
    estimate = model.estimate_effect(
        estimand,
        method_name="backdoor.econml.dml.LinearDML",
        target_units="ate",
        confidence_intervals=False,
        method_params={
            "init_params": {
                "model_y": lgbm_regressor(),
                "model_t": lgbm_regressor(),
                "discrete_treatment": False,
                "cv": 3,
                "random_state": RANDOM_STATE,
            },
            "fit_params": {},
        },
    )
    print(f"  LinearDML ATE (Term_Years -> default) = {float(estimate.value):.6f}")
    return estimate, common_causes


def _unwrap_econml(estimate):
    est = getattr(estimate, "estimator", estimate)
    for attr in ("econml_estimator_", "_econml_estimator", "estimator"):
        inner = getattr(est, attr, None)
        if inner is not None and hasattr(inner, "effect"):
            return inner
    return est


def cate_term_extension(estimate, W: pd.DataFrame, t0: np.ndarray, t1: np.ndarray) -> np.ndarray:
    """ATE-scaled Delta-P for T1 vs T0 (LinearDML fitted with no effect modifiers).

    Do not pass confounders as X — EconML expects X to match the (empty) fit-time X.
    """
    del W  # retained in the signature for call-site compatibility
    t0 = np.asarray(t0, dtype=float).reshape(-1)
    t1 = np.asarray(t1, dtype=float).reshape(-1)
    est = _unwrap_econml(estimate)
    attempts = []
    if hasattr(est, "effect"):
        attempts.extend(
            [
                lambda: est.effect(X=None, T0=t0, T1=t1),
                lambda: est.effect(X=None, T0=t0.reshape(-1, 1), T1=t1.reshape(-1, 1)),
                lambda: np.ravel(est.effect(X=None)) * (t1 - t0),
            ]
        )
    if hasattr(est, "const_marginal_effect"):
        attempts.append(lambda: np.ravel(est.const_marginal_effect(X=None)) * (t1 - t0))
    last = None
    for fn in attempts:
        try:
            cate = np.ravel(np.asarray(fn(), dtype=float))
            if cate.size == 1:
                cate = np.full(len(t0), float(cate[0]))
            if len(cate) == len(t0) and np.isfinite(cate).any():
                return cate
        except Exception as exc:
            last = exc
            continue
    ate = float(estimate.value)
    print(f"  CATE.effect() unavailable ({last}); falling back to ATE * ΔT.")
    return ate * (t1 - t0)


# ---------------------------------------------------------------------------
# Bootstrap of causal vs baseline portfolio value
# ---------------------------------------------------------------------------
def bootstrap_uplift(ev_causal: np.ndarray, ev_base: np.ndarray, n: int = BOOTSTRAP_ITERS) -> dict:
    rng = np.random.default_rng(RANDOM_STATE)
    diff = np.asarray(ev_causal, dtype=float) - np.asarray(ev_base, dtype=float)
    m = len(diff)
    stats = np.empty(n, dtype=float)
    for i in range(n):
        idx = rng.integers(0, m, size=m)
        stats[i] = float(diff[idx].sum())
    lo, hi = np.percentile(stats, [2.5, 97.5])
    # Two-sided bootstrap p-value against uplift = 0.
    p_le = float(np.mean(stats <= 0.0))
    p_ge = float(np.mean(stats >= 0.0))
    p_val = min(1.0, 2.0 * min(p_le, p_ge))
    return {
        "n_iterations": int(n),
        "mean": float(stats.mean()),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "p_value_two_sided": p_val,
        "significant_at_0.05": bool((lo > 0.0) or (hi < 0.0)),
    }


def main() -> int:
    os.chdir(PROJECT_ROOT)
    ensure_dirs()
    print("=" * 78)
    print("Policy simulator v3 — XGBoost vs LinearDML term extension")
    print("=" * 78)

    imputer_path = ARTIFACTS_DIR / "imputer_v3.joblib"
    scaler_path = ARTIFACTS_DIR / "scaler_v3.joblib"
    model_path = ARTIFACTS_DIR / "xgboost_best_v3.json"
    for p in (imputer_path, scaler_path, model_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing artifact: {p}. Run scripts/trainer_v3.py first.")

    print("\n[1] Load v3 splits and frozen XGBoost pipeline")
    X_train, y_train = load_xy("train")
    X_oot, _y_oot = load_xy("oot")

    print(f"  Loading {imputer_path.relative_to(PROJECT_ROOT)}")
    imputer = load_imputer(imputer_path)
    scaler = joblib.load(scaler_path)
    champion = XGBClassifier()
    champion.load_model(str(model_path))

    print("\n[2] Fit LinearDML Model A on complete-case X_train_v3")
    dml_df = complete_case_train(X_train, y_train)
    estimate, common_causes = fit_linear_dml_model_a(dml_df)

    print("\n[3] Baseline XGBoost probabilities on OOT")
    X_oot_p = transform_for_xgb(X_oot, imputer, scaler)
    p_base = clip_p(champion.predict_proba(X_oot_p)[:, 1])
    principal = principal_amount(X_oot)
    term0 = term_years(X_oot)
    guar = guarantee_ratio(X_oot)
    appr_base, ev_base = portfolio_from_proba(p_base, principal, term0, guar)
    baseline_value = float(ev_base.sum())
    print(
        f"  Approved={int(appr_base.sum()):,} / {len(X_oot):,}  "
        f"Baseline_Portfolio_Value={baseline_value:,.0f}"
    )

    targeted = (p_base >= MARGINAL_LO) & (p_base <= MARGINAL_HI)
    n_targeted = int(targeted.sum())
    print(
        f"\n[4] Marginal rejections: {MARGINAL_LO:.2f} <= P <= {MARGINAL_HI:.2f}  "
        f"n={n_targeted:,}"
    )
    if n_targeted == 0:
        print("  No targeted loans; causal/naive portfolios equal the baseline.")

    print("\n[5] Naive ML policy — XGBoost on Term_Years + 5 (targeted rows only)")
    X_oot_naive = X_oot.copy()
    X_oot_naive.loc[targeted, TREATMENT_TERM] = (
        pd.to_numeric(X_oot_naive.loc[targeted, TREATMENT_TERM], errors="coerce") + TERM_DELTA
    )
    term_naive = term0.copy()
    term_naive[targeted] = term0[targeted] + TERM_DELTA
    X_oot_naive_p = transform_for_xgb(X_oot_naive, imputer, scaler)
    p_naive = clip_p(champion.predict_proba(X_oot_naive_p)[:, 1])
    appr_naive, ev_naive = portfolio_from_proba(p_naive, principal, term_naive, guar)
    naive_value = float(ev_naive.sum())
    print(
        f"  Approved={int(appr_naive.sum()):,}  "
        f"Naive_Portfolio_Value={naive_value:,.0f}  "
        f"delta={naive_value - baseline_value:,.0f}"
    )

    print("\n[6] Causal policy — LinearDML CATE as Delta-P on targeted rows")
    p_causal = p_base.copy()
    cate = np.zeros(len(X_oot), dtype=float)
    if n_targeted:
        W_tgt = X_oot.loc[targeted, common_causes].apply(pd.to_numeric, errors="coerce")
        t0 = term0[targeted]
        t1 = t0 + TERM_DELTA
        cate_tgt = cate_term_extension(estimate, W_tgt, t0, t1)
        cate[targeted] = cate_tgt
        p_causal[targeted] = clip_p(p_base[targeted] + cate_tgt)
        print(
            f"  Targeted CATE (Delta-P): mean={float(np.mean(cate_tgt)):.6f}  "
            f"median={float(np.median(cate_tgt)):.6f}"
        )
    term_causal = term_naive  # same +5 years on originated intervened loans
    appr_causal, ev_causal = portfolio_from_proba(p_causal, principal, term_causal, guar)
    causal_value = float(ev_causal.sum())
    print(
        f"  Approved={int(appr_causal.sum()):,}  "
        f"Causal_Portfolio_Value={causal_value:,.0f}  "
        f"delta={causal_value - baseline_value:,.0f}"
    )

    print("\n[6b] Confounded danger zone — naive approve, causal deny")
    dangerous_approvals = (p_naive < APPROVE_CUT) & (p_causal >= APPROVE_CUT)
    n_dangerous = int(dangerous_approvals.sum())
    if n_dangerous:
        toxic_principal = float(
            (principal[dangerous_approvals] * (1.0 - guar[dangerous_approvals])).sum()
        )
        ev_dangerous = float(
            expected_value(
                p_causal[dangerous_approvals],
                principal[dangerous_approvals],
                term_causal[dangerous_approvals],
                guar[dangerous_approvals],
            ).sum()
        )
    else:
        toxic_principal = 0.0
        ev_dangerous = 0.0
    print(f"  n_dangerous_naive_approvals={n_dangerous:,}")
    print(f"  toxic_principal_exposure_avoided={toxic_principal:,.0f}")
    print(f"  true_ev_of_dangerous_approvals={ev_dangerous:,.0f}")

    print(f"\n[7] Bootstrap {BOOTSTRAP_ITERS} OOT resamples of causal − baseline EV")
    boot = bootstrap_uplift(ev_causal, ev_base)
    print(
        f"  Uplift mean={boot['mean']:,.0f}  "
        f"95% CI=[{boot['ci95_low']:,.0f}, {boot['ci95_high']:,.0f}]  "
        f"p={boot['p_value_two_sided']:.4f}  "
        f"sig@0.05={boot['significant_at_0.05']}"
    )

    report = {
        "interest_rate": INTEREST_RATE,
        "approval_threshold": APPROVE_CUT,
        "marginal_band": [MARGINAL_LO, MARGINAL_HI],
        "term_extension_years": TERM_DELTA,
        "n_oot": int(len(X_oot)),
        "n_targeted_marginal": n_targeted,
        "n_approved_baseline": int(appr_base.sum()),
        "n_approved_naive": int(appr_naive.sum()),
        "n_approved_causal": int(appr_causal.sum()),
        "linear_dml_ate": float(estimate.value),
        "mean_cate_targeted": float(np.mean(cate[targeted])) if n_targeted else None,
        "Baseline_Portfolio_Value": baseline_value,
        "Naive_Portfolio_Value": naive_value,
        "Naive_delta_from_baseline": naive_value - baseline_value,
        "Causal_Portfolio_Value": causal_value,
        "Causal_delta_from_baseline": causal_value - baseline_value,
        "n_dangerous_naive_approvals": n_dangerous,
        "toxic_principal_exposure_avoided": toxic_principal,
        "true_ev_of_dangerous_approvals": ev_dangerous,
        "causal_uplift_bootstrap": boot,
        "note": (
            "Naive ML re-scores Term_Years+5 through XGBoost (confounded). "
            "Causal adds LinearDML CATE (Delta-P) to baseline XGBoost probabilities. "
            "EV uses interest on performing originations minus unsecured loss on defaults."
        ),
    }
    out_path = RESULTS_DIR / "policy_simulation_v3.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nWrote {out_path.relative_to(PROJECT_ROOT)}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nPolicy simulator failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1) from exc
