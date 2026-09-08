"""Causal validation of contract-term effects on SBA default (JRFM-oriented).

Phase 1 — Permutation GCM falsification of the proposed DAG (DoWhy GCM).
Phase 2 — Backdoor ATE of Term_Years and Guarantee_Ratio via EconML LinearDML.
Phase 3 — Classical DoWhy refutation suite; JSON report.

Target: MIS_Status (0 = Paid in Full, 1 = Default).
Treatments: Term_Years, Guarantee_Ratio.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from dowhy import CausalModel
from dowhy.gcm.falsify import FalsifyConst, apply_suggestions, falsify_graph
from dowhy.gcm.independence_test.generalised_cov_measure import generalised_cov_based
from dowhy.gcm.ml import SklearnRegressionModel
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"

TARGET = "MIS_Status"
TREATMENT_TERM = "Term_Years"
TREATMENT_GUAR = "Guarantee_Ratio"

# Proposed observed confounders. Unemployment / Prime_Rate are part of the
# scientific DAG; they are skipped with a warning if absent from X_train.
CONFOUNDERS = [
    "Log_GrAppv",
    "NoEmp",
    "NewExist_Clean",
    "State_Points",
    "NAICS_Sector_Points",
    "UrbanRural",
    "Unemployment",
    "Prime_Rate",
]

GCM_N = 10_000
N_PERMUTATIONS = 100
ALPHA = 0.05
RANDOM_STATE = 42

LGBM_KW = dict(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.05,
    verbosity=-1,
    n_jobs=-1,
    random_state=RANDOM_STATE,
)


def ensure_dirs() -> None:
    for d in (FIGURES_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
        print(f"  [ok] {d.relative_to(PROJECT_ROOT)}")


def load_train() -> pd.DataFrame:
    x_path = PROCESSED_DIR / "X_train.csv"
    y_path = PROCESSED_DIR / "y_train.csv"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError("Run scripts/preprocess.py first (X_train / y_train).")
    X = pd.read_csv(x_path, low_memory=False)
    y = pd.read_csv(y_path, low_memory=False)
    y = y[TARGET] if TARGET in y.columns else y.iloc[:, 0]
    df = X.copy()
    df[TARGET] = y.astype(int).to_numpy()
    print(f"  Loaded train: {df.shape}  default rate={float(df[TARGET].mean()):.6f}")
    return df


def resolve_confounders(columns: list[str]) -> list[str]:
    present = [c for c in CONFOUNDERS if c in columns]
    missing = [c for c in CONFOUNDERS if c not in columns]
    if missing:
        print(
            "  WARNING: confounders not in X_train (skipped until merged): "
            f"{missing}"
        )
    if not present:
        raise KeyError("None of the listed confounders are present in X_train.")
    print(f"  Observed confounders used: {present}")
    return present


def analysis_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Numeric subset; drop incomplete rows (no KNN — that belongs in trainer.py)."""
    out = df[cols].apply(pd.to_numeric, errors="coerce")
    n0 = len(out)
    out = out.dropna().reset_index(drop=True)
    print(f"  Complete-case rows for causal DAG: {len(out):,} / {n0:,}")
    if len(out) < 500:
        raise ValueError("Too few complete-case rows to run GCM / DML.")
    return out


def stratified_sample(df: pd.DataFrame, n: int, seed: int = RANDOM_STATE) -> pd.DataFrame:
    n = min(n, len(df))
    if n == len(df):
        return df.copy()
    sampled, _ = train_test_split(
        df, train_size=n, stratify=df[TARGET], random_state=seed
    )
    return sampled.reset_index(drop=True)


def build_scientific_dag(confounders: list[str]) -> nx.DiGraph:
    """Domain DAG used for GCM falsification.

    Confounders -> both contract terms and default.
    Term_Years -> Guarantee_Ratio (maturity-linked guarantee caps).
    Both contract terms -> MIS_Status.
    """
    g = nx.DiGraph()
    nodes = confounders + [TREATMENT_TERM, TREATMENT_GUAR, TARGET]
    g.add_nodes_from(nodes)
    for x in confounders:
        g.add_edge(x, TREATMENT_TERM)
        g.add_edge(x, TREATMENT_GUAR)
        g.add_edge(x, TARGET)
    g.add_edge(TREATMENT_TERM, TREATMENT_GUAR)
    g.add_edge(TREATMENT_TERM, TARGET)
    g.add_edge(TREATMENT_GUAR, TARGET)
    if not nx.is_directed_acyclic_graph(g):
        raise RuntimeError("Proposed graph is not a DAG.")
    print(f"  DAG nodes={g.number_of_nodes()}  edges={g.number_of_edges()}")
    return g


def graph_to_gml(graph: nx.DiGraph) -> str:
    return "\n".join(nx.generate_gml(graph))


def identification_graph(treatment: str, common_causes: list[str]) -> nx.DiGraph:
    """Backdoor graph: every listed common cause points at treatment and outcome."""
    g = nx.DiGraph()
    g.add_nodes_from(common_causes + [treatment, TARGET])
    for x in common_causes:
        g.add_edge(x, treatment)
        g.add_edge(x, TARGET)
    g.add_edge(treatment, TARGET)
    return g


def create_lgbm_regressor(**kwargs) -> SklearnRegressionModel:
    params = dict(LGBM_KW)
    params.update(kwargs)
    params.pop("random_state", None)
    return SklearnRegressionModel(LGBMRegressor(**params, random_state=RANDOM_STATE))


def gcm_independence(X, Y, Z=None):
    """Generalised covariance measure with LightGBM residualizers (continuous GCM)."""
    return generalised_cov_based(
        X,
        Y,
        Z=Z,
        prediction_model_X=create_lgbm_regressor,
        prediction_model_Y=create_lgbm_regressor,
    )


def extract_falsify_stats(result) -> dict:
    """Pull LMC counts / permutation p-value from DoWhy EvaluationResult."""
    payload = {
        "falsifiable": bool(getattr(result, "falsifiable", False)),
        "falsified": bool(getattr(result, "falsified", False)),
        "summary_text": str(result),
        "lmc_violations": None,
        "permutation_pvalue": None,
        "n_lmc_tests": None,
    }
    summary = getattr(result, "summary", None)
    if not isinstance(summary, dict):
        return payload
    lmc = summary.get(FalsifyConst.VALIDATE_LMC, {})
    if isinstance(lmc, dict):
        payload["lmc_violations"] = _maybe_int(lmc.get(FalsifyConst.N_VIOLATIONS))
        payload["n_lmc_tests"] = _maybe_int(lmc.get(FalsifyConst.N_TESTS))
        payload["permutation_pvalue"] = _maybe_float(lmc.get(FalsifyConst.P_VALUE))
    tpa = summary.get(FalsifyConst.VALIDATE_TPA, {})
    if isinstance(tpa, dict) and payload["permutation_pvalue"] is None:
        payload["permutation_pvalue"] = _maybe_float(tpa.get(FalsifyConst.P_VALUE))
    return payload


def _maybe_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _maybe_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def serialize_suggestions(result, graph: nx.DiGraph) -> dict:
    """Causal-minimality suggestions (edges the test would prune)."""
    suggestions = getattr(result, "suggestions", None)
    if not suggestions:
        return {"pruned_edges": [], "note": "No causal-minimality suggestions returned."}
    try:
        pruned_graph = apply_suggestions(graph, result)
        pruned = sorted(set(graph.edges()) - set(pruned_graph.edges()))
    except Exception as exc:
        return {"pruned_edges": [], "error": str(exc)}
    return {
        "pruned_edges": [f"{a} -> {b}" for a, b in pruned],
        "n_pruned": len(pruned),
    }


def lgbm_regressor() -> LGBMRegressor:
    return LGBMRegressor(**LGBM_KW)


def estimate_dml(df: pd.DataFrame, treatment: str, common_causes: list[str]) -> dict:
    print(f"\n  --- DML  treatment={treatment}  outcome={TARGET} ---")
    g = identification_graph(treatment, common_causes)
    gml = graph_to_gml(g)
    model = CausalModel(
        data=df,
        treatment=treatment,
        outcome=TARGET,
        graph=gml,
    )
    print("  Identifying via backdoor criterion ...")
    estimand = model.identify_effect(proceed_when_unidentifiable=True)
    print(estimand)

    print("  Estimating ATE with backdoor.econml.dml.LinearDML ...")
    estimate = model.estimate_effect(
        estimand,
        method_name="backdoor.econml.dml.LinearDML",
        target_units="ate",
        confidence_intervals=True,
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
    ate = float(estimate.value)
    ci = None
    try:
        raw_ci = estimate.get_confidence_intervals()
        flat = np.ravel(raw_ci)
        ci = [float(flat[0]), float(flat[1])]
    except Exception:
        try:
            ci = [float(x) for x in np.ravel(estimate.estimator.effect_interval())[:2]]
        except Exception:
            ci = None
    print(f"  ATE ({treatment} -> {TARGET}) = {ate:.6f}")
    if ci:
        print(f"  95% CI = [{ci[0]:.6f}, {ci[1]:.6f}]")
    return {
        "treatment": treatment,
        "outcome": TARGET,
        "common_causes": common_causes,
        "ate": ate,
        "ci95": ci,
        "estimand": str(estimand),
        "estimate": str(estimate),
        "_model": model,
        "_estimand": estimand,
        "_estimate": estimate,
    }


def _refuter_pvalue(ref) -> float | None:
    p = _maybe_float(getattr(ref, "p_value", None))
    if p is not None:
        return p
    inner = getattr(ref, "refutation_result", None)
    if isinstance(inner, dict):
        return _maybe_float(inner.get("p_value"))
    return None


def refute_bundle(model, estimand, estimate) -> dict:
    """Four classical DoWhy refuters. Missing p-values are stored as null."""
    tests = [
        ("random_common_cause", {"method_name": "random_common_cause"}),
        (
            "placebo_treatment_refuter",
            {"method_name": "placebo_treatment_refuter", "placebo_type": "permute"},
        ),
        (
            "data_subset_refuter",
            {"method_name": "data_subset_refuter", "subset_fraction": 0.5},
        ),
        (
            "add_unobserved_common_cause",
            {
                "method_name": "add_unobserved_common_cause",
                "confounders_effect_on_treatment": "linear",
                "confounders_effect_on_outcome": "linear",
                "effect_strength_on_treatment": 0.05,
                "effect_strength_on_outcome": 0.05,
            },
        ),
    ]
    out: dict[str, dict] = {}
    for name, kwargs in tests:
        print(f"    Refuter: {name} ...", flush=True)
        try:
            ref = model.refute_estimate(estimand, estimate, **kwargs)
            rec = {
                "ok": True,
                "new_effect": _maybe_float(getattr(ref, "new_effect", None)),
                "p_value": _refuter_pvalue(ref),
                "text": str(ref),
            }
            print(f"      new_effect={rec['new_effect']}  p={rec['p_value']}")
            out[name] = rec
        except Exception as exc:
            print(f"      FAILED: {exc}")
            out[name] = {"ok": False, "error": str(exc)}
    return out


def json_safe(obj):
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)


def main() -> int:
    os.chdir(PROJECT_ROOT)
    print("=" * 78)
    print("Causal validation — GCM falsification, LinearDML, refutation suite")
    print("=" * 78)
    print("Creating output directories:")
    ensure_dirs()

    print("\nLoading chronological training split ...")
    df_raw = load_train()
    confounders = resolve_confounders(list(df_raw.columns))
    dag_cols = confounders + [TREATMENT_TERM, TREATMENT_GUAR, TARGET]
    missing_core = [c for c in (TREATMENT_TERM, TREATMENT_GUAR, TARGET) if c not in df_raw.columns]
    if missing_core:
        raise KeyError(f"Required causal columns missing: {missing_core}")

    df = analysis_frame(df_raw, dag_cols)

    # ------------------------------------------------------------------
    # Phase 1 — GCM graph falsification (10k stratified sample)
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("PHASE 1  GCM permutation falsification of the proposed DAG")
    print("=" * 78)
    gcm_df = stratified_sample(df, GCM_N)
    print(
        f"  GCM sample n={len(gcm_df):,}  default rate={float(gcm_df[TARGET].mean()):.4f}"
    )
    graph = build_scientific_dag(confounders)

    hist_path = FIGURES_DIR / "gcm_falsify_histogram.png"
    print(
        f"  Running falsify_graph (n_permutations={N_PERMUTATIONS}, GCM independence) ..."
    )
    result = falsify_graph(
        graph,
        gcm_df,
        n_permutations=N_PERMUTATIONS,
        suggestions=True,
        independence_test=gcm_independence,
        conditional_independence_test=gcm_independence,
        significance_level=ALPHA,
        plot_histogram=True,
        plot_kwargs={"savepath": str(hist_path), "display": False},
        show_progress_bar=True,
    )
    plt.close("all")
    if hist_path.exists():
        print(f"  Wrote {hist_path.relative_to(PROJECT_ROOT)}")

    stats = extract_falsify_stats(result)
    suggestions = serialize_suggestions(result, graph)
    print("\n  --- GCM falsification summary ---")
    print(stats["summary_text"])
    print(f"  Local Markov Condition violations: {stats['lmc_violations']}")
    print(f"  LMC tests run:                      {stats['n_lmc_tests']}")
    print(f"  Permutation p-value:                {stats['permutation_pvalue']}")
    print(f"  Graph falsifiable (alpha={ALPHA}):   {stats['falsifiable']}")
    print(f"  Graph falsified   (alpha={ALPHA}):   {stats['falsified']}")
    print(f"  Suggested pruned edges: {suggestions.get('pruned_edges') or 'none'}")

    # ------------------------------------------------------------------
    # Phase 2 — DML on the full complete-case training table
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("PHASE 2  Double Machine Learning (LinearDML) on full training complete cases")
    print("=" * 78)
    print(
        "  Identification note: DML Model A treats Guarantee_Ratio as a common cause "
        "(direct-effect / backdoor), not as a mediator. Model B treats Term_Years as a "
        "common cause. This matches the requested CausalModel setup; it is a different "
        "adjustment set than the GCM DAG's Term_Years -> Guarantee_Ratio path."
    )

    model_a = estimate_dml(
        df,
        treatment=TREATMENT_TERM,
        common_causes=confounders + [TREATMENT_GUAR],
    )
    model_b = estimate_dml(
        df,
        treatment=TREATMENT_GUAR,
        common_causes=confounders + [TREATMENT_TERM],
    )

    # ------------------------------------------------------------------
    # Phase 3 — refutation suite
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("PHASE 3  Refutation suite (both treatments)")
    print("=" * 78)
    print("  random_common_cause     -> estimate should stay similar")
    print("  placebo_treatment        -> estimate should collapse toward 0")
    print("  data_subset              -> estimate should stay similar")
    print("  add_unobserved_common_cause -> sensitivity band")

    print("\n  Model A (Term_Years):")
    refute_a = refute_bundle(model_a["_model"], model_a["_estimand"], model_a["_estimate"])
    print("\n  Model B (Guarantee_Ratio):")
    refute_b = refute_bundle(model_b["_model"], model_b["_estimand"], model_b["_estimate"])

    report = {
        "target": TARGET,
        "treatments": [TREATMENT_TERM, TREATMENT_GUAR],
        "confounders_requested": CONFOUNDERS,
        "confounders_used": confounders,
        "n_complete_case_train": int(len(df)),
        "n_gcm_sample": int(len(gcm_df)),
        "alpha": ALPHA,
        "phase1_gcm": {
            **{k: v for k, v in stats.items() if k != "summary_text"},
            "suggestions": suggestions,
            "dag_edges": [f"{a} -> {b}" for a, b in graph.edges()],
        },
        "phase2_dml": {
            "term_years": json_safe(model_a),
            "guarantee_ratio": json_safe(model_b),
        },
        "phase3_refutation": {
            "term_years": refute_a,
            "guarantee_ratio": refute_b,
        },
    }
    out_path = RESULTS_DIR / "causal_stability_report.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(json_safe(report), fh, indent=2)
    print(f"\nWrote {out_path.relative_to(PROJECT_ROOT)}")
    print("=" * 78)
    print("Causal pipeline finished.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nCausal pipeline failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1) from exc
