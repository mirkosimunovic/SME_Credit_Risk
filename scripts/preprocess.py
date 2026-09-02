"""Leak-proof chronological preprocessing for SBA SME credit-risk modeling.

Ingests data/raw/SBAnational.csv, defines the default target, splits oldest 80%
vs newest 20% (out-of-time), and computes State / NAICS-sector risk points
using training-split statistics only.

No global imputer or StandardScaler is applied: NaNs are preserved for
fold-isolated imputation in the training script.

Target: MIS_Status (0 = Paid in Full, 1 = Default).
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "SBAnational.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TARGET_COL = "MIS_Status"
DATE_COL = "ApprovalDate"
TRAIN_FRACTION = 0.80
N_QUANTILES = 5
UNSEEN_POINTS = 3  # training-average quintile for unseen / missing keys
DEFAULT_YEAR_CAP = 2024  # two-digit years parsed into the future are rolled back 100y

CURRENCY_COLS = ("GrAppv", "SBA_Appv")

LEAKAGE_COLS = [
    "ChgOffDate",
    "ChgOffPrinGr",
    "DisbursementDate",
    "DisbursementGross",
    "BalanceGross",
    "SBA_Appv",
]
IDENTIFIER_COLS = [
    "LoanNr_ChkDgt",
    "Name",
    "Zip",
    "City",
    "Bank",
    "BankState",
    "ApprovalFY",
    "FranchiseCode",
    "CreateJob",
    "RetainedJob",
    "State",
    "NAICS",
]
# Engineered replacements / intermediates (not model features).
PROCESSED_ORIGINALS = [
    DATE_COL,
    "LowDoc",
    "NewExist",
    "NAICS_Sector",
]


def log(msg: str) -> None:
    print(msg, flush=True)


def parse_currency_series(series: pd.Series) -> pd.Series:
    """Strip $, commas, and whitespace; cast to float. Empty -> NaN."""
    cleaned = (
        series.astype(str)
        .str.replace(r"[\$,\s]", "", regex=True)
        .str.replace(r"[^0-9.\-]", "", regex=True)
        .replace({"": np.nan, "nan": np.nan, "None": np.nan, "NaN": np.nan})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def parse_approval_date(series: pd.Series) -> pd.Series:
    """Parse SBA approval dates such as '10-Jul-97' / '2-Jun-80'."""
    raw = series.astype(str).str.strip().replace({"": np.nan, "nan": np.nan, "NaT": np.nan})
    parsed = pd.to_datetime(raw, format="%d-%b-%y", errors="coerce")
    still_missing = parsed.isna() & raw.notna()
    if still_missing.any():
        parsed.loc[still_missing] = pd.to_datetime(raw.loc[still_missing], errors="coerce")

    future = parsed.notna() & (parsed.dt.year > DEFAULT_YEAR_CAP)
    if future.any():
        rolled = parsed.loc[future].apply(lambda ts: ts.replace(year=ts.year - 100))
        parsed.loc[future] = rolled
        log(f"  Rolled {int(future.sum()):,} two-digit years back by 100 years.")
    return parsed


def recode_target(series: pd.Series) -> pd.Series:
    """CHGOFF -> 1 (Default), PIF / 'P I F' -> 0. Other tokens remain NaN."""
    normalized = (
        series.astype(str)
        .str.upper()
        .str.replace(r"\s+", "", regex=True)
        .str.strip()
        .replace({"NAN": np.nan, "NONE": np.nan, "": np.nan})
    )
    mapped = normalized.map({"CHGOFF": 1, "PIF": 0})
    return mapped


def extract_naics_sector(series: pd.Series) -> pd.Series:
    """First two digits of NAICS; codes <= 0 or unparsable -> 0 (Undefined)."""
    numeric = pd.to_numeric(series, errors="coerce")
    sector = pd.Series(np.nan, index=series.index, dtype="float")
    valid = numeric.notna() & (numeric > 0)
    as_int = numeric.loc[valid].astype(np.int64).astype(str)
    sector.loc[valid] = as_int.str[:2].astype(int)
    return sector.fillna(0).astype(int)


def rates_to_quintile_points(rates: pd.Series) -> dict:
    """Bin entity-level default rates into 5 points.

    Lowest-default quintile -> 5; highest-default quintile -> 1.
    If fewer than 5 distinct rates, bins are stretched onto the 1–5 scale.
    """
    rates = rates.dropna()
    if rates.empty:
        return {}
    n_unique = int(rates.nunique())
    if n_unique == 1:
        return {key: UNSEEN_POINTS for key in rates.index}

    q = min(N_QUANTILES, n_unique)
    try:
        codes = pd.qcut(rates, q=q, labels=False, duplicates="drop")
    except ValueError:
        return {key: UNSEEN_POINTS for key in rates.index}

    n_bins = int(codes.max()) - int(codes.min()) + 1
    if n_bins <= 1:
        return {key: UNSEEN_POINTS for key in rates.index}

    # code 0 = lowest default rate -> 5 points
    span = n_bins - 1
    points = 5 - np.round(codes.to_numpy() * (4 / span)).astype(int)
    points = np.clip(points, 1, 5)
    return dict(zip(rates.index.tolist(), points.tolist()))


def map_points(keys: pd.Series, mapping: dict, default: int = UNSEEN_POINTS) -> pd.Series:
    mapped = keys.map(mapping)
    return pd.to_numeric(mapped, errors="coerce").fillna(default).astype(int)


def build_entity_points(train: pd.DataFrame, key_col: str, label: str) -> dict:
    rates = train.groupby(key_col, dropna=False)[TARGET_COL].mean()
    mapping = rates_to_quintile_points(rates)
    log(f"  {label}: {len(rates):,} keys, {len(mapping):,} scored.")
    if mapping:
        preview = (
            rates.rename("default_rate")
            .to_frame()
            .assign(points=lambda d: d.index.map(mapping))
            .sort_values("default_rate")
        )
        log(f"  {label} lowest-default keys:\n{preview.head(5).to_string()}")
        log(f"  {label} highest-default keys:\n{preview.tail(5).to_string()}")
    return mapping


def recode_lowdoc(series: pd.Series) -> pd.Series:
    """Y -> 1, N -> 0; invalid tokens and nulls stay NaN."""
    token = series.astype(str).str.upper().str.strip()
    out = pd.Series(np.nan, index=series.index, dtype="float")
    out.loc[token.eq("Y")] = 1.0
    out.loc[token.eq("N")] = 0.0
    # original missing values were converted to the string 'NAN'
    return out


def recode_new_exist(new_exist: pd.Series, retained_job: pd.Series) -> pd.Series:
    """1 = Existing, 2 = New; 0/null with RetainedJob >= 1 -> Existing; else 0 -> NaN."""
    ne = pd.to_numeric(new_exist, errors="coerce")
    retained = pd.to_numeric(retained_job, errors="coerce")
    out = pd.Series(np.nan, index=new_exist.index, dtype="float")

    out.loc[ne.eq(1)] = 1.0
    out.loc[ne.eq(2)] = 2.0

    salvage = (ne.isna() | ne.eq(0)) & retained.ge(1)
    out.loc[salvage] = 1.0
    return out


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    gr = pd.to_numeric(out["GrAppv"], errors="coerce")
    sba = pd.to_numeric(out["SBA_Appv"], errors="coerce")
    term = pd.to_numeric(out["Term"], errors="coerce")
    create_job = pd.to_numeric(out["CreateJob"], errors="coerce")
    retained = pd.to_numeric(out["RetainedJob"], errors="coerce")
    franchise = pd.to_numeric(out["FranchiseCode"], errors="coerce")

    out["Log_GrAppv"] = np.log1p(gr.clip(lower=0))
    out["Guarantee_Ratio"] = np.where(gr > 0, sba / gr, np.nan)
    out["Term_Years"] = np.floor_divide(term, 12)
    out["IsCreateJob"] = (create_job > 0).astype("float")
    out["IsRetained"] = (retained > 0).astype("float")
    out.loc[create_job.isna(), "IsCreateJob"] = np.nan
    out.loc[retained.isna(), "IsRetained"] = np.nan
    out["IsFranchise"] = (franchise > 1).astype("float")
    out.loc[franchise.isna(), "IsFranchise"] = np.nan
    out["LowDoc_Binary"] = recode_lowdoc(out["LowDoc"])
    out["NewExist_Clean"] = recode_new_exist(out["NewExist"], out["RetainedJob"])
    return out


def drop_non_features(frame: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [
        c
        for c in LEAKAGE_COLS + IDENTIFIER_COLS + PROCESSED_ORIGINALS
        if c in frame.columns and c != TARGET_COL
    ]
    return frame.drop(columns=drop_cols)


def chronological_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    if n < 2:
        raise ValueError("Not enough rows to form a chronological train/OOT split.")
    cut = int(n * TRAIN_FRACTION)
    if cut <= 0 or cut >= n:
        raise ValueError(f"Invalid train cut {cut} for n={n}.")
    train = df.iloc[:cut].copy()
    oot = df.iloc[cut:].copy()
    log(
        f"  Chronological cut at index {cut:,} / {n:,} "
        f"({TRAIN_FRACTION:.0%} train, {1 - TRAIN_FRACTION:.0%} OOT)."
    )
    log(
        f"  Train dates: {train[DATE_COL].min().date()} -> {train[DATE_COL].max().date()} "
        f"(n={len(train):,})"
    )
    log(
        f"  OOT dates:   {oot[DATE_COL].min().date()} -> {oot[DATE_COL].max().date()} "
        f"(n={len(oot):,})"
    )
    if train[DATE_COL].max() > oot[DATE_COL].min():
        log(
            "  Note: identical ApprovalDate values straddle the cut; "
            "row order after a stable sort is used as the tie-break."
        )
    return train, oot


def save_split(X: pd.DataFrame, y: pd.Series, prefix: str) -> None:
    x_path = PROCESSED_DIR / f"X_{prefix}.csv"
    y_path = PROCESSED_DIR / f"y_{prefix}.csv"
    X.to_csv(x_path, index=False)
    y.to_frame(name=TARGET_COL).to_csv(y_path, index=False)
    nan_cells = int(X.isna().sum().sum())
    nan_cols = X.columns[X.isna().any()].tolist()
    log(f"  Wrote {x_path.relative_to(PROJECT_ROOT)}  shape={X.shape}  NaN cells={nan_cells:,}")
    log(f"  Wrote {y_path.relative_to(PROJECT_ROOT)}  default rate={float(y.mean()):.4f}")
    if nan_cols:
        log(f"    Columns with preserved NaNs ({prefix}): {nan_cols}")


def run() -> None:
    os.chdir(PROJECT_ROOT)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw dataset not found: {RAW_PATH}")

    log("=" * 78)
    log("SBA SME preprocessing — chronological, leak-proof")
    log("=" * 78)

    log(f"\n[1] Ingestion: {RAW_PATH.relative_to(PROJECT_ROOT)}")
    df = pd.read_csv(RAW_PATH, low_memory=False)
    log(f"  Raw shape: {df.shape}")

    required = [
        TARGET_COL,
        DATE_COL,
        "State",
        "NAICS",
        "GrAppv",
        "SBA_Appv",
        "Term",
        "CreateJob",
        "RetainedJob",
        "FranchiseCode",
        "LowDoc",
        "NewExist",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Raw file missing required columns: {missing}")

    n_raw = len(df)
    df[TARGET_COL] = recode_target(df[TARGET_COL])
    before = len(df)
    df = df.dropna(subset=[TARGET_COL]).copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    log(f"  Dropped {before - len(df):,} rows with missing/unmapped {TARGET_COL}.")
    log(
        f"  Class counts: Paid in Full (0)={(df[TARGET_COL] == 0).sum():,} | "
        f"Default (1)={(df[TARGET_COL] == 1).sum():,}"
    )

    log("\n[1b] Currency cleanup (GrAppv, SBA_Appv)")
    for col in CURRENCY_COLS:
        df[col] = parse_currency_series(df[col])
        log(f"  {col}: non-null={df[col].notna().sum():,}  min={df[col].min()}  max={df[col].max()}")

    log("\n[2] Chronological sort and 80/20 OOT split")
    df[DATE_COL] = parse_approval_date(df[DATE_COL])
    n_bad_dates = int(df[DATE_COL].isna().sum())
    if n_bad_dates:
        log(f"  Dropping {n_bad_dates:,} rows with unparseable {DATE_COL}.")
        df = df.dropna(subset=[DATE_COL]).copy()
    df = df.sort_values(DATE_COL, kind="mergesort").reset_index(drop=True)
    log(f"  Sorted span: {df[DATE_COL].min().date()} -> {df[DATE_COL].max().date()}")

    train, oot = chronological_split(df)
    log(
        f"  Train default rate={train[TARGET_COL].mean():.4f} | "
        f"OOT default rate={oot[TARGET_COL].mean():.4f}"
    )

    log("\n[3] Training-only State and NAICS_Sector risk points")
    train["NAICS_Sector"] = extract_naics_sector(train["NAICS"])
    oot["NAICS_Sector"] = extract_naics_sector(oot["NAICS"])

    state_map = build_entity_points(train, "State", "State")
    sector_map = build_entity_points(train, "NAICS_Sector", "NAICS_Sector")

    train["State_Points"] = map_points(train["State"], state_map)
    oot["State_Points"] = map_points(oot["State"], state_map)
    train["NAICS_Sector_Points"] = map_points(train["NAICS_Sector"], sector_map)
    oot["NAICS_Sector_Points"] = map_points(oot["NAICS_Sector"], sector_map)

    unseen_states = sorted(set(oot["State"].dropna()) - set(state_map))
    unseen_sectors = sorted(set(oot["NAICS_Sector"].dropna()) - set(sector_map))
    log(f"  Unseen OOT states (scored {UNSEEN_POINTS}): {unseen_states or 'none'}")
    log(f"  Unseen OOT sectors (scored {UNSEEN_POINTS}): {unseen_sectors or 'none'}")

    log("\n[4] Row-wise feature engineering (applied independently to each split)")
    train = engineer_features(train)
    oot = engineer_features(oot)

    log("\n[5] Drop leakage, identifiers, and consumed originals")
    train = drop_non_features(train)
    oot = drop_non_features(oot)

    y_train = train.pop(TARGET_COL).astype(int).reset_index(drop=True)
    y_oot = oot.pop(TARGET_COL).astype(int).reset_index(drop=True)
    X_train = train.reset_index(drop=True)
    X_oot = oot.reset_index(drop=True)

    if list(X_train.columns) != list(X_oot.columns):
        raise RuntimeError("Train/OOT feature columns diverged after cleanup.")

    log(f"  Final feature columns ({len(X_train.columns)}): {list(X_train.columns)}")

    log("\n[6] Export (NaNs preserved; no imputer / scaler)")
    save_split(X_train, y_train, "train")
    save_split(X_oot, y_oot, "oot")

    log("\n" + "=" * 78)
    log(
        f"Done. Kept {len(X_train) + len(X_oot):,} / {n_raw:,} raw rows. "
        "Do not fit imputers or scalers on these files until CV folds are formed."
    )
    log("=" * 78)


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:
        print(f"\nPreprocessing failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1) from exc
