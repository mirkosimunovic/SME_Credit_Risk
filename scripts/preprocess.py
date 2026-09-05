"""Leak-proof chronological preprocessing for SBA SME credit-risk modeling.

Reads data/raw/SBAnational.csv, defines MIS_Status (1 = Default), splits the
oldest 95% vs newest 5% (out-of-time), and computes State / NAICS-sector risk
points from the training split only.

Feature-engineering source columns (Term, GrAppv, NAICS, LowDoc, NewExist,
FranchiseCode, CreateJob, RetainedJob, State, RevLineCr, ...) are KEPT in
their original form. Only identifiers and post-decision leakage columns are
dropped.

No imputer or StandardScaler is applied here. NaNs are preserved so the
training script can impute inside each CV fold.

Target: MIS_Status (0 = Paid in Full, 1 = Default).
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths and split constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "SBAnational.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TARGET_COL = "MIS_Status"
DATE_COL = "ApprovalDate"

# Newest 5% is the OOT holdout; oldest 95% is the modeling sample.
OOT_FRACTION = 0.05
UNSEEN_POINTS = 3  # midpoint / "average" quintile for unseen or missing keys
YEAR_CAP = 2024    # two-digit years parsed into the future are rolled back 100y

# Post-decision fields that reveal the outcome (or are determined after default).
# SBA_Appv is dropped after Guarantee_Ratio is built (it would be redundant).
LEAKAGE_COLS = [
    "ChgOffDate",
    "ChgOffPrinGr",
    "DisbursementDate",
    "DisbursementGross",
    "BalanceGross",
    "SBA_Appv",
]

# Unique IDs, free-text names, and high-cardinality location/lender keys.
# ApprovalFY is a date proxy and would leak calendar time into the features.
# ApprovalDate is dropped after the chronological split for the same reason.
IDENTIFIER_COLS = [
    "LoanNr_ChkDgt",
    "Name",
    "Zip",
    "City",
    "Bank",
    "BankState",
    "ApprovalFY",
    DATE_COL,
]


# ---------------------------------------------------------------------------
# Small helpers — used more than once so we do not copy-paste the same logic
# ---------------------------------------------------------------------------
def parse_currency(series: pd.Series) -> pd.Series:
    """Strip $, commas, and whitespace from SBA money strings, then cast to float."""
    cleaned = (
        series.astype(str)
        .str.replace(r"[\$,\s]", "", regex=True)
        .str.replace(r"[^0-9.\-]", "", regex=True)
        .replace({"": np.nan, "nan": np.nan, "None": np.nan, "NaN": np.nan})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def assign_quintile_points(rates: pd.Series) -> dict:
    """Turn entity-level default rates into 1–5 risk points.

    Lowest-default quintile -> 5 (safer). Highest-default quintile -> 1 (riskier).
    If qcut cannot form 5 distinct bins, we stretch the available bins onto 1–5.
    """
    rates = rates.dropna()
    if rates.empty:
        return {}
    if rates.nunique() == 1:
        return {key: UNSEEN_POINTS for key in rates.index}

    q = min(5, int(rates.nunique()))
    try:
        codes = pd.qcut(rates, q=q, labels=False, duplicates="drop")
    except ValueError:
        return {key: UNSEEN_POINTS for key in rates.index}

    n_bins = int(codes.max()) - int(codes.min()) + 1
    if n_bins <= 1:
        return {key: UNSEEN_POINTS for key in rates.index}

    # codes == 0 is the lowest default rate. Map that onto 5 points.
    points = 5 - np.round(codes.to_numpy() * (4 / (n_bins - 1))).astype(int)
    points = np.clip(points, 1, 5)
    return dict(zip(rates.index.tolist(), points.tolist()))


def apply_points(keys: pd.Series, mapping: dict) -> pd.Series:
    """Map keys through a training-only dictionary; unseen / missing -> 3."""
    return pd.to_numeric(keys.map(mapping), errors="coerce").fillna(UNSEEN_POINTS).astype(int)


def save_xy(X: pd.DataFrame, y: pd.Series, split_name: str) -> None:
    """Write X_{split}.csv and y_{split}.csv, keeping NaNs as empty cells."""
    x_path = PROCESSED_DIR / f"X_{split_name}.csv"
    y_path = PROCESSED_DIR / f"y_{split_name}.csv"
    X.to_csv(x_path, index=False)
    y.to_frame(name=TARGET_COL).to_csv(y_path, index=False)
    nan_cols = X.columns[X.isna().any()].tolist()
    print(f"  {x_path.relative_to(PROJECT_ROOT)}  shape={X.shape}  NaN cells={int(X.isna().sum().sum()):,}")
    print(f"  {y_path.relative_to(PROJECT_ROOT)}  default rate={float(y.mean()):.4f}")
    if nan_cols:
        print(f"    Columns still containing NaNs ({split_name}): {nan_cols}")


# ===========================================================================
# Sequential pipeline
# ===========================================================================
def main() -> None:
    os.chdir(PROJECT_ROOT)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw dataset not found: {RAW_PATH}")

    print("=" * 78)
    print("SBA SME preprocessing — chronological, leak-proof")
    print("=" * 78)

    # -----------------------------------------------------------------------
    # Step 1. Load raw loans and define the default target
    # -----------------------------------------------------------------------
    # The raw MIS_Status field uses 'CHGOFF' and 'P I F' (spaces in PIF).
    # We normalize whitespace so both 'PIF' and 'P I F' become class 0.
    print(f"\n[1] Load {RAW_PATH.relative_to(PROJECT_ROOT)}")
    df = pd.read_csv(RAW_PATH, low_memory=False)
    n_raw = len(df)
    print(f"  Raw shape: {df.shape}")

    mis_token = (
        df["MIS_Status"]
        .astype(str)
        .str.upper()
        .str.replace(r"\s+", "", regex=True)
        .str.strip()
        .replace({"NAN": np.nan, "NONE": np.nan, "": np.nan})
    )
    df["MIS_Status"] = mis_token.map({"CHGOFF": 1, "PIF": 0})

    n_before = len(df)
    df = df.dropna(subset=["MIS_Status"]).copy()
    df["MIS_Status"] = df["MIS_Status"].astype(int)
    print(f"  Dropped {n_before - len(df):,} rows with missing / unmapped MIS_Status.")
    print(
        f"  Paid in Full (0)={(df['MIS_Status'] == 0).sum():,} | "
        f"Default (1)={(df['MIS_Status'] == 1).sum():,}"
    )

    # Currency strings must be numeric before Guarantee_Ratio or log(GrAppv).
    print("\n[1b] Parse GrAppv and SBA_Appv currency strings -> float")
    for col in ("GrAppv", "SBA_Appv"):
        df[col] = parse_currency(df[col])
        print(f"  {col}: non-null={df[col].notna().sum():,}  min={df[col].min()}  max={df[col].max()}")

    # -----------------------------------------------------------------------
    # Step 2. Sort by ApprovalDate, then cut oldest 95% / newest 5%
    # -----------------------------------------------------------------------
    # Group default rates (State, NAICS sector) are computed AFTER this split,
    # using the training slice only. That is what blocks temporal leakage.
    print("\n[2] Parse ApprovalDate, sort oldest -> newest, chronological split")
    raw_dates = df["ApprovalDate"].astype(str).str.strip().replace({"": np.nan, "nan": np.nan, "NaT": np.nan})

    # Primary format in this file is '10-Jul-97' / '2-Jun-80' (%d-%b-%y).
    df["ApprovalDate"] = pd.to_datetime(raw_dates, format="%d-%b-%y", errors="coerce")
    still_bad = df["ApprovalDate"].isna() & raw_dates.notna()
    if still_bad.any():
        df.loc[still_bad, "ApprovalDate"] = pd.to_datetime(raw_dates.loc[still_bad], errors="coerce")

    # %y maps 00–68 to 2000–2068. A handful of 1960s loans would land in 206x.
    future_mask = df["ApprovalDate"].notna() & (df["ApprovalDate"].dt.year > YEAR_CAP)
    if future_mask.any():
        df.loc[future_mask, "ApprovalDate"] = df.loc[future_mask, "ApprovalDate"].apply(
            lambda ts: ts.replace(year=ts.year - 100)
        )
        print(f"  Rolled {int(future_mask.sum()):,} two-digit years back by 100 years.")

    n_bad_dates = int(df["ApprovalDate"].isna().sum())
    if n_bad_dates:
        print(f"  Dropping {n_bad_dates:,} rows with unparseable ApprovalDate.")
        df = df.dropna(subset=["ApprovalDate"]).copy()

    # mergesort is stable so loans sharing a date keep their relative order.
    df = df.sort_values("ApprovalDate", kind="mergesort").reset_index(drop=True)
    print(f"  Sorted span: {df['ApprovalDate'].min().date()} -> {df['ApprovalDate'].max().date()}")

    n = len(df)
    cut = int(n * (1.0 - OOT_FRACTION))  # last 5% of the timeline is OOT
    if cut <= 0 or cut >= n:
        raise ValueError(f"Invalid chronological cut {cut} for n={n}.")

    train = df.iloc[:cut].copy()
    oot = df.iloc[cut:].copy()
    print(
        f"  Train (oldest {1 - OOT_FRACTION:.0%}): "
        f"{train['ApprovalDate'].min().date()} -> {train['ApprovalDate'].max().date()}  n={len(train):,}"
    )
    print(
        f"  OOT   (newest {OOT_FRACTION:.0%}): "
        f"{oot['ApprovalDate'].min().date()} -> {oot['ApprovalDate'].max().date()}  n={len(oot):,}"
    )
    print(
        f"  Default rate  train={train['MIS_Status'].mean():.4f}  "
        f"OOT={oot['MIS_Status'].mean():.4f}"
    )
    if train["ApprovalDate"].max() > oot["ApprovalDate"].min():
        print("  Note: the cut date appears in both splits; stable sort is the tie-break.")

    # ApprovalDate has done its job. It must not become a model feature.
    train = train.drop(columns=["ApprovalDate"])
    oot = oot.drop(columns=["ApprovalDate"])

    # -----------------------------------------------------------------------
    # Step 3. Training-only State and NAICS-sector risk points
    # -----------------------------------------------------------------------
    # NAICS is a 6-digit industry code. The first two digits are the sector
    # (e.g. 72 = Accommodation and Food Services). 0 / missing -> Undefined.
    print("\n[3] NAICS_Sector + training-only target encoding (State, sector)")
    for frame in (train, oot):
        naics_num = pd.to_numeric(frame["NAICS"], errors="coerce")
        sector = pd.Series(0, index=frame.index, dtype=int)
        valid = naics_num.notna() & (naics_num > 0)
        sector.loc[valid] = (
            naics_num.loc[valid].astype(np.int64).astype(str).str[:2].astype(int)
        )
        frame["NAICS_Sector"] = sector

    # Default rate per State, computed on TRAIN rows only.
    state_rates = train.groupby("State", dropna=False)["MIS_Status"].mean()
    state_map = assign_quintile_points(state_rates)
    print(f"  State keys scored from train: {len(state_map)}")
    print("  Safest states (lowest train default rate):")
    print(state_rates.sort_values().head(5).to_string())
    print("  Riskiest states (highest train default rate):")
    print(state_rates.sort_values().tail(5).to_string())

    # Same idea for 2-digit NAICS sector.
    sector_rates = train.groupby("NAICS_Sector", dropna=False)["MIS_Status"].mean()
    sector_map = assign_quintile_points(sector_rates)
    print(f"  NAICS_Sector keys scored from train: {len(sector_map)}")
    print("  Safest sectors:")
    print(sector_rates.sort_values().head(5).to_string())
    print("  Riskiest sectors:")
    print(sector_rates.sort_values().tail(5).to_string())

    # Apply the *training* maps to both splits. Never refit on OOT.
    train["State_Points"] = apply_points(train["State"], state_map)
    oot["State_Points"] = apply_points(oot["State"], state_map)
    train["NAICS_Sector_Points"] = apply_points(train["NAICS_Sector"], sector_map)
    oot["NAICS_Sector_Points"] = apply_points(oot["NAICS_Sector"], sector_map)

    unseen_states = sorted(set(oot["State"].dropna()) - set(state_map))
    unseen_sectors = sorted(set(oot["NAICS_Sector"].dropna()) - set(sector_map))
    print(f"  Unseen OOT states  (score={UNSEEN_POINTS}): {unseen_states or 'none'}")
    print(f"  Unseen OOT sectors (score={UNSEEN_POINTS}): {unseen_sectors or 'none'}")

    # -----------------------------------------------------------------------
    # Step 4. Row-wise feature engineering (same formulas on train and OOT)
    # -----------------------------------------------------------------------
    # These transforms do not use group statistics, so they cannot leak.
    # Original source columns are left in place — trees can still split on
    # Term months, raw GrAppv, FranchiseCode, job counts, etc.
    print("\n[4] Feature engineering on both splits (originals kept)")
    for split_name, frame in (("train", train), ("oot", oot)):
        gr = pd.to_numeric(frame["GrAppv"], errors="coerce")
        sba = pd.to_numeric(frame["SBA_Appv"], errors="coerce")
        term = pd.to_numeric(frame["Term"], errors="coerce")
        create_job = pd.to_numeric(frame["CreateJob"], errors="coerce")
        retained = pd.to_numeric(frame["RetainedJob"], errors="coerce")
        franchise = pd.to_numeric(frame["FranchiseCode"], errors="coerce")

        # Log1p compresses the heavy right tail of approved loan size.
        frame["Log_GrAppv"] = np.log1p(gr.clip(lower=0))

        # Share of the bank approval that the SBA guarantees. GrAppv == 0 -> NaN.
        frame["Guarantee_Ratio"] = np.where(gr > 0, sba / gr, np.nan)

        # Term is stored in months; integer years is a more interpretable tenor.
        frame["Term_Years"] = np.floor_divide(term, 12)

        # Job and franchise flags. SBA codebook: 00000 or 00001 = no franchise.
        # So 0 and 1 are non-franchise; any code > 1 is a franchise.
        frame["IsCreateJob"] = np.where(create_job.isna(), np.nan, (create_job > 0).astype(float))
        frame["IsRetained"] = np.where(retained.isna(), np.nan, (retained > 0).astype(float))
        frame["IsFranchise"] = np.where(franchise.isna(), np.nan, (franchise > 1).astype(float))

        # LowDoc: only Y/N are valid. 'C', '0', '1', blanks stay NaN for the imputer.
        lowdoc_token = frame["LowDoc"].astype(str).str.upper().str.strip()
        lowdoc = pd.Series(np.nan, index=frame.index, dtype="float")
        lowdoc.loc[lowdoc_token.eq("Y")] = 1.0
        lowdoc.loc[lowdoc_token.eq("N")] = 0.0
        frame["LowDoc_Binary"] = lowdoc

        # NewExist: 1 = existing firm, 2 = new firm.
        # If the flag is 0/null but the firm retained >= 1 job, treat as existing.
        # Remaining 0s become NaN (not dropped).
        new_exist = pd.to_numeric(frame["NewExist"], errors="coerce")
        new_clean = pd.Series(np.nan, index=frame.index, dtype="float")
        new_clean.loc[new_exist.eq(1)] = 1.0
        new_clean.loc[new_exist.eq(2)] = 2.0
        salvage = (new_exist.isna() | new_exist.eq(0)) & retained.ge(1)
        new_clean.loc[salvage] = 1.0
        frame["NewExist_Clean"] = new_clean

        print(f"  Engineered columns added on {split_name}.")

    # -----------------------------------------------------------------------
    # Step 5. Drop leakage and identifiers only
    # -----------------------------------------------------------------------
    # We do NOT drop Term, GrAppv, NAICS, NAICS_Sector, State, LowDoc,
    # NewExist, FranchiseCode, CreateJob, RetainedJob, RevLineCr, UrbanRural,
    # or NoEmp. Those can still be informative in raw form; CV will impute
    # whatever NaNs remain (e.g. LowDoc, RevLineCr, NewExist).
    print("\n[5] Drop target-leakage and identifier columns")
    drop_cols = [c for c in LEAKAGE_COLS + IDENTIFIER_COLS if c in train.columns]
    print(f"  Dropping: {drop_cols}")
    train = train.drop(columns=drop_cols)
    oot = oot.drop(columns=drop_cols)

    # Separate X / y. Row order is already chronological and aligned.
    y_train = train.pop("MIS_Status").astype(int).reset_index(drop=True)
    y_oot = oot.pop("MIS_Status").astype(int).reset_index(drop=True)
    X_train = train.reset_index(drop=True)
    X_oot = oot.reset_index(drop=True)

    if list(X_train.columns) != list(X_oot.columns):
        raise RuntimeError("Train and OOT feature columns diverged after cleanup.")

    print(f"  Final feature columns ({len(X_train.columns)}): {list(X_train.columns)}")

    # -----------------------------------------------------------------------
    # Step 6. Export. Leave NaNs in the CSVs on purpose.
    # -----------------------------------------------------------------------
    print("\n[6] Write processed splits (no imputer, no scaler)")
    save_xy(X_train, y_train, "train")
    save_xy(X_oot, y_oot, "oot")

    print("\n" + "=" * 78)
    print(
        f"Done. Kept {len(X_train) + len(X_oot):,} / {n_raw:,} raw rows. "
        "Impute and scale later, inside each CV fold."
    )
    print("=" * 78)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nPreprocessing failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1) from exc
