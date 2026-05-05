"""
Selected output columns
    year          extracted from `date`
    month         extracted from `date`
    longitude     decimal degrees
    latitude      decimal degrees
    attack_type   category
    vessel_status category
    shore_distance km (float)
    region        derived from `nearest_country` via country_codes.csv
"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT  = Path(__file__).resolve().parents[2]
RAW   = ROOT / "data" / "raw"
PROC  = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

SELECTED_RAW_COLS = [
    "date", "longitude", "latitude",
    "attack_type", "vessel_status", "shore_distance", "nearest_country",
]

NA_TOKENS = {"na", "n/a", "nan", "none", "unknown", "unspecified", "", "?", "null", " "}


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    attacks = pd.read_csv(RAW / "pirate_attacks.csv", low_memory=False)
    codes   = pd.read_csv(RAW / "country_codes.csv", low_memory=False)
    print(f"[load] pirate_attacks: {attacks.shape}")
    print(f"[load] country_codes : {codes.shape}")
    return attacks, codes


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in SELECTED_RAW_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")
    return df[SELECTED_RAW_COLS].copy()


def drop_unknown_rows(df: pd.DataFrame) -> pd.DataFrame:
   
    mask = df.apply(
        lambda col: col.astype(str).str.strip().str.lower().isin(NA_TOKENS)
    ).any(axis=1)

    dropped = mask.sum()
    print(f"[drop_unknown] Removed {dropped:,} rows containing unknown/empty/? values")
    return df[~mask].reset_index(drop=True)


def clean(df: pd.DataFrame, codes: pd.DataFrame) -> pd.DataFrame:

  
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.insert(0, "year",  df["date"].dt.year.astype("Int16"))
    df.insert(1, "month", df["date"].dt.month.astype("Int8"))
    df.drop(columns=["date"], inplace=True)

    
    for col in ("longitude", "latitude", "shore_distance"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.loc[df["longitude"].abs() > 180, "longitude"] = np.nan
    df.loc[df["latitude"].abs()  >  90, "latitude"]  = np.nan
    df.loc[df["shore_distance"]  <   0, "shore_distance"] = np.nan
    for col in ("attack_type", "vessel_status", "nearest_country"):
        df[col] = df[col].astype(str).str.strip().str.title()

    codes_clean = (
        codes[["country", "region"]]
        .dropna(subset=["country"])
        .assign(country=lambda x: x["country"].str.strip().str.upper())
    )

    df["_nc_upper"] = df["nearest_country"].str.upper()
    df = df.merge(codes_clean, left_on="_nc_upper", right_on="country", how="left")
    df.drop(columns=["_nc_upper", "country"], inplace=True)


    df = df[[
        "year", "month", "longitude", "latitude",
        "attack_type", "vessel_status", "shore_distance",
        "nearest_country", "region",
]].copy()

    df = drop_unknown_rows(df)

  
    before = len(df)
    df.dropna(inplace=True)
    print(f"[clean] dropna removed {before - len(df):,} additional NaN rows")
    print(f"[clean] Final dataset: {len(df):,} rows")


    df["attack_type"]   = df["attack_type"].astype("category")
    df["vessel_status"] = df["vessel_status"].astype("category")
    df["region"]        = df["region"].astype("category")

    return df.reset_index(drop=True)


def save(df: pd.DataFrame, filename: str = "pirate_attacks_clean.csv") -> Path:
    out = PROC / filename
    df.to_csv(out, index=False)
    print(f"[save] Saved → {out}")
    return out


def run() -> pd.DataFrame:
    attacks, codes = load_raw()
    df = select_columns(attacks)
    df = clean(df, codes)
    save(df)
    return df


if __name__ == "__main__":
    print("[start] Cleaning pipeline running...")
    run()
    print("[done] CSV generated successfully.")