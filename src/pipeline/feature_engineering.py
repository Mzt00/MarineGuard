"""
Standards applied
      ICC-IMB Annual Report 2023, Appendix 1
      Attack-type severity hierarchy and IMB-designated high-risk corridors.
IMO MSC-FAL.1/Circ.3/Rev.8
      Vessel-status vulnerability classification for reporting piracy incidents.
    ISO 31000:2018 §6.4.2
      Risk evaluation formula: Risk = Likelihood * Consequence
      Implemented as: risk_index = likelihood * consequence  (both normalised [0, 1])

Public endpoints

  engineer(df)          To pd.DataFrame   (original columns + engineered columns)
  get_model_features()  To list[str]      (feature columns for risk model)
  get_lloyd_features()  To list[str]      (feature columns for premium calculator)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# 5 = vessel seized / crew held for ransom or physically harmed
#4 = weapons discharged / explosive device
#3 = successful boarding (cargo/crew threatened at close quarters)
#2 = boarding in progress (perpetrators on board, outcome interrupted)
# 1 = crew detained by authorities (non-piracy incident)
#0 = suspicious approach (no physical contact)
SEVERITY_MAP: dict[str, int] = {
    "Hijacked":    5,
    "Fired Upon":  4,
    "Explosion":   4,
    "Boarded":     3,  
    "Boarding":    3,   
    "Detained":    1,
    "Attempted":   2,
    "Suspicious":  0,
}
_SEVERITY_MAX = max(SEVERITY_MAP.values())  
VULNERABILITY_MAP: dict[str, int] = {
    "Anchored":             3,   
    "Berthed":              3,  
    "Moored":               3,
    "Stationary":           3,
    "Drifting":             3,   
    "Grounded":             3,   
    "Bunkering Operations": 2,  
    "Fishing":              2,   
    "Towed":                2,  
    "Underway":             1,   
    "Steaming":             0,  
}
_VULNERABILITY_MAX = max(VULNERABILITY_MAP.values())   


HIGH_RISK_ZONES: dict[str, tuple[float, float, float, float]] = {
    "Strait of Malacca":     (1.0,   6.0,  98.0, 105.0),
    "Singapore Strait":      (1.0,   2.0, 103.0, 105.0),
    "Gulf of Guinea":        (-5.0,  5.0,  -5.0,   9.0),
    "Gulf of Aden":          (11.0, 16.0,  43.0,  52.0),
    "Somali Basin":          ( 2.0, 15.0,  50.0,  65.0),
    "Bay of Bengal":         ( 5.0, 23.0,  80.0,  95.0),
    "Sulu-Celebes Sea":      ( 4.0, 12.0, 117.0, 127.0),
}


#shore distance in kms
_PROX_BINS   = [0,   10,   50,  200, np.inf]
_PROX_LABELS = ["Coastal", "Near-Shore", "Offshore", "Open-Sea"]


def _add_temporal(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy()
    df["quarter"] = ((df["month"] - 1) // 3 + 1).astype("Int8")
    df["decade"]  = (df["year"] // 10 * 10).astype("Int16")
    return df


def _add_severity(df: pd.DataFrame) -> pd.DataFrame:
    #severity_score [0, 1]
   
    df = df.copy()
    raw = df["attack_type"].map(SEVERITY_MAP).fillna(0).astype(float)
    df["severity_score"] = raw / _SEVERITY_MAX
    return df


def _add_vulnerability(df: pd.DataFrame) -> pd.DataFrame:
    #vulnerability_score [0, 1]
   
    df = df.copy()
    raw = df["vessel_status"].map(VULNERABILITY_MAP)
    median_fallback = np.median(list(VULNERABILITY_MAP.values()))
    raw = raw.fillna(median_fallback).astype(float)
    df["vulnerability_score"] = raw / _VULNERABILITY_MAX
    return df


def _add_proximity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    proximity_risk [0, 1]

    proximity_band : category
        Ordinal bin for EDA (Coastal / Near-Shore / Offshore / Open-Sea)
    """
    df = df.copy()
    d = df["shore_distance"].clip(lower=0.0)
    d_max = np.percentile(d.dropna(), 99)
    df["proximity_risk"] = (1.0 - np.log1p(d) / np.log1p(d_max)).clip(0.0, 1.0)

    df["proximity_band"] = pd.cut(
        d,
        bins=_PROX_BINS,
        labels=_PROX_LABELS,
        right=False,
    ).astype("category")

    return df


def _add_zone_features(df: pd.DataFrame) -> pd.DataFrame:
   #returns true if the region is  an IMB designated high risk region

   
    
    df = df.copy()
    zone_col = pd.Series("Other", index=df.index, dtype="object")

    lat = df["latitude"]
    lon = df["longitude"]

    for zone, (lat_min, lat_max, lon_min, lon_max) in HIGH_RISK_ZONES.items():
        in_zone = (
            lat.between(lat_min, lat_max) &
            lon.between(lon_min, lon_max) &
            (zone_col == "Other")          # first match wins
        )
        zone_col[in_zone] = zone

    df["zone_name"]      = zone_col.astype("category")
    df["high_risk_zone"] = (zone_col != "Other")
    return df


def _add_regional_attack_rate(df: pd.DataFrame) -> pd.DataFrame:
   #attack rates in a specific region [0,1]
    df = df.copy()
    total = len(df)

   
    region_counts = df["region"].astype(str).value_counts()
    df["regional_attack_rate"] = (
        df["region"].astype(str).map(region_counts) / total
    ).astype(float)

    
    annual_counts = df["year"].astype(int).value_counts()
    yr_min, yr_max = annual_counts.min(), annual_counts.max()
    denom = max(yr_max - yr_min, 1)        
    df["annual_trend_factor"] = (
        (df["year"].astype(int).map(annual_counts) - yr_min) / denom
    ).astype(float)

    return df


def _add_risk_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    likelihood  = df["regional_attack_rate"] * df["proximity_risk"]
    consequence = df["severity_score"]       * df["vulnerability_score"]

    def _norm(s: pd.Series) -> pd.Series:
        lo, hi = s.min(), s.max()
        return (s - lo) / max(hi - lo, 1e-9)

    df["likelihood"]  = _norm(likelihood)
    df["consequence"] = _norm(consequence)
    df["risk_index"]  = _norm(df["likelihood"] * df["consequence"])

    return df




MODEL_FEATURES: list[str] = [ #featuers for regression/classification
    "severity_score",
    "vulnerability_score",
    "proximity_risk",
    "regional_attack_rate",
    "annual_trend_factor",
    "high_risk_zone",
    "likelihood",
    "consequence",
    "risk_index",
]


LLOYD_FEATURES: list[str] = [ #features for lloyds formula
    "risk_index",
    "likelihood",
    "consequence",
    "severity_score",
    "regional_attack_rate",
]

EDA_FEATURES: list[str] = [ #will be used in eda
    "quarter",
    "decade",
    "proximity_band",
    "zone_name",
]


#main entry point

def engineer(df: pd.DataFrame) -> pd.DataFrame:
   
    required = {
        "year", "month", "longitude", "latitude",
        "attack_type", "vessel_status", "shore_distance", "region",
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"[feature_engineering] Missing required columns: {missing}")

    df = _add_temporal(df)
    df = _add_severity(df)
    df = _add_vulnerability(df)
    df = _add_proximity_features(df)
    df = _add_zone_features(df)
    df = _add_regional_attack_rate(df)
    df = _add_risk_index(df)

    engineered = MODEL_FEATURES + EDA_FEATURES
    print(f"[feature_engineering] Added {len(engineered)} columns: {engineered}")
    print(
        f"[feature_engineering] risk_index  — "
        f"min={df['risk_index'].min():.4f}  "
        f"mean={df['risk_index'].mean():.4f}  "
        f"max={df['risk_index'].max():.4f}"
    )
    return df


def get_model_features() -> list[str]:
    """Return the list of numeric feature columns for risk models."""
    return MODEL_FEATURES.copy()


def get_lloyd_features() -> list[str]:
    """Return the list of feature columns required by lloyds_formula.py."""
    return LLOYD_FEATURES.copy()

if __name__ == "__main__":
    from loader import load, summary   # noqa: PLC0415

    raw = load()
    engineered = engineer(raw)
    summary(engineered)

    print("\nSample risk_index by region")
    print(
        engineered.groupby("region")["risk_index"]
        .agg(["mean", "median", "max"])
        .sort_values("mean", ascending=False)
        .round(4)
    )

    print("\nZone coverage")
    print(engineered["zone_name"].value_counts())

    print("\nProximity band distribution")
    print(engineered["proximity_band"].value_counts().sort_index())