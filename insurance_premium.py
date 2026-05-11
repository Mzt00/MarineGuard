
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Optional

import pandas as pd
import numpy as np



REGION_MULTIPLIERS: dict[str, float] = {
    "Gulf of Aden":                2.50,  
    "Indian Ocean":                2.00,  
    "Middle East & North Africa":  1.90,   
    "West Africa":                 2.00,   
    "South Asia":                  1.60,   
    "Malacca Strait":              1.80,   
    "East Asia & Pacific":         1.40,   
    "South America":               1.30,  
    "Central America & Caribbean": 1.20,
    "Sub-Saharan Africa":          1.70,   
    "Europe & Central Asia":       1.05,
    "North America":               1.00,   
    "default":                     1.00,  
}

#Default economic parameters
DEFAULT_LGD       = 0.35       # Loss Given Default(ICC IMB 2020)
DEFAULT_THETA     = 0.20       # safety loading (Swiss Re 2011)
DEFAULT_BASE_RATE = 0.0005    
DEFAULT_INSURED_VALUE = 25_000_000.0  # USD  representative bulk carrier hull


@dataclass
class PremiumResult:
  
    p_attack:           float 
    insured_value_usd:  float 
    region:             str   
    lgd:                float  
    theta:              float 
    lambda_region:      float  
    expected_loss_usd:  float  #p × V × LGD × λ
    risk_premium_usd:   float  
    base_premium_usd:   float 
    total_premium_usd:  float  
    premium_rate_pct:   float 
    risk_band:          str    

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        """Human-readable one-block summary."""
        lines = [
            "── Insurance Premium Summary ──────────────────────────────",
            f"  Region              : {self.region}",
            f"  Insured Value       : ${self.insured_value_usd:>15,.2f}",
            f"  Attack Probability  : {self.p_attack * 100:.2f}%",
            f"  Risk Band           : {self.risk_band}",
            f"  Loss-Given-Default  : {self.lgd * 100:.0f}%",
            f"  Regional Multiplier : ×{self.lambda_region:.2f}  (JWC / Stopford 2009)",
            f"  Safety Loading (θ)  : {self.theta * 100:.0f}%  (Swiss Re 2011)",
            "──────────────────────────────────────────────────────────",
            f"  Expected Loss       : ${self.expected_loss_usd:>15,.2f}",
            f"  Risk Premium        : ${self.risk_premium_usd:>15,.2f}",
            f"  Base War-Risk Premium: ${self.base_premium_usd:>14,.2f}",
            "──────────────────────────────────────────────────────────",
            f"  TOTAL VOYAGE PREMIUM: ${self.total_premium_usd:>15,.2f}",
            f"  Premium Rate        : {self.premium_rate_pct:.4f}% of insured value",
            "──────────────────────────────────────────────────────────",
        ]
        return "\n".join(lines)


def _assign_risk_band(p_attack: float) -> str:
    
    pct = p_attack * 100
    if pct < 25:
        return "Low"
    elif pct < 50:
        return "Moderate"
    elif pct < 75:
        return "High"
    else:
        return "Critical"


def calculate_premium(
    p_attack:       float,
    region:         str   = "default",
    insured_value:  float = DEFAULT_INSURED_VALUE,
    lgd:            float = DEFAULT_LGD,
    theta:          float = DEFAULT_THETA,
    base_rate:      float = DEFAULT_BASE_RATE,
) -> PremiumResult:
 
    if not (0.0 <= p_attack <= 1.0):
        raise ValueError(f"p_attack must be in [0, 1]; got {p_attack}")
    if insured_value <= 0:
        raise ValueError(f"insured_value must be > 0; got {insured_value}")
    if not (0.0 < lgd <= 1.0):
        raise ValueError(f"lgd must be in (0, 1]; got {lgd}")
    if theta < 0:
        raise ValueError(f"theta (safety loading) must be >= 0; got {theta}")
    if base_rate < 0:
        raise ValueError(f"base_rate must be >= 0; got {base_rate}")

   
    lambda_region = REGION_MULTIPLIERS.get(region, REGION_MULTIPLIERS["default"])


    expected_loss = p_attack * insured_value * lgd * lambda_region

  
    risk_premium = expected_loss * (1.0 + theta)

    
    base_premium = insured_value * base_rate

    total_premium = risk_premium + base_premium

   
    premium_rate_pct = (total_premium / insured_value) * 100.0

    return PremiumResult(
        p_attack=round(p_attack, 6),
        insured_value_usd=round(insured_value, 2),
        region=region,
        lgd=round(lgd, 4),
        theta=round(theta, 4),
        lambda_region=round(lambda_region, 4),
        expected_loss_usd=round(expected_loss, 2),
        risk_premium_usd=round(risk_premium, 2),
        base_premium_usd=round(base_premium, 2),
        total_premium_usd=round(total_premium, 2),
        premium_rate_pct=round(premium_rate_pct, 6),
        risk_band=_assign_risk_band(p_attack),
    )


def append_premium_columns(
    df:             pd.DataFrame,
    insured_value:  float = DEFAULT_INSURED_VALUE,
    lgd:            float = DEFAULT_LGD,
    theta:          float = DEFAULT_THETA,
    base_rate:      float = DEFAULT_BASE_RATE,
    prob_col:       str   = "attack_probability_pct",
    region_col:     str   = "region",
) -> pd.DataFrame:
    df = df.copy()
    p_vec = df[prob_col].clip(0, 100) / 100.0

    lambda_vec = (
        df[region_col]
        .map(REGION_MULTIPLIERS)
        .fillna(REGION_MULTIPLIERS["default"])
    )
    expected_loss_vec = p_vec * insured_value * lgd * lambda_vec
    risk_premium_vec  = expected_loss_vec * (1.0 + theta)
    base_premium_vec  = insured_value * base_rate          # scalar × len(df)
    total_premium_vec = risk_premium_vec + base_premium_vec

    df["lambda_region"]     = lambda_vec.round(4)
    df["expected_loss_usd"] = expected_loss_vec.round(2)
    df["risk_premium_usd"]  = risk_premium_vec.round(2)
    df["base_premium_usd"]  = round(base_premium_vec, 2)
    df["total_premium_usd"] = total_premium_vec.round(2)
    df["premium_rate_pct"]  = (total_premium_vec / insured_value * 100).round(6)

    return df


def print_premium_statistics(df: pd.DataFrame) -> None:
    req_cols = ["total_premium_usd", "premium_rate_pct",
                "expected_loss_usd", "risk_premium_usd"]
    missing  = [c for c in req_cols if c not in df.columns]
    if missing:
        print(f"  [premium stats] Missing columns: {missing}")
        return

    sep = "─" * 65
    print(f"\n{sep}")
    print("  INSURANCE PREMIUM STATISTICS  (default V=$25M, LGD=35%, θ=20%)")
    print(sep)

    desc = df[req_cols].describe()
    for col in req_cols:
        mean = desc.loc["mean", col]
        mn   = desc.loc["min",  col]
        mx   = desc.loc["max",  col]
        std  = desc.loc["std",  col]
        lbl  = col.replace("_", " ").title()
        print(f"  {lbl:<28}: mean=${mean:>12,.0f}  "
              f"std=${std:>12,.0f}  "
              f"[${mn:>12,.0f}  –  ${mx:>12,.0f}]")

    print(sep)
    by_band = (
        df.groupby("risk_band")["total_premium_usd"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "Avg Premium", "count": "N"})
        .reindex(["Low", "Moderate", "High", "Critical"], fill_value=0)
    )
    print("  Average Premium by Risk Band:")
    for band, row in by_band.iterrows():
        print(f"    {band:<12}: avg=${row['Avg Premium']:>12,.0f}  "
              f"(N={int(row['N']):,})")
    print(sep)
