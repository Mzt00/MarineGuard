"""
insurance_premium.py
====================
Maritime Piracy Insurance Premium Calculator
============================================

ACADEMIC BASIS
--------------
The premium model implements the actuarial Expected-Value (EV) pricing
principle with a safety loading, which is foundational in non-life insurance
pricing theory:

  Net Premium   = E[Loss]          [Bowers et al., 1997]
  Gross Premium = E[Loss] × (1+θ)  [safety-loading form; Bowers et al., 1997]

For maritime piracy war-risk policies the expected loss is decomposed as:

  E[Loss] = p_attack × V × LGD × λ_region

where each term has an empirical calibration source:

  p_attack    – ML-predicted probability that a piracy incident results in a
                completed attack (output of GradientBoostingClassifier).
  V           – Insured value (USD).  Typical range: $5M–$80M for bulk/tanker.
  LGD (Loss   – Fraction of V lost in a completed attack.  Calibrated at 0.35
  Given         (35%) following average hijacking / cargo loss estimates in
  Default)      ICC International Maritime Bureau Global Piracy Report 2020;
                also aligned with IMO MSC working paper estimates.
  λ_region    – JWC (Joint War Committee) breach-of-warranty regional
                multiplier, reflecting declared war/piracy-risk areas.
                Values derived from Stopford (2009) Maritime Economics 3rd ed.,
                Table 14.3, and updated from Lloyd's JWC Listed Areas (2023).
  θ (theta)   – Underwriter risk/profit safety loading (default 0.20 = 20%).
                Consistent with Swiss Re (2011) piracy-risk margin guidance
                and standard Lloyd's war-risk market practice.

An annual base war-risk rate r_base is added as a contractual floor:

  r_base = 0.05% p.a. of insured value  [IUMI Ocean Hull Statistics 2019]

Full voyage-premium formula:
  P_voyage = (p_attack × V × LGD × λ_region × (1 + θ)) + (V × r_base)

References:
  Bowers, N.L. et al. (1997). Actuarial Mathematics, 2nd ed. Society of Actuaries.
  Stopford, M. (2009). Maritime Economics, 3rd ed. Routledge, Ch. 14.
  ICC IMB (2020). Piracy and Armed Robbery Against Ships – Annual Report 2019.
  Swiss Re (2011). Understanding Piracy and Maritime Terrorism Risks.
  IUMI (2019). Ocean Hull Statistics Factfile.
  Lloyd's / JWC. Listed Areas for Hull War, Strikes & Related Perils (2023).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Optional

import pandas as pd
import numpy as np


# ── Regional JWC multipliers ──────────────────────────────────────────────────
# Source: Stopford (2009) Table 14.3 + Lloyd's JWC Listed Areas (2023).
# Multiplier embeds regional piracy frequency, consequence severity, and
# JWC breach-of-warranty premium structure.
REGION_MULTIPLIERS: dict[str, float] = {
    "Gulf of Aden":                2.50,   # JWC highest-risk Listed Area
    "Indian Ocean":                2.00,   # Somali Basin corridor
    "Middle East & North Africa":  1.90,   # includes Arabian Sea approaches
    "West Africa":                 2.00,   # Gulf of Guinea — fast-rising risk
    "South Asia":                  1.60,   # Bangladesh, India coastal waters
    "Malacca Strait":              1.80,   # Singapore Strait / Indonesia
    "East Asia & Pacific":         1.40,   # South China Sea, Philippines
    "South America":               1.30,   # Venezuela, Ecuador, Peru
    "Central America & Caribbean": 1.20,
    "Sub-Saharan Africa":          1.70,   # East/West Africa coastal blend
    "Europe & Central Asia":       1.05,
    "North America":               1.00,   # near-zero piracy risk baseline
    "default":                     1.00,   # fallback for unlisted regions
}

# Default economic parameters
DEFAULT_LGD       = 0.35       # Loss Given Default (ICC IMB 2020)
DEFAULT_THETA     = 0.20       # safety loading (Swiss Re 2011)
DEFAULT_BASE_RATE = 0.0005     # 0.05% p.a. base war-risk rate (IUMI 2019)
DEFAULT_INSURED_VALUE = 25_000_000.0  # USD — representative bulk carrier hull


@dataclass
class PremiumResult:
    """Structured result returned by calculate_premium()."""
    p_attack:           float  # input probability (0–1)
    insured_value_usd:  float  # V — insured value
    region:             str    # voyage region
    lgd:                float  # Loss-Given-Default fraction
    theta:              float  # safety loading
    lambda_region:      float  # JWC regional multiplier
    expected_loss_usd:  float  # p × V × LGD × λ
    risk_premium_usd:   float  # expected_loss × (1 + θ)
    base_premium_usd:   float  # V × r_base (annual floor)
    total_premium_usd:  float  # risk_premium + base_premium
    premium_rate_pct:   float  # total_premium / V × 100
    risk_band:          str    # "Low" / "Moderate" / "High" / "Critical"

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
    """Classify probability into IMB-aligned risk tiers."""
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
    """
    Compute a voyage-level piracy insurance premium.

    Formula (Bowers et al., 1997 + Stopford, 2009):
        P = (p × V × LGD × λ_region × (1 + θ)) + (V × r_base)

    Parameters
    ----------
    p_attack : float
        Attack completion probability in [0, 1] from the ML classifier.
    region : str
        Voyage / incident region.  Must match a key in REGION_MULTIPLIERS
        or falls back to "default" multiplier (1.0).
    insured_value : float
        Hull + cargo insured value in USD.  Default: $25,000,000.
    lgd : float
        Loss-Given-Default fraction.  Default: 0.35 (ICC IMB 2020).
    theta : float
        Underwriter safety / profit loading factor.  Default: 0.20 (Swiss Re 2011).
    base_rate : float
        Annual base war-risk rate as decimal.  Default: 0.0005 (IUMI 2019).

    Returns
    -------
    PremiumResult
        Dataclass with all intermediate and final premium components.

    Raises
    ------
    ValueError
        If p_attack is not in [0, 1], or insured_value <= 0, or lgd not in (0, 1].
    """
    # ── Input validation (system boundary) ────────────────────────────────────
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

    # ── JWC regional multiplier ───────────────────────────────────────────────
    lambda_region = REGION_MULTIPLIERS.get(region, REGION_MULTIPLIERS["default"])

    # ── Expected loss (actuarial EV principle; Bowers et al. 1997) ────────────
    expected_loss = p_attack * insured_value * lgd * lambda_region

    # ── Risk premium: expected loss + safety loading (Swiss Re 2011) ──────────
    risk_premium = expected_loss * (1.0 + theta)

    # ── Annual base war-risk floor (IUMI 2019) ─────────────────────────────────
    base_premium = insured_value * base_rate

    # ── Total voyage premium ──────────────────────────────────────────────────
    total_premium = risk_premium + base_premium

    # ── Expressed as %  of insured value ─────────────────────────────────────
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
    """
    Vectorised: adds insurance premium columns to a DataFrame that already
    contains attack_probability_pct and region columns.

    The function is vectorised using pandas/numpy operations for speed
    on the full 6,555-row dataset.

    New columns added
    -----------------
    lambda_region       : JWC regional multiplier for each row
    expected_loss_usd   : p × V × LGD × λ_region
    risk_premium_usd    : expected_loss × (1 + θ)
    base_premium_usd    : V × r_base  (constant per row — same insured value)
    total_premium_usd   : risk_premium + base_premium
    premium_rate_pct    : total_premium / V × 100

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns for prob_col and region_col.
    insured_value : float
        Insured hull + cargo value in USD for all rows.  Default: $25M.
    lgd, theta, base_rate : float
        Premium parameters (see calculate_premium docstring).
    prob_col : str
        Column holding attack probability in percentage points (0–100).
    region_col : str
        Column holding the region string.

    Returns
    -------
    pd.DataFrame  (copy with added columns)
    """
    df = df.copy()

    # Convert percentage → decimal probability
    p_vec = df[prob_col].clip(0, 100) / 100.0

    # Map each region to its JWC multiplier (vectorised)
    lambda_vec = (
        df[region_col]
        .map(REGION_MULTIPLIERS)
        .fillna(REGION_MULTIPLIERS["default"])
    )

    # Actuarial EV formula (vectorised)
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
    """
    Print a formatted summary of premium columns in a DataFrame produced
    by append_premium_columns().
    """
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
