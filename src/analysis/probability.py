import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"

DTYPE_MAP = {
    "year": "Int16",
    "month": "Int8",
    "longitude": "float32",
    "latitude": "float32",
    "shore_distance": "float32",
    "nearest_country": "string",
}

CAT_COLS = ["attack_type", "vessel_status", "region"]

ATTACK_SEVERITY = {
    "Hijacked": 1, "Fired Upon": 1, "Boarded": 1,
    "Boarding": 1, "Detained": 1, "Explosion": 2,
    "Attempted": 3, "Suspicious": 3,
}


def load(filename: str = "pirate_attacks_clean.csv") -> pd.DataFrame:
    path = PROC / filename
    df = pd.read_csv(path, dtype=DTYPE_MAP, low_memory=False)
    for col in CAT_COLS:
        df[col] = df[col].astype("category")
    df["severity"] = df["attack_type"].map(ATTACK_SEVERITY).astype("Int8")
    return df


def attack_probability_by_region(df: pd.DataFrame) -> dict:
    total = len(df)
    attack_count = df["region"].value_counts()
    return (attack_count / total).to_dict()


def attack_rate_by_region(df: pd.DataFrame) -> dict:
    years = df["year"].max() - df["year"].min() + 1
    attack_count = df["region"].value_counts()
    return (attack_count / years).to_dict()


def poisson_model_by_region(df: pd.DataFrame, region: str) -> dict:
    region_data = df[df["region"] == region].groupby("year").size()
    lambda_param = region_data.mean()
    
    conf_lower = stats.poisson.ppf(0.025, lambda_param)
    conf_upper = stats.poisson.ppf(0.975, lambda_param)
    
    return {
        "lambda": lambda_param,
        "ci_lower": conf_lower,
        "ci_upper": conf_upper,
    }


def conditional_severity_prob(df: pd.DataFrame, region: str) -> dict:
    region_data = df[df["region"] == region]
    severity_dist = region_data["severity"].value_counts(normalize=True).sort_index()
    return severity_dist.to_dict()


def severity_distribution(df: pd.DataFrame) -> dict:
    dist = df["severity"].value_counts().sort_index()
    pct = (dist / len(df) * 100).round(2)
    return {int(sev): {"count": int(count), "pct": float(pct[sev])} for sev, count in dist.items()}


def run():
    print("=== PROBABILITY ANALYSIS ===")
    print()
    
    df = load()
    
    print("Attack Probability by Region (proportion of all attacks):")
    p_attack = attack_probability_by_region(df)
    for region, prob in sorted(p_attack.items(), key=lambda x: x[1], reverse=True):
        print(f"  {region:30s} {prob:.4f}")
    print()
    
    print("Attack Rate by Region (attacks/year):")
    attack_rate = attack_rate_by_region(df)
    for region, rate in sorted(attack_rate.items(), key=lambda x: x[1], reverse=True):
        print(f"  {region:30s} {rate:6.2f}")
    print()
    
    print("Poisson Model - Annual Attack Distribution (top 3):")
    for region in ["East Asia & Pacific", "Sub-Saharan Africa", "South Asia"]:
        model = poisson_model_by_region(df, region)
        print(f"  {region:30s} rate={model['lambda']:6.2f} [CI: {model['ci_lower']:.0f}-{model['ci_upper']:.0f}]")
    print()
    
    print("Severity Distribution:")
    sev_dist = severity_distribution(df)
    for sev, data in sev_dist.items():
        print(f"  Level {sev}: {data['count']:,} ({data['pct']:.2f}%)")
    print()
    
    print("Conditional Severity Probability - Top 3 Regions:")
    for region in ["East Asia & Pacific", "Sub-Saharan Africa", "South Asia"]:
        cond_prob = conditional_severity_prob(df, region)
        print(f"  {region}:")
        for sev, prob in cond_prob.items():
            print(f"    Level {sev}: {prob:.4f}")


if __name__ == "__main__":
    run()