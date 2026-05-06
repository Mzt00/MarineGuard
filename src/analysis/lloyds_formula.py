 
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"

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

BASE_RATE = 0.008

SEVERITY_WEIGHTS = {
    1: 2.1465,
    2: 0.0014,
    3: 0.8522,
}


def load_data(filename: str = "pirate_attacks_clean.csv") -> pd.DataFrame:
    path = PROC / filename
    df = pd.read_csv(path, dtype=DTYPE_MAP, low_memory=False)
    for col in CAT_COLS:
        df[col] = df[col].astype("category")
    return df


def compute_attack_probabilities(df: pd.DataFrame) -> dict:
    total_attacks = len(df)
    attack_by_region = df["region"].value_counts()
    return (attack_by_region / total_attacks).to_dict()


def compute_regional_factors(df: pd.DataFrame) -> dict:
    attack_by_region = df["region"].value_counts()
    max_attacks = attack_by_region.max()
    return (attack_by_region / max_attacks).to_dict()


def shore_distance_factor(distance_km: float) -> float:
    if distance_km < 10:
        return 2.0
    elif distance_km < 50:
        return 1.5
    elif distance_km < 200:
        return 1.0
    else:
        return 0.5


def calculate_premium(
    vessel_value: float,
    region: str,
    distance_km: float,
    severity_level: int,
    p_attack_dict: dict,
    regional_factors: dict,
) -> float:
    p_attack = p_attack_dict.get(region, 0.0)
    sev_weight = SEVERITY_WEIGHTS.get(severity_level, 1.0)
    reg_factor = regional_factors.get(region, 1.0)
    dist_factor = shore_distance_factor(distance_km)
    
    premium = BASE_RATE * vessel_value * p_attack * sev_weight * reg_factor * dist_factor
    return premium


def run():
    print("=== LLOYD'S PREMIUM CALCULATOR ===")
    print()
    
    df = load_data()
    p_attack = compute_attack_probabilities(df)
    reg_factors = compute_regional_factors(df)
    
    print("Attack Probability by Region:")
    for region, prob in sorted(p_attack.items(), key=lambda x: x[1], reverse=True):
        print(f"  {region:30s} {prob:.4f}")
    print()
    
    print("Regional Load Factors:")
    for region, factor in sorted(reg_factors.items(), key=lambda x: x[1], reverse=True):
        print(f"  {region:30s} {factor:.2f}")
    print()
    
    test_cases = [
        {
            "name": "High-Risk: East Asia, Close Shore",
            "vessel_value": 15_000_000,
            "region": "East Asia & Pacific",
            "distance": 8,
            "severity": 1,
        },
        {
            "name": "Medium-Risk: Sub-Saharan Africa, Medium Distance",
            "vessel_value": 12_000_000,
            "region": "Sub-Saharan Africa",
            "distance": 75,
            "severity": 2,
        },
        {
            "name": "Low-Risk: North America, Far Shore",
            "vessel_value": 20_000_000,
            "region": "North America",
            "distance": 250,
            "severity": 3,
        },
    ]
    
    print("=== TEST SCENARIOS ===")
    for scenario in test_cases:
        premium = calculate_premium(
            scenario["vessel_value"],
            scenario["region"],
            scenario["distance"],
            scenario["severity"],
            p_attack,
            reg_factors,
        )
        pct_of_value = (premium / scenario["vessel_value"]) * 100
        
        print()
        print(f"{scenario['name']}")
        print(f"  Vessel Value: ${scenario['vessel_value'] / 1e6:.1f}M")
        print(f"  Region: {scenario['region']}")
        print(f"  Distance: {scenario['distance']} km")
        print(f"  Severity Level: {scenario['severity']}")
        print(f"  Annual Premium: ${premium:,.2f}")
        print(f"  % of Vessel Value: {pct_of_value:.4f}%")


if __name__ == "__main__":
    run()