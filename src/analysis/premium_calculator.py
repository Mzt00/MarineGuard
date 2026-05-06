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

BASE_RATE = 0.008

SEVERITY_WEIGHTS = {
    1: 2.1465,
    2: 0.0014,
    3: 0.8522,
}


def load_model_and_encoders():
    model = joblib.load(MODELS / "severity_model.pkl")
    le_vessel = joblib.load(MODELS / "le_vessel.pkl")
    le_region = joblib.load(MODELS / "le_region.pkl")
    return model, le_vessel, le_region


def load_baseline_data(filename: str = "pirate_attacks_clean.csv") -> pd.DataFrame:
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


def predict_severity(
    year: int,
    month: int,
    longitude: float,
    latitude: float,
    shore_distance: float,
    vessel_status: str,
    region: str,
    model,
    le_vessel,
    le_region,
) -> int:
    X_new = pd.DataFrame({
        "year": [year],
        "month": [month],
        "longitude": [longitude],
        "latitude": [latitude],
        "vessel_status": [le_vessel.transform([vessel_status])[0]],
        "region": [le_region.transform([region])[0]],
        "log_distance": [np.log1p(shore_distance)],
    })
    
    severity_pred = model.predict(X_new)[0]
    return int(severity_pred)


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


def estimate_premium(
    year: int,
    month: int,
    longitude: float,
    latitude: float,
    shore_distance: float,
    vessel_status: str,
    region: str,
    vessel_value: float,
    model=None,
    le_vessel=None,
    le_region=None,
    p_attack_dict=None,
    regional_factors=None,
) -> dict:
    
    if model is None:
        model, le_vessel, le_region = load_model_and_encoders()
    
    if p_attack_dict is None or regional_factors is None:
        df = load_baseline_data()
        p_attack_dict = compute_attack_probabilities(df)
        regional_factors = compute_regional_factors(df)
    
    severity = predict_severity(
        year, month, longitude, latitude, shore_distance,
        vessel_status, region, model, le_vessel, le_region
    )
    
    premium = calculate_premium(
        vessel_value, region, shore_distance, severity,
        p_attack_dict, regional_factors
    )
    
    return {
        "predicted_severity": severity,
        "annual_premium": premium,
        "premium_pct_of_value": (premium / vessel_value) * 100,
        "base_rate": BASE_RATE,
        "vessel_value": vessel_value,
        "region": region,
        "distance_km": shore_distance,
        "vessel_status": vessel_status,
    }


def run():
    print("=== PREMIUM CALCULATOR (REGRESSION + LLOYD'S) ===")
    print()
    
    model, le_v, le_r = load_model_and_encoders()
    df = load_baseline_data()
    p_attack = compute_attack_probabilities(df)
    reg_factors = compute_regional_factors(df)
    
    test_cases = [
        {
            "name": "Scenario 1: Steaming in East Asia, Close Shore",
            "year": 2023,
            "month": 6,
            "longitude": 110.0,
            "latitude": 10.0,
            "shore_distance": 8.5,
            "vessel_status": "Steaming",
            "region": "East Asia & Pacific",
            "vessel_value": 18_000_000,
        },
        {
            "name": "Scenario 2: Anchored in Sub-Saharan Africa, Medium Distance",
            "year": 2023,
            "month": 3,
            "longitude": 45.0,
            "latitude": -5.0,
            "shore_distance": 65.0,
            "vessel_status": "Anchored",
            "region": "Sub-Saharan Africa",
            "vessel_value": 25_000_000,
        },
        {
            "name": "Scenario 3: Berthed in North America, Far Shore",
            "year": 2023,
            "month": 9,
            "longitude": -75.0,
            "latitude": 35.0,
            "shore_distance": 320.0,
            "vessel_status": "Berthed",
            "region": "North America",
            "vessel_value": 12_000_000,
        },
    ]
    
    for scenario in test_cases:
        print(f"{scenario['name']}")
        result = estimate_premium(
            scenario["year"],
            scenario["month"],
            scenario["longitude"],
            scenario["latitude"],
            scenario["shore_distance"],
            scenario["vessel_status"],
            scenario["region"],
            scenario["vessel_value"],
            model=model,
            le_vessel=le_v,
            le_region=le_r,
            p_attack_dict=p_attack,
            regional_factors=reg_factors,
        )
        
        print(f"  Vessel: {scenario['vessel_status']} | Value: ${scenario['vessel_value'] / 1e6:.1f}M")
        print(f"  Location: {scenario['region']} ({scenario['shore_distance']}km from shore)")
        print(f"  Predicted Severity: Level {result['predicted_severity']}")
        print(f"  Annual Premium: ${result['annual_premium']:,.2f}")
        print(f"  % of Vessel Value: {result['premium_pct_of_value']:.4f}%")
        print()


if __name__ == "__main__":
    run()