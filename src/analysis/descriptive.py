 
import pandas as pd
import numpy as np
from pathlib import Path

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


def dataset_summary(df: pd.DataFrame) -> dict:
    return {
        "total_attacks": len(df),
        "time_period": f"{df['year'].min()}-{df['year'].max()}",
        "years": df["year"].max() - df["year"].min() + 1,
        "regions": df["region"].nunique(),
        "lat_range": (float(df["latitude"].min()), float(df["latitude"].max())),
        "lon_range": (float(df["longitude"].min()), float(df["longitude"].max())),
    }


def temporal_trends(df: pd.DataFrame) -> pd.DataFrame:
    yearly = df.groupby("year").size().reset_index(name="count")
    yearly["avg_per_month"] = yearly["count"] / 12
    return yearly


def seasonal_trends(df: pd.DataFrame) -> pd.DataFrame:
    monthly = df.groupby("month").size().reset_index(name="count")
    monthly["pct"] = (monthly["count"] / len(df) * 100).round(2)
    return monthly


def geographic_distribution(df: pd.DataFrame) -> pd.DataFrame:
    region_stats = df.groupby("region", observed=True).size().reset_index(name="count")
    region_stats["pct"] = (region_stats["count"] / len(df) * 100).round(2)
    return region_stats.sort_values("count", ascending=False)


def attack_type_distribution(df: pd.DataFrame) -> pd.DataFrame:
    attack_stats = df.groupby("attack_type", observed=True).size().reset_index(name="count")
    attack_stats["pct"] = (attack_stats["count"] / len(df) * 100).round(2)
    return attack_stats.sort_values("count", ascending=False)


def vessel_status_distribution(df: pd.DataFrame) -> pd.DataFrame:
    vessel_stats = df.groupby("vessel_status", observed=True).size().reset_index(name="count")
    vessel_stats["pct"] = (vessel_stats["count"] / len(df) * 100).round(2)
    return vessel_stats.sort_values("count", ascending=False)


def shore_distance_stats(df: pd.DataFrame) -> dict:
    return {
        "mean": float(df["shore_distance"].mean()),
        "median": float(df["shore_distance"].median()),
        "std": float(df["shore_distance"].std()),
        "min": float(df["shore_distance"].min()),
        "max": float(df["shore_distance"].max()),
        "q25": float(df["shore_distance"].quantile(0.25)),
        "q75": float(df["shore_distance"].quantile(0.75)),
    }


def severity_distribution(df: pd.DataFrame) -> pd.DataFrame:
    sev_stats = df.groupby("severity").size().reset_index(name="count")
    sev_stats["pct"] = (sev_stats["count"] / len(df) * 100).round(2)
    sev_stats.columns = ["level", "count", "pct"]
    return sev_stats.sort_values("level")


def run():
    print("=== DESCRIPTIVE STATISTICS ===")
    print()
    
    df = load()
    
    summary = dataset_summary(df)
    print("Dataset Overview:")
    print(f"  Total Attacks: {summary['total_attacks']:,}")
    print(f"  Time Period: {summary['time_period']} ({summary['years']} years)")
    print(f"  Regions: {summary['regions']}")
    lat_min, lat_max = summary["lat_range"]
    lon_min, lon_max = summary["lon_range"]
    print(f"  Geographic: {lat_min:.2f}° to {lat_max:.2f}° | {lon_min:.2f}° to {lon_max:.2f}°")
    print()
    
    print("Temporal Trends (last 5 years):")
    temporal = temporal_trends(df).tail(5)
    for _, row in temporal.iterrows():
        print(f"  {int(row['year'])}: {int(row['count']):,} ({row['avg_per_month']:.1f}/month)")
    print()
    
    print("Seasonal Distribution (by month):")
    seasonal = seasonal_trends(df)
    month_names = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                   7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
    for _, row in seasonal.iterrows():
        print(f"  {month_names[row['month']]}: {int(row['count']):,} ({row['pct']:.1f}%)")
    print()
    
    print("Geographic Distribution (top 5):")
    geo = geographic_distribution(df).head(5)
    for _, row in geo.iterrows():
        print(f"  {row['region']:30s} {int(row['count']):,} ({row['pct']:.1f}%)")
    print()
    
    print("Attack Type Distribution:")
    attacks = attack_type_distribution(df)
    for _, row in attacks.iterrows():
        print(f"  {row['attack_type']:20s} {int(row['count']):,} ({row['pct']:.1f}%)")
    print()
    
    print("Vessel Status Distribution:")
    vessel = vessel_status_distribution(df)
    for _, row in vessel.iterrows():
        print(f"  {row['vessel_status']:25s} {int(row['count']):,} ({row['pct']:.1f}%)")
    print()
    
    print("Shore Distance Statistics (km):")
    shore = shore_distance_stats(df)
    print(f"  Mean: {shore['mean']:.2f}")
    print(f"  Median: {shore['median']:.2f}")
    print(f"  Std Dev: {shore['std']:.2f}")
    print(f"  Range: {shore['min']:.2f} - {shore['max']:.2f}")
    print(f"  IQR: {shore['q25']:.2f} - {shore['q75']:.2f}")
    print()
    
    print("Severity Level Distribution:")
    sev = severity_distribution(df)
    for _, row in sev.iterrows():
        print(f"  Level {int(row['level'])}: {int(row['count']):,} ({row['pct']:.2f}%)")


if __name__ == "__main__":
    run()