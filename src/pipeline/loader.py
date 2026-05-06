import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"


DTYPE_MAP = {
    "year":            "Int16",
    "month":           "Int8",
    "longitude":       "float32",
    "latitude":        "float32",
    "shore_distance":  "float32",
    "nearest_country": "string",
}

CAT_COLS = ["attack_type", "vessel_status", "region"]


def load(filename: str = "pirate_attacks_clean.csv") -> pd.DataFrame:
    path = PROC / filename
    df = pd.read_csv(path, dtype=DTYPE_MAP, low_memory=False)

    for col in CAT_COLS:
        df[col] = df[col].astype("category")

    return df


def summary(df: pd.DataFrame) -> None:
    print(f"Shape       : {df.shape}")
    print(f"Years       : {df['year'].min()} – {df['year'].max()}")
    print(f"Regions     : {df['region'].nunique()}")
    print(f"Attack types: {df['attack_type'].nunique()}")
    print(f"Memory      : {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    print(f"\nDtypes:\n{df.dtypes}")
    print(f"\nNull counts:\n{df.isnull().sum()}")


if __name__ == "__main__":
    df = load()
    summary(df)