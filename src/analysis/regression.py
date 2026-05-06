import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

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
    return df


def prepare_data(df):
    X = df[["year", "month", "longitude", "latitude", "shore_distance", "vessel_status", "region"]].copy()
    
    X["log_distance"] = np.log1p(X["shore_distance"])
    X = X.drop("shore_distance", axis=1)
    
    le_vessel = LabelEncoder()
    le_region = LabelEncoder()
    X["vessel_status"] = le_vessel.fit_transform(X["vessel_status"].astype(str))
    X["region"] = le_region.fit_transform(X["region"].astype(str))
    
    y = df["attack_type"].map(ATTACK_SEVERITY).astype("int")
    
    return X, y, le_vessel, le_region


def train(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_train, X_test, y_train, y_test):
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    
    print("=== RISK SEVERITY MODEL ===")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy:  {test_acc:.4f}")
    print()
    print("Classification Report (Test Set):")
    print(classification_report(y_test, y_pred, target_names=["Level 1", "Level 2", "Level 3"], zero_division=0))
    print()
    print("Feature Importance:")
    for col, imp in sorted(zip(X_train.columns, model.feature_importances_), key=lambda x: x[1], reverse=True):
        print(f"  {col:20s} {imp:.4f}")


def run():
    df = load()
    X, y, le_v, le_r = prepare_data(df)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = train(X_train, y_train)
    evaluate(model, X_train, X_test, y_train, y_test)
    
    joblib.dump(model, MODELS / "severity_model.pkl")
    joblib.dump(le_v, MODELS / "le_vessel.pkl")
    joblib.dump(le_r, MODELS / "le_region.pkl")
    
    print()
    print(f"[save] Model saved to {MODELS / 'severity_model.pkl'}")


if __name__ == "__main__":
    run()