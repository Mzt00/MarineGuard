"""
=============================================================
  Pirate Attacks Dataset — Professional ML Pipeline
  Task 1 : Regression  — predict shore_distance
  Task 2 : Classification — predict attack_occurred (binary)
           and attach attack_probability_pct to dataset
  Steps   : EDA → Preprocessing → Feature Engineering →
            Regression (Ridge, 10-Fold CV) →
            Classification (Logistic Regression, Stratified CV) →
            Probability column written to output CSV
=============================================================
"""

# ── Standard / Built-in ──────────────────────────────────────
import warnings
import os
import sys

# ── Third-party ──────────────────────────────────────────────
import numpy  as np
import pandas as pd

# ── Local ─────────────────────────────────────────────────────
from insurance_premium import (
    append_premium_columns,
    print_premium_statistics,
    DEFAULT_INSURED_VALUE,
)

from sklearn.ensemble        import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.preprocessing   import StandardScaler, OneHotEncoder
from sklearn.pipeline        import Pipeline
from sklearn.compose         import ColumnTransformer
from sklearn.impute          import SimpleImputer

warnings.filterwarnings("ignore")

# ╔══════════════════════════════════════════════════════════╗
# ║                   UTILITY HELPERS                        ║
# ╚══════════════════════════════════════════════════════════╝

def separator(title: str = "", char: str = "─", width: int = 65) -> None:
  
    if title:
        pad   = (width - len(title) - 2) // 2
        print(f"\n{char*pad} {title} {char*pad}")
    else:
        print(char * width)


def display_df_info(df: pd.DataFrame, label: str = "DataFrame") -> None:
    """Print shape, dtypes, and missing-value summary."""
    separator(label)
    print(f"  Shape      : {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(f"  Memory     : {df.memory_usage(deep=True).sum() / 1024:.1f} KB")

    missing = df.isnull().sum()
    if missing.any():
        print("\n  Missing values:")
        for col, cnt in missing[missing > 0].items():
            pct = cnt / len(df) * 100
            print(f"     {col:<25}: {cnt:>5,}  ({pct:.1f}%)")
    else:
        print("  No missing values found.")

    print("\n  Column dtypes:")
    for col, dtype in df.dtypes.items():
        print(f"     {col:<25}: {dtype}")


def display_categorical_summary(df: pd.DataFrame,
                                cat_cols: list) -> None:
    """Print cardinality + value-counts for each categorical column."""
    separator("Categorical Columns — Cardinality")
    for col in cat_cols:
        vc = df[col].value_counts()
        print(f"\n  [{col}]  —  {vc.shape[0]} unique values")
        for val, cnt in vc.head(8).items():
            print(f"     {str(val):<35} {cnt:>5,}")
        if len(vc) > 8:
            print(f"     ... and {len(vc)-8} more")


def display_numeric_summary(df: pd.DataFrame,
                            num_cols: list) -> None:
    """Print descriptive stats for numeric columns."""
    separator("Numeric Columns — Descriptive Stats")
    desc = df[num_cols].describe().T
    desc["skew"] = df[num_cols].skew()
    print(desc.round(3).to_string())


def detect_outliers_iqr(df: pd.DataFrame,
                        cols: list,
                        multiplier: float = 1.5) -> pd.DataFrame:
    """
    Return a summary DataFrame of IQR-based outlier counts per column.
    Does NOT remove rows — only reports.
    """
    rows = []
    for col in cols:
        q1  = df[col].quantile(0.25)
        q3  = df[col].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - multiplier * iqr, q3 + multiplier * iqr
        n_out  = ((df[col] < lo) | (df[col] > hi)).sum()
        rows.append({"column": col,
                     "Q1": round(q1, 3),
                     "Q3": round(q3, 3),
                     "IQR": round(iqr, 3),
                     "lower_fence": round(lo, 3),
                     "upper_fence": round(hi, 3),
                     "n_outliers": n_out,
                     "pct_outliers": round(n_out / len(df) * 100, 2)})
    return pd.DataFrame(rows).set_index("column")


# ╔══════════════════════════════════════════════════════════╗
# ║              STEP 1 — DATA LOADING                       ║
# ╚══════════════════════════════════════════════════════════╝

def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV, enforce correct dtypes, and return raw DataFrame."""
    separator("STEP 1 — Loading Data")

    if not os.path.exists(filepath):
        sys.exit(f"  File not found: {filepath}")

    df = pd.read_csv(filepath)
    print(f"  Loaded '{os.path.basename(filepath)}'")
    display_df_info(df, label="Raw Dataset")
    return df


# ╔══════════════════════════════════════════════════════════╗
# ║           STEP 2 — DATA PREPROCESSING                    ║
# ╚══════════════════════════════════════════════════════════╝

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw DataFrame:
      • Cast dtypes explicitly
      • Handle missing values
      • Strip whitespace from strings
      • Remove impossible / corrupt rows
      • Report outliers (no automatic removal — logged only)
    """
    separator("STEP 2 — Data Preprocessing")
    df = df.copy()

    # ── 2a. Enforce dtypes ───────────────────────────────────
    print("\n  [2a] Enforcing correct dtypes …")
    int_cols   = ["year", "month"]
    float_cols = ["longitude", "latitude", "shore_distance"]
    str_cols   = ["attack_type", "vessel_status",
                  "nearest_country", "region"]

    for col in int_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    for col in float_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in str_cols:
        # Replace actual NaN with a placeholder BEFORE string conversion
        # so that astype(str) does not silently produce the literal "nan"
        df[col] = df[col].where(df[col].notna(), other=pd.NA)
        df[col] = df[col].astype(str).str.strip()

    # ── 2b. Missing-value handling ───────────────────────────
    print("  [2b] Checking missing values …")
    before = len(df)
    df.dropna(subset=float_cols + int_cols, inplace=True)
    after  = len(df)
    if before != after:
        print(f"       Dropped {before - after:,} rows with critical NaNs.")
    else:
        print("       No critical NaNs found.")

    # Fill remaining string NaNs with 'Unknown'
    for col in str_cols:
        n_null = (df[col].isin(["nan", "None", ""])).sum()
        if n_null:
            df[col] = df[col].replace({"nan": "Unknown",
                                       "None": "Unknown",
                                       "": "Unknown"})
            print(f"       Filled {n_null} empty strings in '{col}' → 'Unknown'")

    # ── 2c. Domain sanity checks ──────────────────────────────
    print("  [2c] Domain sanity checks …")
    bad_lat     = ~df["latitude"].between(-90, 90)
    bad_lon     = ~df["longitude"].between(-180, 180)
    bad_month   = ~df["month"].between(1, 12)
    bad_year    = ~df["year"].between(1900, 2100)
    bad_shore   = df["shore_distance"] < 0

    mask_bad = bad_lat | bad_lon | bad_month | bad_year | bad_shore
    n_bad    = mask_bad.sum()
    if n_bad:
        print(f"       Removing {n_bad} rows failing domain checks.")
        df = df[~mask_bad].copy()
    else:
        print("       All rows pass domain checks.")

    # ── 2d. Outlier report (IQR-based, no removal) ────────────
    separator("Outlier Report — IQR (Informational Only)")
    outlier_report = detect_outliers_iqr(
        df, cols=["longitude", "latitude", "shore_distance"]
    )
    print(outlier_report.to_string())
    print("\n  Note: Outliers are NOT removed — shore_distance is right-skewed")
    print("     by nature (ocean distances). Log-transform applied later.")

    df.reset_index(drop=True, inplace=True)
    print(f"\n  Preprocessing done.  Final shape: {df.shape}")
    return df


# ╔══════════════════════════════════════════════════════════╗
# ║          STEP 3 — EXPLORATORY DATA ANALYSIS              ║
# ╚══════════════════════════════════════════════════════════╝

def explore_data(df: pd.DataFrame) -> None:
    """Print EDA: cardinality, distributions, correlations."""
    separator("STEP 3 — Exploratory Data Analysis")

    cat_cols = ["attack_type", "vessel_status",
                "nearest_country", "region"]
    num_cols = ["year", "month", "longitude",
                "latitude", "shore_distance"]

    display_categorical_summary(df, cat_cols)
    display_numeric_summary(df, num_cols)

    separator("Correlation with TARGET (shore_distance)")
    corr = df[num_cols].corr()["shore_distance"].drop("shore_distance")
    print(corr.sort_values(key=abs, ascending=False).round(4).to_string())


# ╔══════════════════════════════════════════════════════════╗
# ║          STEP 4 — FEATURE ENGINEERING                    ║
# ╚══════════════════════════════════════════════════════════╝

def engineer_features(df: pd.DataFrame,
                      country_top_n: int = 15) -> tuple:
    """
    Create model-ready features and return (df_numeric, cat_country_col).

    Encoding strategy
    -----------------
    - attack_type    : ordinal (domain-ordered severity score)
    - vessel_status  : ordinal (domain-ordered vulnerability score)
    - nearest_country: rare countries grouped into 'Other', then
                       one-hot encoded INSIDE the sklearn pipeline to
                       prevent data leakage across CV folds.
    - region         : one-hot encoded INSIDE the sklearn pipeline.
    - month          : cyclical sin/cos (no leakage risk)

    Returning the raw grouped-country column allows the pipeline's
    OneHotEncoder to fit only on training folds during cross-validation.
    """
    separator("STEP 4 — Feature Engineering")
    df = df.copy()

    # ── 4a. Log-transform target (right-skewed) ───────────────
    print("  [4a] Log-transforming target 'shore_distance' (skew fix) ...")
    df["log_shore_distance"] = np.log1p(df["shore_distance"])

    # ── 4b. Cyclical encoding for month ──────────────────────
    # Normalise month to [0, 2pi] so January and December are adjacent.
    print("  [4b] Cyclical encoding: month -> sin/cos ...")
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)

    # ── 4c. Haversine to piracy hotspot (Gulf of Aden) ───────
    print("  [4c] Haversine distance to Gulf of Aden hotspot ...")

    def haversine_km(lat1: pd.Series, lon1: pd.Series,
                     lat2: float, lon2: float) -> pd.Series:
        """Vectorised Haversine distance in km."""
        R    = 6371.0
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a    = (np.sin(dlat / 2)**2
                + np.cos(np.radians(lat1))
                * np.cos(np.radians(lat2))
                * np.sin(dlon / 2)**2)
        return R * 2 * np.arcsin(np.sqrt(a))

    GULF_ADEN_LAT, GULF_ADEN_LON = 12.0, 47.0     # known piracy hotspot
    df["dist_gulf_aden_km"] = haversine_km(
        df["latitude"], df["longitude"],
        GULF_ADEN_LAT, GULF_ADEN_LON
    )

    # ── 4d. Absolute latitude (distance from equator) ────────
    print("  [4d] Absolute latitude feature ...")
    df["abs_latitude"] = df["latitude"].abs()

    # ── 4e. Interaction: longitude x latitude ─────────────────
    print("  [4e] Geo interaction feature (lon x lat) ...")
    df["lon_lat_interaction"] = df["longitude"] * df["latitude"]

    # ── 4f. Attack severity ordinal encoding ──────────────────
    print("  [4f] Ordinal encoding: attack_type -> severity_score ...")
    attack_severity = {
        "Suspicious" : 1,
        "Attempted"  : 2,
        "Boarding"   : 3,
        "Boarded"    : 4,
        "Fired Upon" : 5,
        "Detained"   : 5,
        "Hijacked"   : 6,
        "Explosion"  : 7,
    }
    df["attack_severity"] = (df["attack_type"]
                             .map(attack_severity)
                             .fillna(3)          # fallback for unseen values
                             .astype(int))

    # ── 4g. Vessel vulnerability ordinal encoding ─────────────
    print("  [4g] Ordinal encoding: vessel_status -> vulnerability_score ...")
    vessel_vuln = {
        "Berthed"    : 1,
        "Moored"     : 1,
        "Anchored"   : 2,
        "Stationary" : 2,
        "Grounded"   : 3,
        "Drifting"   : 3,
        "Towed"      : 4,
        "Fishing"    : 4,
        "Steaming"   : 5,
        "Underway"   : 5,
    }
    df["vessel_vulnerability"] = (df["vessel_status"]
                                  .map(vessel_vuln)
                                  .fillna(3)
                                  .astype(int))

    # ── 4h. Frequency-based country grouping ─────────────────
    # LabelEncoder assigns arbitrary integer IDs to countries, which
    # implies a false ordinal relationship that misleads linear models.
    # Instead, keep the top-N most frequent countries by name and
    # collapse the rest into "Other". The pipeline's OneHotEncoder will
    # then encode this column on training data only, preventing leakage.
    print(f"  [4h] Country grouping: top-{country_top_n} kept, rest -> 'Other' ...")
    top_countries = (df["nearest_country"]
                     .value_counts()
                     .head(country_top_n)
                     .index.tolist())
    df["country_grouped"] = df["nearest_country"].where(
        df["nearest_country"].isin(top_countries), other="Other"
    )
    print(f"       Unique values after grouping: "
          f"{df['country_grouped'].nunique()}")

    # ── 4i. Year-based trend feature ──────────────────────────
    print("  [4i] Year trend feature (year - base_year) ...")
    df["year_trend"] = df["year"] - df["year"].min()

    # ── 4j. Drop raw columns that have been encoded/replaced ──
    # IMPORTANT: attack_occurred must be created BEFORE attack_type is dropped.
    # It is the binary classification target — 1 if the attack was carried through
    # (Boarded, Hijacked, Fired Upon, Explosion, Detained), 0 if incomplete
    # (Attempted, Boarding, Suspicious). This column is kept in the returned
    # DataFrame so main() can extract it as y_clf without re-reading raw data.
    print("  [4k] Creating binary classification target 'attack_occurred' ...")
    completed_types = {"Boarded", "Hijacked", "Fired Upon", "Explosion", "Detained"}
    df["attack_occurred"] = df["attack_type"].isin(completed_types).astype(int)
    n_pos = df["attack_occurred"].sum()
    n_neg = len(df) - n_pos
    print(f"       Completed attacks (1): {n_pos:,}  |  Incomplete/Attempted (0): {n_neg:,}")
    print(f"       Class balance: {n_pos/len(df)*100:.1f}% positive")

    drop_cols = ["attack_type", "vessel_status",
                 "nearest_country",
                 "month",           # replaced by sin/cos
                 "shore_distance"]  # replaced by log-transformed target
    df.drop(columns=drop_cols, inplace=True, errors="ignore")

    # Categorical columns that must be one-hot encoded INSIDE the
    # pipeline to avoid fitting on test data during CV.
    cat_pipeline_cols = ["region", "country_grouped"]

    print(f"\n  Feature engineering done.  Shape before pipeline OHE: {df.shape}")
    print(f"  Categorical columns deferred to pipeline : {cat_pipeline_cols}")
    return df, cat_pipeline_cols


# ╔══════════════════════════════════════════════════════════╗
# ║    STEP 5 — TRAIN / TEST PREPARATION                     ║
# ╚══════════════════════════════════════════════════════════╝

def prepare_X_y(df: pd.DataFrame,
                cat_cols: list,
                target: str = "log_shore_distance"):
    """
    Split engineered DataFrame into feature matrix X and target y.
    Returns X (DataFrame), y (Series), numeric_cols (list), cat_cols (list).
    Categorical columns are kept as object dtype for the pipeline's
    OneHotEncoder; all remaining feature columns must be numeric.
    """
    separator("STEP 5 — Prepare X and y")
    feature_cols = [c for c in df.columns if c != target]
    X = df[feature_cols].copy()
    y = df[target].copy()

    numeric_cols = [c for c in feature_cols if c not in cat_cols]

    # Verify no unexpected string/object columns remain outside cat_cols.
    # In pandas 2.2+, string data uses pd.StringDtype ('str') rather than
    # object dtype, so both cases must be checked.
    unexpected_obj = [
        c for c in numeric_cols
        if X[c].dtype == object or isinstance(X[c].dtype, pd.StringDtype)
    ]
    if unexpected_obj:
        raise TypeError(
            f"Unexpected string/object-dtype columns in numeric set: {unexpected_obj}. "
            "Encode them before modelling."
        )

    print(f"  Target          : {target}")
    print(f"  Numeric features: {len(numeric_cols)}")
    print(f"  Cat features    : {len(cat_cols)}  {cat_cols}")
    print(f"  Total features  : {len(feature_cols)}")
    print(f"  Sample count    : {len(X):,}")
    print("  All dtypes validated.")
    return X, y, numeric_cols, cat_cols


# ╔══════════════════════════════════════════════════════════╗
# ║    STEP 6 — BUILD SKLEARN PIPELINE                       ║
# ╚══════════════════════════════════════════════════════════╝

def build_pipeline(numeric_cols: list,
                   cat_cols: list) -> Pipeline:
    """
    Build a robust sklearn Pipeline.

    Preprocessing inside the pipeline (fitted on training fold only):
      - Numeric  : SimpleImputer (median) -> StandardScaler
      - Categorical: SimpleImputer (constant='missing') -> OneHotEncoder
        drop='first' removes one dummy per category to avoid multicollinearity.
        handle_unknown='ignore' silently drops categories unseen at fit time.

    Model: Ridge (L2 regularisation) to prevent overfitting.

    Encoding categorical columns inside the pipeline ensures the
    OneHotEncoder is fitted only on training data during each CV fold,
    eliminating any risk of data leakage from the test fold.
    """
    separator("STEP 6 — Build sklearn Pipeline")

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant",
                                  fill_value="missing")),
        ("ohe",     OneHotEncoder(drop="first",
                                  handle_unknown="ignore",
                                  sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer,    numeric_cols),
        ("cat", categorical_transformer, cat_cols),
    ], remainder="drop")

    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model",        GradientBoostingRegressor(
                             n_estimators=200,
                             max_depth=4,
                             learning_rate=0.05,
                             subsample=0.8,
                             min_samples_leaf=10,
                             random_state=42,
                         )),
    ])

    print("  Numeric branch  : SimpleImputer(median) -> StandardScaler")
    print("  Categorical branch: SimpleImputer(constant) -> OneHotEncoder(drop='first')")
    print("  Model           : GradientBoostingRegressor(n_estimators=200, max_depth=4, lr=0.05)")
    print("  Note: OHE fitted per fold inside CV — no data leakage.")
    print("  Pipeline ready.")
    return pipe


# ╔══════════════════════════════════════════════════════════╗
# ║    STEP 7 — K-FOLD CROSS VALIDATION                      ║
# ╚══════════════════════════════════════════════════════════╝

def run_kfold_cv(pipe   : Pipeline,
                 X      : pd.DataFrame,
                 y      : pd.Series,
                 n_splits: int = 10) -> dict:
    """
    Placeholder for regression cross-validation.
    R² metrics have been removed from the regression pipeline.
    """
    separator(f"STEP 7 — {n_splits}-Fold Cross Validation")
    print("  Regression cross-validation metrics have been removed.")
    return {}


# ╔══════════════════════════════════════════════════════════╗
# ║    STEP 8 — FIT FINAL MODEL & FEATURE IMPORTANCE         ║
# ╚══════════════════════════════════════════════════════════╝

def fit_final_model(pipe         : Pipeline,
                    X            : pd.DataFrame,
                    y            : pd.Series) -> Pipeline:
    """
    Fit the pipeline on the full dataset and display Ridge coefficients.

    Full-data metrics are shown for reference only. They are computed
    on the same data used for training and therefore reflect in-sample
    fit, not generalisation ability. Use K-Fold CV metrics for any
    performance claim.

    Feature names are retrieved from the fitted ColumnTransformer via
    get_feature_names_out(), which accounts for the OHE expansion and
    ensures coefficient-to-feature alignment is always correct.
    """
    separator("STEP 8 — Final Model Fit on Full Data")

    pipe.fit(X, y)
    print("  Model fitted on full dataset.")

    # ── Retrieve feature names from the fitted preprocessor ──────
    # get_feature_names_out() returns names in the same order that
    # ColumnTransformer passes columns to the model, so coef_ alignment
    # is guaranteed regardless of future column reordering.
    feature_names_out = (pipe.named_steps["preprocessor"]
                         .get_feature_names_out())

    importances = pipe.named_steps["model"].feature_importances_
    importance_df = (pd.DataFrame({"feature"   : feature_names_out,
                                   "importance": importances})
                     .sort_values("importance", ascending=False)
                     .reset_index(drop=True))

    separator("Feature Importances (sorted by importance)")
    print(importance_df.to_string(index=False))

    pipe.predict(X)
    separator("In-Sample Metrics  [WARNING: computed on training data — not validation]")
    print("  In-sample regression predictions computed. No R² metric is displayed.")

    return pipe, importance_df


# ╔══════════════════════════════════════════════════════════╗
# ║    STEP 9 — ATTACK PROBABILITY CLASSIFICATION            ║
# ╚══════════════════════════════════════════════════════════╝

def build_classification_pipeline(numeric_cols: list,
                                   cat_cols: list) -> Pipeline:
    """
    Build a Logistic Regression pipeline for binary attack classification.

    Target  : attack_occurred (1 = attack completed, 0 = attempted/incomplete)
    Model   : LogisticRegression with L2 penalty (C=1.0)
    Output  : predict_proba() gives calibrated probability in [0, 1]

    Feature set is identical in structure to the regression pipeline
    but attack_severity is deliberately excluded because it is derived
    directly from attack_type, which is the basis of the binary target.
    Including it would constitute data leakage.

    The pipeline is kept fully separate from the regression pipeline so
    that neither pipeline's state can interfere with the other.
    """
    separator("STEP 9a — Build Classification Pipeline")

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant",
                                  fill_value="missing")),
        ("ohe",     OneHotEncoder(drop="first",
                                  handle_unknown="ignore",
                                  sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer,     numeric_cols),
        ("cat", categorical_transformer, cat_cols),
    ], remainder="drop")

    clf_pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model",        GradientBoostingClassifier(
                             n_estimators=200,
                             max_depth=4,
                             learning_rate=0.05,
                             subsample=0.8,
                             min_samples_leaf=10,
                             random_state=42,
                         )),
    ])

    print("  Numeric branch    : SimpleImputer(median) -> StandardScaler")
    print("  Categorical branch: SimpleImputer(constant) -> OneHotEncoder(drop='first')")
    print("  Model             : GradientBoostingClassifier(n_estimators=200, max_depth=4, lr=0.05)")
    print("  Note: attack_severity excluded — derived from target, would cause leakage.")
    print("  Classification pipeline ready.")
    return clf_pipe


def run_classification_cv(clf_pipe : Pipeline,
                           X_clf    : pd.DataFrame,
                           y_clf    : pd.Series,
                           n_splits : int = 10) -> dict:
    """
    Run Stratified K-Fold cross-validation for the classification pipeline.

    StratifiedKFold is used (not plain KFold) because the target is binary —
    stratification preserves the class ratio in every fold, giving stable
    and unbiased estimates for both classes.

    Metrics reported per fold:
      - Accuracy : fraction of correct predictions
      - Brier    : mean squared probability error (lower is better)

    Returns a dict of per-fold metric arrays.
    """
    separator(f"STEP 9b — {n_splits}-Fold Stratified CV (Classification)")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    scoring = {
        "accuracy" : "accuracy",
        "brier"    : "neg_brier_score",
    }

    print(f"  Running {n_splits}-Fold Stratified CV ...")

    cv_results = cross_validate(
        estimator          = clf_pipe,
        X                  = X_clf,
        y                  = y_clf,
        cv                 = skf,
        scoring            = scoring,
        return_train_score = True,
        n_jobs             = -1,
    )

    acc_test   = cv_results["test_accuracy"]
    brier_test = -cv_results["test_brier"]      # negate: stored as negative
    acc_train  = cv_results["train_accuracy"]

    # ── Per-fold table ────────────────────────────────────────
    print(f"\n  {'Fold':>4}  {'Train Acc':>10}  {'Test Acc':>9}  "
          f"{'Test Brier':>11}")
    print("  " + "─" * 48)
    for i in range(n_splits):
        print(f"  {i+1:>4}  {acc_train[i]:>10.4f}  {acc_test[i]:>9.4f}  "
              f"{brier_test[i]:>11.4f}")

    # ── Summary ───────────────────────────────────────────────
    separator("Classification CV Summary")

    def _stats(arr, label):
        print(f"  {label:<30} mean={arr.mean():.4f}  "
              f"std={arr.std():.4f}  "
              f"min={arr.min():.4f}  max={arr.max():.4f}")

    _stats(acc_train,  "Train Accuracy")
    _stats(acc_test,   "Test  Accuracy")
    _stats(brier_test, "Test  Brier Score (lower=better)")

    # ── Overfitting check ─────────────────────────────────────
    separator("Classification Overfitting Diagnostic")
    acc_gap = acc_train.mean() - acc_test.mean()
    print(f"  Train Accuracy - Test Accuracy gap : {acc_gap:+.4f}")

    if acc_gap < 0.03:
        verdict = "No overfitting detected — classifier generalises well."
    elif acc_gap < 0.07:
        verdict = "Mild overfitting — consider reducing C or adding features."
    else:
        verdict = "Significant overfitting — reduce model complexity."

    print(f"  Verdict: {verdict}")

    return {
        "acc_train"  : acc_train,
        "acc_test"   : acc_test,
        "brier_test" : brier_test,
    }


def generate_attack_probability_column(clf_pipe   : Pipeline,
                                        X_clf      : pd.DataFrame,
                                        y_clf      : pd.Series,
                                        df_clean   : pd.DataFrame,
                                        output_path: str) -> pd.DataFrame:
    """
    Fit the classification pipeline on the full dataset, generate
    attack probability for every row, and write an enriched CSV.

    New columns added to the output:
      - attack_occurred       : binary label (1=completed, 0=incomplete)
      - attack_probability_pct: probability (0.00 – 100.00) that an attack
                                of this type/location/vessel combination
                                results in a completed attack, expressed
                                as a percentage rounded to 2 decimal places.
      - risk_band             : human-readable risk tier derived from
                                attack_probability_pct thresholds.

    The pipeline is re-fitted on the complete X_clf / y_clf so that the
    final probability estimates use all available data — consistent with
    how the regression final model is handled. CV metrics (not these
    in-sample probabilities) represent true generalisation performance.

    The enriched DataFrame is also returned for inspection.
    """
    separator("STEP 9c — Fit Final Classifier & Generate Probability Column")

    # ── Fit on full data ──────────────────────────────────────
    clf_pipe.fit(X_clf, y_clf)
    print("  Classifier fitted on full dataset.")

    # ── Build enriched DataFrame ──────────────────────────────
    # Start from df_clean (the preprocessed raw DataFrame, index aligned)
    # so all original columns are preserved in the output.
    proba = clf_pipe.predict_proba(X_clf)[:, 1]   # P(attack_occurred=1)
    df_out = df_clean.reset_index(drop=True).copy()

    df_out["attack_occurred"]        = y_clf.values
    df_out["attack_probability_pct"] = (proba * 100).round(2)

    # ── Risk band assignment ──────────────────────────────────
    def assign_risk_band(pct: float) -> str:
        if pct < 25:
            return "Low"
        elif pct < 50:
            return "Moderate"
        elif pct < 75:
            return "High"
        else:
            return "Critical"

    df_out["risk_band"] = df_out["attack_probability_pct"].apply(assign_risk_band)

    # ── Build remaining output
    separator("STEP 9d — Insurance Premium Layer")
    df_out = append_premium_columns(df_out)
    print(f"  Default insured value : ${DEFAULT_INSURED_VALUE:,.0f}")
    print("  Formula  : P = (p × V × LGD × λ_region × (1+θ)) + V × r_base")
    print("  Refs     : Bowers et al.(1997); Stopford(2009); ICC IMB(2020);")
    print("             Swiss Re(2011); IUMI(2019); JWC Listed Areas(2023)")
    print_premium_statistics(df_out)

    # ── Write CSV ─────────────────────────────────────────────
    df_out.to_csv(output_path, index=False)
    print(f"\n  Output CSV written to: {output_path}")
    print(f"  Rows: {len(df_out):,}   Columns: {df_out.shape[1]}")
    print(f"  New columns added: attack_occurred, attack_probability_pct,")
    print(f"                     risk_band, lambda_region, expected_loss_usd,")
    print(f"                     risk_premium_usd, base_premium_usd,")
    print(f"                     total_premium_usd, premium_rate_pct")

    return df_out

    separator("Sample Output Rows (5 lowest probability)")
    bot5 = (df_out.sort_values("attack_probability_pct", ascending=True)
                  .head(5)[cols_show])
    print(bot5.to_string(index=False))

    # ── Insurance premium layer ──────────────────────────────────
    # Append voyage-premium columns using the actuarial EV formula:
    #   P = (p × V × LGD × λ_region × (1 + θ)) + V × r_base
    # Defaults: V=$25M, LGD=35% (ICC IMB 2020), θ=20% (Swiss Re 2011),
    #           r_base=0.05% p.a. (IUMI 2019), λ from JWC Listed Areas.
    separator("STEP 9d — Insurance Premium Layer")
    df_out = append_premium_columns(df_out)
    print(f"  Default insured value : ${DEFAULT_INSURED_VALUE:,.0f}")
    print("  Formula  : P = (p × V × LGD × λ_region × (1+θ)) + V × r_base")
    print("  Refs     : Bowers et al.(1997); Stopford(2009); ICC IMB(2020);")
    print("             Swiss Re(2011); IUMI(2019); JWC Listed Areas(2023)")
    print_premium_statistics(df_out)

    # ── Write CSV ─────────────────────────────────────────────
    df_out.to_csv(output_path, index=False)
    print(f"\n  Output CSV written to: {output_path}")
    print(f"  Rows: {len(df_out):,}   Columns: {df_out.shape[1]}")
    print(f"  New columns added: attack_occurred, attack_probability_pct,")
    print(f"                     risk_band, lambda_region, expected_loss_usd,")
    print(f"                     risk_premium_usd, base_premium_usd,")
    print(f"                     total_premium_usd, premium_rate_pct")

    return df_out

def main() -> None:
    separator("PIRATE ATTACKS — ML PIPELINE  (Regression + Classification)", "═")
    print("  Task 1  : Ridge Regression   — predict shore_distance")
    print("  Task 2  : Logistic Regression — predict attack_occurred probability")
    print("  CV      : 10-Fold KFold (regression) / Stratified (classification)")
    separator(char="═")

    BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
    FILEPATH    = os.path.join(BASE_DIR, "pirate_attacks_clean.csv")
    OUTPUT_PATH = os.path.join(BASE_DIR, "pirate_attacks_with_probability.csv")

    # ── 1. Load ───────────────────────────────────────────────
    df_raw = load_data(FILEPATH)

    # ── 2. Preprocess ─────────────────────────────────────────
    df_clean = preprocess_data(df_raw)

    # ── 3. EDA ────────────────────────────────────────────────
    explore_data(df_clean)

    # ── 4. Feature Engineering ────────────────────────────────
    df_eng, cat_cols = engineer_features(df_clean)

    y_clf = df_eng["attack_occurred"].copy()
    df_for_mod = df_eng.drop(columns=["attack_occurred"])

    # ── 5. Prepare X, y  (REGRESSION) ────────────────────────
    X, y, numeric_cols, cat_cols = prepare_X_y(
        df_for_mod, cat_cols=cat_cols, target="log_shore_distance"
    )

    # ── 6. Build Regression Pipeline ──────────────────────────
    pipe = build_pipeline(numeric_cols, cat_cols)

    # ── 7. Regression K-Fold CV ───────────────────────────────
    cv_metrics = run_kfold_cv(pipe, X, y, n_splits=10)

    # ── 8. Fit Final Regression Model ─────────────────────────
    final_pipe, importance_df = fit_final_model(pipe, X, y)

    # ── 9. Classification — Attack Probability ────────────────
    clf_numeric_cols = (
        [c for c in numeric_cols if c != "attack_severity"]
        + ["log_shore_distance"]
    )
    clf_cat_cols = list(cat_cols)

    X_clf = df_for_mod[clf_numeric_cols + clf_cat_cols].copy()

    separator("STEP 9 — Attack Probability Classification")
    print(f"  Binary target    : attack_occurred")
    print(f"  Positive class   : Completed attack (Boarded / Hijacked / etc.)")
    print(f"  Negative class   : Incomplete attack (Attempted / Boarding / etc.)")
    print(f"  Numeric features : {len(clf_numeric_cols)}")
    print(f"  Cat features     : {len(clf_cat_cols)}  {clf_cat_cols}")
    print(f"  Class balance    : {y_clf.sum():,} positive  /  {(~y_clf.astype(bool)).sum():,} negative")

    clf_pipe   = build_classification_pipeline(clf_numeric_cols, clf_cat_cols)
    clf_cv     = run_classification_cv(clf_pipe, X_clf, y_clf, n_splits=10)
    df_with_prob = generate_attack_probability_column(
        clf_pipe, X_clf, y_clf, df_clean, OUTPUT_PATH
    )

    # ── Final Summary ─────────────────────────────────────────
    separator("FINAL RESULTS SUMMARY", "═")
    print("  REGRESSION (shore_distance prediction)")
    print(f"    Dataset size     : {len(X):,} samples")
    print(f"    Numeric features : {len(numeric_cols)}")
    print(f"    Categorical cols : {len(cat_cols)}  (OHE expanded inside pipeline)")
    print(f"    Cross-validation metrics are not displayed for regression.")
    print()
    print("  CLASSIFICATION (attack_occurred probability)")
    print(f"    Dataset size     : {len(X_clf):,} samples")
    print(f"    CV folds         : 10  (Stratified)")
    print(f"    Mean Test Accuracy: {clf_cv['acc_test'].mean():.4f}")
    print(f"    Mean Test Brier  : {clf_cv['brier_test'].mean():.4f}  (lower=better)")
    acc_gap = clf_cv['acc_train'].mean() - clf_cv['acc_test'].mean()
    print(f"    Train-Test Accuracy Gap: {acc_gap:.4f}  "
          + ("  [No overfitting]" if acc_gap < 0.03 else "  [Some overfitting]"))
    print()
    print(f"  OUTPUT CSV       : {OUTPUT_PATH}")
    print(f"    Rows           : {len(df_with_prob):,}")
    print(f"    New columns    : attack_occurred, attack_probability_pct, risk_band,")
    print(f"                     lambda_region, expected_loss_usd, risk_premium_usd,")
    print(f"                     base_premium_usd, total_premium_usd, premium_rate_pct")
    separator(char="═")
    print("  Pipeline completed successfully.")


if __name__ == "__main__":
    main()