from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import TransformedTargetRegressor

ROOT = Path(__file__).resolve().parents[1]
TARGET = "price_inr"
CATEGORICAL = ["city", "furnishing", "property_type"]
NUMERIC = ["area_sqft", "bedrooms", "bathrooms", "property_age_years", "floor_number",
           "parking_spaces", "locality_score", "distance_to_city_center_km"]
FEATURES = CATEGORICAL + NUMERIC

def build_pipeline():
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    prep = ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC),
        ("categorical", categorical_pipe, CATEGORICAL)
    ])
    # ExtraTrees is a strong tabular baseline and handles non-linear feature relationships.
    return Pipeline([
        ("preprocess", prep),
        ("regressor", ExtraTreesRegressor(
            n_estimators=350, min_samples_leaf=2, max_features=0.9,
            random_state=42, n_jobs=-1
        ))
    ])

def validate_data(df):
    required = set(FEATURES + [TARGET])
    missing = required - set(df.columns)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))
    if df.empty:
        raise ValueError("Training dataset is empty.")
    if (pd.to_numeric(df[TARGET], errors="coerce").dropna() <= 0).any():
        raise ValueError("Target prices must be positive.")
    return df[FEATURES].copy(), pd.to_numeric(df[TARGET], errors="coerce")

def train_model(data_path, model_path=None):
    data_path = Path(data_path)
    df = pd.read_csv(data_path)
    X, y = validate_data(df)
    valid = y.notna()
    X, y = X.loc[valid], y.loc[valid]
    if len(X) < 30:
        raise ValueError("At least 30 valid rows are recommended for training.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = build_pipeline()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": float(mean_squared_error(y_test, preds) ** 0.5),
        "r2": float(r2_score(y_test, preds)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "dataset_rows": int(len(X)),
    }
    bundle = {
        "model": model,
        "model_name": "ExtraTreesRegressor",
        "features": FEATURES,
        "target": TARGET,
        "metrics": metrics,
        "data_source": str(data_path.name),
    }
    if model_path:
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, model_path)
        report_path = ROOT / "reports" / "metrics.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(metrics, indent=2))
    return bundle

def load_model_bundle(path):
    return joblib.load(path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train the house price regression model.")
    parser.add_argument("--data", default=str(ROOT / "data" / "sample_houses_demo.csv"))
    parser.add_argument("--model", default=str(ROOT / "models" / "house_price_model.joblib"))
    args = parser.parse_args()
    result = train_model(args.data, args.model)
    print(json.dumps(result["metrics"], indent=2))
