from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

from src.train import FEATURES, CATEGORICAL, NUMERIC, load_model_bundle, train_model

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "house_price_model.joblib"
DATA_PATH = ROOT / "data" / "sample_houses_demo.csv"

st.set_page_config(page_title="House Price Estimator", page_icon="🏠", layout="wide")
st.title("🏠 ML House Price Estimator")
st.caption("A practical baseline for house-price estimation. The included data is synthetic demo data—not verified market transactions.")

@st.cache_resource
def get_bundle():
    if MODEL_PATH.exists():
        try:
            return load_model_bundle(MODEL_PATH)
        except Exception:
            pass
    return train_model(DATA_PATH, MODEL_PATH)

bundle = get_bundle()
model = bundle["model"]
metrics = bundle.get("metrics", {})

with st.sidebar:
    st.header("Model information")
    st.write(f"**Model:** {bundle.get('model_name', 'Random Forest Regressor')}")
    if metrics:
        st.metric("Test MAE", f"₹{metrics.get('mae', 0):,.0f}")
        st.metric("Test R²", f"{metrics.get('r2', 0):.3f}")
    st.warning("Demo model metrics measure fit to synthetic data. They do not represent real-world accuracy.")
    st.divider()
    st.subheader("Use your own data")
    st.write("Upload a CSV with the same feature columns and a `price_inr` target, then retrain from the project terminal using the README instructions.")

left, right = st.columns([1, 1])
with left:
    st.subheader("Property details")
    city = st.selectbox("City", ["Mumbai","Pune","Bengaluru","Hyderabad","Delhi","Chennai","Kolhapur","Nashik","Nagpur","Ahmedabad"])
    area = st.number_input("Built-up area (sq ft)", min_value=100, max_value=20000, value=1000, step=50)
    bedrooms = st.slider("Bedrooms", 1, 10, 2)
    bathrooms = st.slider("Bathrooms", 1, 10, 2)
    age = st.slider("Property age (years)", 0, 100, 5)
    floor = st.slider("Floor number", 0, 60, 3)
    parking = st.selectbox("Parking spaces", [0,1,2,3,4,5], index=1)
    furnishing = st.selectbox("Furnishing", ["Unfurnished","Semi-furnished","Furnished"])
    property_type = st.selectbox("Property type", ["Apartment","Independent House","Villa"])
    locality_score = st.slider("Locality score (1 = low, 10 = premium)", 1.0, 10.0, 6.0, 0.5)
    distance = st.slider("Distance to city centre (km)", 0.0, 80.0, 8.0, 0.5)

with right:
    st.subheader("Estimated price")
    row = pd.DataFrame([{
        "city": city, "area_sqft": area, "bedrooms": bedrooms, "bathrooms": bathrooms,
        "property_age_years": age, "floor_number": floor, "parking_spaces": parking,
        "furnishing": furnishing, "property_type": property_type, "locality_score": locality_score,
        "distance_to_city_center_km": distance
    }], columns=FEATURES)
    prediction = float(model.predict(row)[0])
    st.metric("Predicted property value", f"₹{prediction:,.0f}")
    st.caption(f"Approx. ₹{prediction / max(area,1):,.0f} per sq ft")
    st.info("Treat this as an indicative estimate only. Actual prices depend on exact locality, legal title, building condition, amenities, negotiation, and current market data.")
    st.subheader("What affects the estimate?")
    st.write("The model learns patterns from the training data. For a credible valuation, train it on recent, verified transactions from the target city/localities.")
    if st.button("Download prediction as CSV"):
        st.download_button("Save result", row.assign(predicted_price_inr=round(prediction)).to_csv(index=False).encode("utf-8"), "house_price_prediction.csv", "text/csv")

st.divider()
st.subheader("Training data preview")
preview = pd.read_csv(DATA_PATH).head(10)
st.dataframe(preview, use_container_width=True)
st.download_button("Download demo dataset", DATA_PATH.read_bytes(), "sample_houses_demo.csv", "text/csv")
