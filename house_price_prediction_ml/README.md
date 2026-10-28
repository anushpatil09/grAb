# ML-Based House Price Prediction

A runnable Python + Streamlit project that estimates residential property prices from property features. It includes a preprocessing pipeline, a tree-based regression model, held-out test metrics, CSV export, and a small demo dataset.

## Important: what this is (and is not)

The bundled `data/sample_houses_demo.csv` is **synthetic demonstration data**, generated for testing the full workflow. The city price levels and feature relationships are illustrative—not verified listings, registered transactions, or an authoritative valuation source. The app runs immediately, but its displayed predictions are not suitable for financial, lending, buying, or selling decisions until retrained and evaluated on recent local market data.

## Features

- Streamlit interface with property details and predicted INR value
- ExtraTrees regression model with imputation + one-hot encoding in a single sklearn pipeline
- Fixed random train/test split and MAE, RMSE, and R² reporting
- Unknown categories handled safely by the encoder
- Saved model bundle and metrics JSON after training
- CSV download for dataset and individual prediction

## Requirements

- Python 3.10–3.12 recommended
- Internet is not required after Python dependencies are installed.

## Run on Windows (PowerShell)

```powershell
cd house_price_prediction_ml
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m src.train
streamlit run app.py
```

If PowerShell blocks activation, use Command Prompt:
```bat
.venv\Scripts\activate.bat
```

The browser will open the Streamlit app, normally at `http://localhost:8501`.

## Run on macOS/Linux

```bash
cd house_price_prediction_ml
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m src.train
streamlit run app.py
```

## Use a real dataset (recommended before presenting it as a real predictor)

Collect a licensed dataset of recent, verified sale transactions from the exact region you want to predict. Prefer registered sale prices or verified closed transactions over asking prices. Avoid mixing cities or time periods without controlling for location and date.

Create a CSV with these columns (names and units matter):

```text
city,area_sqft,bedrooms,bathrooms,property_age_years,floor_number,parking_spaces,furnishing,property_type,locality_score,distance_to_city_center_km,price_inr
```

- `city`: consistent city name
- `area_sqft`: built-up area in square feet (do not mix carpet and built-up area)
- `bedrooms`, `bathrooms`, `property_age_years`, `floor_number`, `parking_spaces`: numeric values
- `furnishing`: `Unfurnished`, `Semi-furnished`, or `Furnished` (or your own consistent labels)
- `property_type`: e.g. `Apartment`, `Independent House`, `Villa`
- `locality_score`: numeric 1–10 rating; define and apply a consistent rule. Better yet, replace it with real locality identifiers/amenity data.
- `distance_to_city_center_km`: distance in kilometres measured consistently
- `price_inr`: sale price in INR (target)

Put your file at `data/real_houses.csv`, then run:

```bash
python -m src.train --data data/real_houses.csv --model models/house_price_model.joblib
streamlit run app.py
```

The app loads the saved model bundle if one exists. If you train a model on your real dataset, restart the app so it picks up the new model. To avoid accidentally using a stale model, delete `models/house_price_model.joblib` before the first demo run or retrain as shown above. The app currently previews and offers the included demo CSV; change `DATA_PATH` in `app.py` if you want the preview to show your real dataset.

## How to evaluate properly

Do not judge quality by R² alone. Check:
- **MAE**: typical absolute error in INR; lower is better.
- **RMSE**: penalizes large misses more heavily.
- **R²**: fraction of target variance explained on the held-out split; can be negative.
- Compare against a simple baseline such as median price per city or a linear model.
- Use a time-based split (train on older sales, test on newer sales) for market deployment. If predicting across areas, also consider a location-grouped split to test generalization.
- Report sample size, date range, city coverage, and missingness with metrics.
- Do not claim that synthetic-data metrics establish real-world accuracy.

## Project structure

```text
house_price_prediction_ml/
├── app.py
├── src/
│   ├── __init__.py
│   └── train.py
├── data/
│   └── sample_houses_demo.csv
├── models/                 # trained model saved here (generated locally)
├── reports/                # metrics.json generated locally
├── requirements.txt
├── .gitignore
└── README.md
```

## Limitations and next steps

1. The included data is synthetic, so use a real dataset before drawing conclusions.
2. The model does not know exact street/locality, sale date, land title, floor area definitions, building condition, amenities, or legal encumbrances.
3. Predictions outside the training range may be unreliable.
4. For stronger production performance, add a verified locality ID, sale date, property condition, age of transaction, amenities, and a temporal validation plan. Track drift and retrain periodically.
5. Add authentication, rate limits, logging, monitoring, and privacy controls before exposing the app publicly.

## Troubleshooting

- `ModuleNotFoundError`: activate your virtual environment and run `pip install -r requirements.txt`.
- Streamlit command not found: run `python -m streamlit run app.py`.
- Model/data mismatch: retrain with `python -m src.train --data data/real_houses.csv`.
- If a model file was trained with a different scikit-learn version, reinstall matching requirements or retrain it.
