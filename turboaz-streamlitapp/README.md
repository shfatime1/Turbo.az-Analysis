---
title: Turbo.az Data Insights
emoji: 🚀
colorFrom: red
colorTo: red
sdk: docker
app_port: 8501
tags:
- streamlit
pinned: false
short_description: Interactive insights from car listings.
---

# 🚗 Turbo.az Car Market Explorer

A multi-page Streamlit app for exploring the Azerbaijani used-car market
(Turbo.az listings) with interactive Plotly dashboards and a Random Forest
price prediction tool.

## Project structure

```
turboaz_dashboard/
├── streamlit_app.py          # Home page (entry point)
├── utils.py                  # Shared data loading, filters, charts, prediction helpers
├── pages/
│   ├── 1_📊_Market_Dashboard.py     # Market-wide charts (brand, city, price, fuel, body type...)
│   ├── 2_🔍_Explore_and_Filter.py   # Detailed filtering, data table, CSV export
│   └── 3_🤖_Price_Prediction.py     # Price prediction form + model metrics
├── data/
│   └── turboaz_cars_cleaned.csv     # Cleaned dataset
├── models/
│   ├── train_model.py        # Script that trains and saves the pipeline
│   ├── price_model.joblib    # Trained Random Forest pipeline (generated)
│   └── model_meta.json       # Metrics + dropdown options for the UI (generated)
└── requirements.txt
```

## Running locally

```bash
pip install -r requirements.txt

# (Optional) retrain the model if the dataset changes:
python models/train_model.py

streamlit run streamlit_app.py
```

## Model

- **Algorithm:** Random Forest Regressor
- **Preprocessing:** StandardScaler for numeric features, log1p + scaling for
  mileage, one-hot encoding for low-cardinality categorical features, target
  encoding for high-cardinality categorical features (city, brand, model,
  seller, color, body type).
- **Outlier handling:** Isolation Forest removes ~1% of extreme training rows
  before fitting.
- **Target:** `price_azn` (price converted to Azerbaijani Manat).
