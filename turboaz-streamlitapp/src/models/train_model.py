"""
Train the RandomForest price prediction pipeline used by the Streamlit app.

This mirrors the workflow from turboaz_prediction.ipynb:
1. Drop leakage / non-predictive columns (price, currency, day, hour, views)
2. Train/test split
3. Remove outliers from the training set only (IsolationForest)
4. Build a ColumnTransformer (scaling, log-transform, one-hot, target encoding)
5. Fit a RandomForestRegressor on top of the ColumnTransformer
6. Save the fitted pipeline + evaluation metrics + reference values for the UI

Run this once (or whenever the dataset changes) to regenerate:
    models/price_model.joblib
    models/model_meta.json
"""

import json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import (
    FunctionTransformer,
    StandardScaler,
    OneHotEncoder,
    TargetEncoder,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

DATA_PATH = "data/turboaz_cars_cleaned.csv"
MODEL_PATH = "models/price_model.joblib"
META_PATH = "models/model_meta.json"

RANDOM_STATE = 42

DROP_COLS = ["price", "currency", "day", "hour", "views"]

NUM_COLS = ["engine_displacement_num", "production_year", "seat_count", "at_gucu"]
LOG_COLS = ["kilometrage_num"]
ONEHOT_COLS = ["market_type", "Qəzalı", "Sahiblər", "transmission", "drivetrain", "yanacaq_novu"]
TARGET_COLS = ["city", "shop_name", "Marka", "Model", "Rəng", "ban_type"]
BINARY_COLS = ["barter", "loan", "salon", "vip", "featured", "Vəziyyəti", "Yeni",
               "Kondisioner", "Arxa_kamera", "Rənglənib"]

OUTLIER_COLS = ["kilometrage_num", "engine_displacement_num", "at_gucu", "production_year"]


def build_column_transformer():
    num_pipeline = Pipeline([("scl", StandardScaler())])
    log_pipeline = Pipeline([
        ("log", FunctionTransformer(func=np.log1p, inverse_func=np.expm1, feature_names_out="one-to-one")),
        ("scl", StandardScaler()),
    ])
    onehot_pipeline = Pipeline([("enc", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))])
    target_pipeline = Pipeline([("tenc", TargetEncoder())])

    return ColumnTransformer([
        ("NUM", num_pipeline, NUM_COLS),
        ("LOG", log_pipeline, LOG_COLS),
        ("ENC", onehot_pipeline, ONEHOT_COLS),
        ("TENC", target_pipeline, TARGET_COLS),
    ], remainder="passthrough")


def main():
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=DROP_COLS)
    df_full = df.copy()  # keep full data for UI dropdown options (before outlier removal)

    # Outlier removal on the WHOLE dataset, before splitting (updated approach)
    iso = IsolationForest(contamination=0.01, random_state=RANDOM_STATE)
    iso.fit(df.loc[:, OUTLIER_COLS])
    mask = iso.predict(df.loc[:, OUTLIER_COLS]) == 1
    df = df.loc[mask].reset_index(drop=True)

    X = df.drop(columns=["price_azn"])
    y = df["price_azn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    col_trans = build_column_transformer()

    pipeline = Pipeline([
        ("cleaning", col_trans),
        ("forestreg", RandomForestRegressor(
            n_estimators=100,
            max_depth=14,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )),
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MAE : {mae:,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"R2  : {r2:.4f}")

    joblib.dump(pipeline, MODEL_PATH)

    # Reference metadata for building the prediction form UI
    full_X = df_full.drop(columns=["price_azn"])
    meta = {
        "metrics": {"mae": mae, "rmse": rmse, "r2": r2},
        "feature_columns": list(full_X.columns),
        "categorical_options": {
            col: sorted(full_X[col].dropna().unique().tolist())
            for col in ONEHOT_COLS + TARGET_COLS
        },
        "marka_model_map": (
            full_X.groupby("Marka")["Model"]
            .apply(lambda s: sorted(s.unique().tolist()))
            .to_dict()
        ),
        "marka_model_bantype_map": {
            marka: group.droplevel(0).to_dict()
            for marka, group in full_X.groupby(["Marka", "Model"])["ban_type"]
            .apply(lambda s: sorted(s.unique().tolist()))
            .groupby(level=0)
        },
        "numeric_ranges": {
            col: {
                "min": float(full_X[col].min()),
                "max": float(full_X[col].max()),
                "median": float(full_X[col].median()),
            }
            for col in NUM_COLS + LOG_COLS
        }
        | {
            "price_azn": {
                "min": float(df_full["price_azn"].min()),
                "max": float(df_full["price_azn"].max()),
                "median": float(df_full["price_azn"].median()),
            }
        },
    }

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved metadata to {META_PATH}")


if __name__ == "__main__":
    main()
