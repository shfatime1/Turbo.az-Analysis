import streamlit as st

from utils import (
    load_model,
    load_model_meta,
    predict_price,
    format_azn,
    apply_custom_theme,
    chart_feature_importance,
    chart_price_gauge,
)

st.set_page_config(page_title="Price Prediction | Turbo.az", page_icon =":space_invader:",layout="wide")
apply_custom_theme()

st.title("👾 Price Prediction")
st.caption(
    "Fill in the car's details below and get an estimated market price from a "
    "Random Forest model trained on real Turbo.az listings."
)

model = load_model()
meta = load_model_meta()
opts = meta["categorical_options"]
ranges = meta["numeric_ranges"]
marka_model_map = meta["marka_model_map"]
feature_columns = meta["feature_columns"]

marka_model_bantype_map = meta["marka_model_bantype_map"]

st.markdown("### Car identity")
st.caption("Brand, model and body type are selected here (outside the form) so the lists update instantly as you choose.")
c1, c2, c3 = st.columns(3)
with c1:
    city = st.selectbox("City", sorted(opts["city"]), index=sorted(opts["city"]).index("bakı") if "bakı" in opts["city"] else 0)
with c2:
    marka = st.selectbox("Brand (Marka)", sorted(marka_model_map.keys()), key="marka_select")
with c3:
    model_options = sorted(marka_model_map.get(marka, []))
    car_model = st.selectbox("Model", model_options if model_options else ["Unknown"], key=f"model_select_{marka}")

c_bt, _ = st.columns(2)
with c_bt:
    bantype_options = sorted(marka_model_bantype_map.get(marka, {}).get(car_model, []))
    if not bantype_options:
        bantype_options = sorted(opts["ban_type"])
    ban_type = st.selectbox(
        "Ban type", bantype_options, key=f"bantype_select_{marka}_{car_model}"
    )

with st.form("prediction_form"):
    c4, c5 = st.columns(2)
    with c4:
        production_year = st.slider(
            "Production year",
            int(ranges["production_year"]["min"]), int(ranges["production_year"]["max"]),
            int(ranges["production_year"]["median"]),
        )
    with c5:
        st.write("")  # spacer — body type now selected above

    st.markdown("###  Technical specifications")
    c6, c7, c8 = st.columns(3)
    with c6:
        engine_displacement_num = st.number_input(
            "Engine displacement (L)", min_value=0.0, max_value=float(ranges["engine_displacement_num"]["max"]),
            value=float(ranges["engine_displacement_num"]["median"]), step=0.1,
        )
    with c7:
        at_gucu = st.number_input(
            "Engine power (hp)", min_value=1.0, max_value=float(ranges["at_gucu"]["max"]),
            value=float(ranges["at_gucu"]["median"]), step=5.0,
        )
    with c8:
        kilometrage_num = st.number_input(
            "Mileage (km)", min_value=0.0, max_value=float(ranges["kilometrage_num"]["max"]),
            value=float(ranges["kilometrage_num"]["median"]), step=1000.0,
        )

    c9, c10, c11 = st.columns(3)
    with c9:
        transmission = st.selectbox("Transmission", sorted(opts["transmission"]))
    with c10:
        drivetrain = st.selectbox("Drivetrain", sorted(opts["drivetrain"]))
    with c11:
        yanacaq_novu = st.selectbox("Fuel type", sorted(opts["yanacaq_novu"]))

    c12, c13 = st.columns(2)
    with c12:
        seat_count = st.slider("Seat count", int(ranges["seat_count"]["min"]), int(ranges["seat_count"]["max"]), 5)
    with c13:
        market_type = st.selectbox("Market type (origin)", sorted(opts["market_type"]))

    st.markdown("###  Condition & extras")
    c14, c15, c16 = st.columns(3)
    with c14:
        qezali = st.selectbox("Accident history", sorted(opts["Qəzalı"]))
    with c15:
        reng = st.selectbox("Color", sorted(opts["Rəng"]))
    with c16:
        sahibler = st.selectbox("Number of previous owners", sorted(opts["Sahiblər"]))

    c17, c18, c19, c20, c21 = st.columns(5)
    with c17:
        yeni = st.checkbox("Brand new")
    with c18:
        vəziyyeti = st.checkbox("Showroom / full condition")
    with c19:
        kondisioner = st.checkbox("Air conditioning", value=True)
    with c20:
        arxa_kamera = st.checkbox("Rear camera")
    with c21:
        renglenib = st.checkbox("Repainted")

    st.markdown("###  Listing details")
    c22, c23, c24, c25, c26 = st.columns(5)
    with c22:
        barter = st.checkbox("Barter accepted")
    with c23:
        loan = st.checkbox("Loan available")
    with c24:
        salon = st.checkbox("Sold by dealership (salon)")
    with c25:
        vip = st.checkbox("VIP listing")
    with c26:
        featured = st.checkbox("Featured listing")

    shop_name = st.selectbox(
        "Seller",
        ["Fiziki şəxs (Private seller)"] + sorted(s for s in opts["shop_name"] if s != "Fiziki şəxs"),
    )
    shop_name = "Fiziki şəxs" if shop_name.startswith("Fiziki") else shop_name

    submitted = st.form_submit_button(" Predict price", width='stretch')

if submitted:
    input_dict = {
        "city": city,
        "engine_displacement_num": engine_displacement_num,
        "kilometrage_num": kilometrage_num,
        "barter": int(barter),
        "loan": int(loan),
        "salon": int(salon),
        "vip": int(vip),
        "featured": int(featured),
        "shop_name": shop_name,
        "ban_type": ban_type,
        "production_year": production_year,
        "market_type": market_type,
        "Marka": marka,
        "Model": car_model,
        "Qəzalı": qezali,
        "Rəng": reng,
        "Sahiblər": sahibler,
        "transmission": transmission,
        "Vəziyyəti": int(vəziyyeti),
        "Yeni": int(yeni),
        "seat_count": seat_count,
        "drivetrain": drivetrain,
        "at_gucu": at_gucu,
        "yanacaq_novu": yanacaq_novu,
        "Kondisioner": int(kondisioner),
        "Arxa_kamera": int(arxa_kamera),
        "Rənglənib": int(renglenib),
    }

    predicted_price = predict_price(model, input_dict, feature_columns)

    st.divider()
    st.markdown("###  Estimated price")
    low = predicted_price * 0.95
    high = predicted_price * 1.05
    dataset_max_price = ranges["price_azn"]["max"]

    gauge_col, metric_col = st.columns([2, 1])
    with gauge_col:
        st.plotly_chart(
            chart_price_gauge(predicted_price, low, high, dataset_max_price),
            use_container_width=True,
        )
    with metric_col:
        st.metric("Predicted market price", format_azn(predicted_price))
        st.caption(f"Typical range: {format_azn(low)} — {format_azn(high)} (±5%, based on model error margin)")

    st.success(
        f"Based on the specifications provided, the model estimates this {marka} {car_model} "
        f"({production_year}) is worth around **{format_azn(predicted_price)}**."
    )

st.divider()

st.markdown("###  Model performance & explainability")
m1, m2, m3 = st.columns(3)
m1.metric("R² score (test set)", f"{meta['metrics']['r2']:.3f}")
m2.metric("MAE (test set)", format_azn(meta["metrics"]["mae"]))
m3.metric("RMSE (test set)", format_azn(meta["metrics"]["rmse"]))

st.plotly_chart(chart_feature_importance(model, feature_columns), width='stretch')

with st.expander(" About this model"):
    st.write(
        "This is a **Random Forest Regressor** trained on cleaned Turbo.az listings. "
        "Numeric features are scaled, mileage is log-transformed, low-cardinality "
        "categorical features are one-hot encoded, and high-cardinality features "
        "(city, brand, model, seller, color, body type) are target-encoded. "
        "Outliers were removed from the training data using Isolation Forest before fitting."
    )
