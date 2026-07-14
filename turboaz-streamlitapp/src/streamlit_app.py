import streamlit as st
from pathlib import Path

from utils import (
    load_data,
    load_model_meta,
    get_kpis,
    format_azn,
    render_kpi_cards,
    apply_custom_theme,
    chart_marka_binary_breakdown,
    chart_daily_trend,
)

BASE_DIR = Path(__file__).parent
ASSETS_DIR = BASE_DIR / "assets"

st.set_page_config(page_title="Turbo.az cars ", page_icon=str(ASSETS_DIR / "logo.webp"), layout="wide")
st.logo(image =str(ASSETS_DIR / "images.png"), size="large", link=None, icon_image=None)

df = load_data()
meta = load_model_meta()

st.title("Turbo.az Car Market Explorer")
st.markdown(
    "An interactive app for exploring Azerbaijan's used-car market "
    "(scraped from **Turbo.az**) and estimating a fair market price with a "
    "trained Machine Learning model."
)

st.divider()

kpis = get_kpis(df)
render_kpi_cards([
    ("Total listings", f"{kpis['listings']:,}"),
    ("Average price", format_azn(kpis["avg_price"])),
    ("Median price", format_azn(kpis["median_price"])),
    ("Average year", f"{kpis['avg_year']:.0f}"),
    ("Average mileage", f"{kpis['avg_km']:,.0f} km"),
])

tab1, tab2 = st.tabs([" Overview", " Quick Charts"])

with tab1:
    st.subheader("Dataset summary")
    st.dataframe(df.describe(include=[float, int]), width='stretch')

    if st.checkbox("Show raw data (first 1,000 rows)", value=False):
        st.dataframe(df.head(1000), width='stretch')

    st.markdown("---")
    st.markdown("###  What's inside this app")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("####  Market Dashboard")
        st.write(
            "High-level charts on prices, brands, cities, fuel types and body "
            "types across the whole market, with quick filters."
        )
    with c2:
        st.markdown("####  Explore & Filter")
        st.write(
            "Drill into the raw data with detailed filters, a searchable table, "
            "and the ability to download the filtered dataset as CSV."
        )
    with c3:
        st.markdown("####  Price Prediction")
        st.write(
            "Enter a car's specifications and get an estimated market price "
            "from a trained Random Forest model, along with model performance metrics."
        )

    st.info("Use the sidebar on the left to navigate between pages.", icon="👈")

    st.markdown("---")
    st.markdown("### About the data & model")
    st.write(
        f"The dataset contains **{len(df):,}** used-car listings collected from Turbo.az, "
        "covering brand, model, mileage, engine specs, location, seller type and more. "
        "The price prediction model is a **Random Forest Regressor** trained on this data "
        f"after cleaning and outlier removal, reaching an R² of **{meta['metrics']['r2']:.3f}** "
        "on unseen test data — see the Price Prediction page for full details."
    )

with tab2:
    st.subheader("Brand breakdown")
    binary_cols = [col for col in df.columns if df[col].nunique() == 2]
    y_val_hist = st.selectbox("Split by", binary_cols, index=0, key="home_hist_col")
    st.plotly_chart(
        chart_marka_binary_breakdown(df, y_val_hist, top_n=10),
        width='stretch',
    )

    st.markdown("---")

    st.subheader("Daily trend")
    y_val_line = st.selectbox("Metric", ["price_azn", "views"], index=0, key="home_line_col")
    st.plotly_chart(
        chart_daily_trend(df, y_val_line),
        width='stretch',
    )
