import streamlit as st

from utils import (
    load_data,
    render_top_filters,
    get_kpis,
    format_azn,
    render_kpi_cards,
    apply_custom_theme,
    CATEGORY_COLUMNS,
    chart_price_by_category,
    chart_count_by_category,
    chart_price_distribution,
    chart_price_vs_year,
    chart_category_share,
    chart_price_box_by_category,
)

st.set_page_config(page_title="Market Dashboard | Turbo.az", page_icon=":chart_with_upwards_trend:", layout="wide")
apply_custom_theme()

df = load_data()

title_col, filter_col = st.columns([5, 1], vertical_alignment="bottom")
with title_col:
    st.title("📈 Market Dashboard")
    st.caption("A bird's-eye view of the used-car market, powered by interactive Plotly charts.")
with filter_col:
    filtered = render_top_filters(df, key_prefix="dash")

if filtered.empty:
    st.warning("No listings match the current filters. Try widening your selection in the filter panel above.")
    st.stop()

kpis = get_kpis(filtered)
render_kpi_cards([
    ("Listings", f"{kpis['listings']:,}"),
    ("Average price", format_azn(kpis["avg_price"])),
    ("Median price", format_azn(kpis["median_price"])),
    ("Average mileage", f"{kpis['avg_km']:,.0f} km"),
])

st.divider()

# --- Row 1: price & count by a user-chosen category column ---
row1_col1, row1_col2 = st.columns(2)
with row1_col1:
    price_cat_col = st.selectbox(
        "Group price by", CATEGORY_COLUMNS, index=0, key="dash_price_cat"
    )
    st.plotly_chart(chart_price_by_category(filtered, price_cat_col), width='stretch')
with row1_col2:
    count_cat_col = st.selectbox(
        "Group listing count by", CATEGORY_COLUMNS, index=1, key="dash_count_cat"
    )
    st.plotly_chart(chart_count_by_category(filtered, count_cat_col), width='stretch')

st.divider()

# --- Row 2: fixed price trend charts ---
row2_col1, row2_col2 = st.columns(2)
with row2_col1:
    st.plotly_chart(chart_price_distribution(filtered), width='stretch')
with row2_col2:
    st.plotly_chart(chart_price_vs_year(filtered), width='stretch')

st.divider()

# --- Row 3: share (donut) chart by a user-chosen category column ---
share_cat_col = st.selectbox(
    "Show share of", CATEGORY_COLUMNS, index=CATEGORY_COLUMNS.index("yanacaq_novu"), key="dash_share_cat"
)
st.plotly_chart(chart_category_share(filtered, share_cat_col), width='stretch')

st.divider()

# --- Row 4: price spread (boxplot) by a user-chosen category column ---
box_cat_col = st.selectbox(
    "Show price spread by", CATEGORY_COLUMNS, index=CATEGORY_COLUMNS.index("market_type"), key="dash_box_cat"
)
st.plotly_chart(chart_price_box_by_category(filtered, box_cat_col), width='stretch')