import streamlit as st

from utils import (
    load_data,
    render_top_filters,
    get_kpis,
    format_azn,
    apply_custom_theme,
    chart_price_vs_km,
    chart_correlation_heatmap,
)

st.set_page_config(page_title="Explore & Filter | Turbo.az",page_icon=":speech_balloon:", layout="wide")
apply_custom_theme()

df = load_data()

title_col, filter_col = st.columns([5, 1], vertical_alignment="bottom")
with title_col:
    st.title("💬 Explore & Filter")
    st.caption("Slice the dataset with detailed filters, inspect individual listings, and export results.")
with filter_col:
    filtered = render_top_filters(df, key_prefix="explore")

if filtered.empty:
    st.warning("No listings match the current filters. Try widening your selection in the filter panel above.")
    st.stop()

kpis = get_kpis(filtered)
col1, col2, col3 = st.columns(3)
col1.metric("Matching listings", f"{kpis['listings']:,}")
col2.metric("Average price", format_azn(kpis["avg_price"]))
col3.metric("Median price", format_azn(kpis["median_price"]))

st.divider()

st.markdown("### Relationships in the data")
c1, c2 = st.columns([3, 2])
with c1:
    st.plotly_chart(chart_price_vs_km(filtered), width='stretch')
with c2:
    st.markdown("#### Quick column search")
    search_cols = ["city", "Marka", "Model", "ban_type", "yanacaq_novu", "transmission"]
    search_term = st.text_input("Search brand or model", placeholder="e.g. Mercedes, Elantra...")
    if search_term:
        mask = False
        for col in ["Marka", "Model"]:
            mask = mask | filtered[col].astype(str).str.contains(search_term, case=False, na=False)
        st.write(f"Found **{mask.sum():,}** listings matching '{search_term}'.")

st.divider()

st.markdown("### Listings table")
display_cols = [
    "city", "Marka", "Model", "production_year", "kilometrage_num",
    "engine_displacement_num", "transmission", "drivetrain", "yanacaq_novu",
    "ban_type", "at_gucu", "price_azn", "shop_name",
]
st.dataframe(filtered[display_cols].reset_index(drop=True), width='stretch', height=420)

csv_data = filtered[display_cols].to_csv(index=False).encode("utf-8")
st.download_button(
    "Download filtered data as CSV",
    data=csv_data,
    file_name="turboaz_filtered.csv",
    mime="text/csv",
)

with st.expander("Correlation heatmap of numeric features"):
    st.plotly_chart(chart_correlation_heatmap(filtered), width='stretch')
