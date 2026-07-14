import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "turboaz_cars_cleaned.csv"
MODEL_PATH = BASE_DIR / "models" / "price_model.joblib"
META_PATH = BASE_DIR / "models" / "model_meta.json"

PRIMARY_COLOR = "#2E86AB"
ACCENT_COLOR = "#F26430"
PLOTLY_TEMPLATE = "plotly_white"


# Data loading

@st.cache_data(show_spinner="Loading dataset...")
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["day"] = pd.to_datetime(df["day"], errors="coerce")
    return df


@st.cache_resource(show_spinner="Loading prediction model...")
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_model_meta() -> dict:
    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# Filtering

def apply_filters(
    df: pd.DataFrame,
    cities=None,
    markas=None,
    ban_types=None,
    fuel_types=None,
    transmissions=None,
    year_range=None,
    price_range=None,
    km_range=None,
) -> pd.DataFrame:
    """Apply a common set of sidebar filters to the dataframe."""
    filtered = df.copy()

    if cities:
        filtered = filtered[filtered["city"].isin(cities)]
    if markas:
        filtered = filtered[filtered["Marka"].isin(markas)]
    if ban_types:
        filtered = filtered[filtered["ban_type"].isin(ban_types)]
    if fuel_types:
        filtered = filtered[filtered["yanacaq_novu"].isin(fuel_types)]
    if transmissions:
        filtered = filtered[filtered["transmission"].isin(transmissions)]
    if year_range:
        filtered = filtered[
            (filtered["production_year"] >= year_range[0])
            & (filtered["production_year"] <= year_range[1])
        ]
    if price_range:
        filtered = filtered[
            (filtered["price_azn"] >= price_range[0])
            & (filtered["price_azn"] <= price_range[1])
        ]
    if km_range:
        filtered = filtered[
            (filtered["kilometrage_num"] >= km_range[0])
            & (filtered["kilometrage_num"] <= km_range[1])
        ]

    return filtered


# Sidebar filter widget 

def render_top_filters(df: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
    """Render filters in a top-right popover (instead of the sidebar) and return the filtered dataframe.

    Call this from within the right-hand column of a `st.columns([...])` layout, e.g.:

        title_col, filter_col = st.columns([4, 1])
        with title_col:
            st.title("...")
        with filter_col:
            filtered = render_top_filters(df, key_prefix="dash")
    """
    with st.popover("🔎 Filters", use_container_width=True):
        cities = st.multiselect(
            "City", sorted(df["city"].unique()), key=f"{key_prefix}_city"
        )
        markas = st.multiselect(
            "Brand (Marka)", sorted(df["Marka"].unique()), key=f"{key_prefix}_marka"
        )
        ban_types = st.multiselect(
            "Body type", sorted(df["ban_type"].unique()), key=f"{key_prefix}_body"
        )
        fuel_types = st.multiselect(
            "Fuel type", sorted(df["yanacaq_novu"].unique()), key=f"{key_prefix}_fuel"
        )
        transmissions = st.multiselect(
            "Transmission", sorted(df["transmission"].unique()), key=f"{key_prefix}_trans"
        )

        year_min, year_max = int(df["production_year"].min()), int(df["production_year"].max())
        year_range = st.slider(
            "Production year", year_min, year_max, (year_min, year_max), key=f"{key_prefix}_year"
        )

        price_min, price_max = float(df["price_azn"].min()), float(df["price_azn"].max())
        price_range = st.slider(
            "Price (AZN)", price_min, price_max, (price_min, price_max), key=f"{key_prefix}_price"
        )

        km_min, km_max = float(df["kilometrage_num"].min()), float(df["kilometrage_num"].max())
        km_range = st.slider(
            "Mileage (km)", km_min, km_max, (km_min, km_max), key=f"{key_prefix}_km"
        )

        st.caption("Adjust filters — charts and tables update instantly.")

    return apply_filters(
        df,
        cities=cities or None,
        markas=markas or None,
        ban_types=ban_types or None,
        fuel_types=fuel_types or None,
        transmissions=transmissions or None,
        year_range=year_range,
        price_range=price_range,
        km_range=km_range,
    )


def sidebar_filters(df: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
    """Render a shared set of sidebar filters and return the filtered dataframe."""
    st.sidebar.markdown("### 🔎 Filters")

    cities = st.sidebar.multiselect(
        "City", sorted(df["city"].unique()), key=f"{key_prefix}_city"
    )
    markas = st.sidebar.multiselect(
        "Brand (Marka)", sorted(df["Marka"].unique()), key=f"{key_prefix}_marka"
    )
    ban_types = st.sidebar.multiselect(
        "Body type", sorted(df["ban_type"].unique()), key=f"{key_prefix}_body"
    )
    fuel_types = st.sidebar.multiselect(
        "Fuel type", sorted(df["yanacaq_novu"].unique()), key=f"{key_prefix}_fuel"
    )
    transmissions = st.sidebar.multiselect(
        "Transmission", sorted(df["transmission"].unique()), key=f"{key_prefix}_trans"
    )

    year_min, year_max = int(df["production_year"].min()), int(df["production_year"].max())
    year_range = st.sidebar.slider(
        "Production year", year_min, year_max, (year_min, year_max), key=f"{key_prefix}_year"
    )

    price_min, price_max = float(df["price_azn"].min()), float(df["price_azn"].max())
    price_range = st.sidebar.slider(
        "Price (AZN)", price_min, price_max, (price_min, price_max), key=f"{key_prefix}_price"
    )

    km_min, km_max = float(df["kilometrage_num"].min()), float(df["kilometrage_num"].max())
    km_range = st.sidebar.slider(
        "Mileage (km)", km_min, km_max, (km_min, km_max), key=f"{key_prefix}_km"
    )

    st.sidebar.caption("Adjust the filters above — every chart and table on this page updates instantly.")

    return apply_filters(
        df,
        cities=cities or None,
        markas=markas or None,
        ban_types=ban_types or None,
        fuel_types=fuel_types or None,
        transmissions=transmissions or None,
        year_range=year_range,
        price_range=price_range,
        km_range=km_range,
    )


# --------------------------------------------------------------------------
# KPI helpers
# --------------------------------------------------------------------------

def format_azn(value: float) -> str:
    return f"{value:,.0f} AZN"


def get_kpis(df: pd.DataFrame) -> dict:
    return {
        "listings": len(df),
        "avg_price": df["price_azn"].mean() if len(df) else 0,
        "median_price": df["price_azn"].median() if len(df) else 0,
        "avg_year": df["production_year"].mean() if len(df) else 0,
        "avg_km": df["kilometrage_num"].mean() if len(df) else 0,
    }


def apply_custom_theme():
    """Inject app-wide CSS polish (typography, spacing, buttons, popover, dataframe).

    Call this once near the top of every page, right after st.set_page_config().
    Works together with .streamlit/config.toml (which sets the base color theme).
    """
    st.markdown(
        """
        <style>
        /* Tighter, more deliberate vertical rhythm */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        /* Titles */
        h1 {
            font-weight: 800 !important;
            letter-spacing: -0.02em;
        }
        h2, h3 {
            font-weight: 700 !important;
            letter-spacing: -0.01em;
        }

        /* Buttons — rounded, consistent accent */
        .stButton > button, .stDownloadButton > button {
            border-radius: 10px;
            border: 1px solid #2A3040;
            font-weight: 600;
            transition: transform 0.12s ease, border-color 0.12s ease;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            transform: translateY(-1px);
            border-color: #5DADE2;
        }

        /* Popover (filter panel) — card-like */
        div[data-testid="stPopover"] > div {
            border-radius: 12px;
        }

        /* Dataframes — rounded corners, subtle border */
        div[data-testid="stDataFrame"] {
            border: 1px solid #2A3040;
            border-radius: 12px;
            overflow: hidden;
        }

        /* Dividers a bit softer */
        hr {
            border-color: #2A3040 !important;
        }

        /* Metric / plotly chart containers get a touch of breathing room */
        div[data-testid="stMetric"] {
            background: #1B1F2A;
            border: 1px solid #2A3040;
            border-radius: 12px;
            padding: 12px 16px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_cards(cards: list):
    """Render a row of styled, Power-BI-style KPI cards.

    cards: list of (label, value) tuples, e.g.
        [("Total listings", "234,780"), ("Average price", "29,534 AZN")]
    """
    accent_colors = ["#5DADE2", "#58D68D", "#F5B041", "#EC7063", "#AF7AC5", "#48C9B0"]

    card_html = "".join(
        f"""
        <div class="kpi-card" style="--accent: {accent_colors[i % len(accent_colors)]};">
            <div class="kpi-topbar"></div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """
        for i, (label, value) in enumerate(cards)
    )

    st.markdown(
        f"""
        <style>
        .kpi-row {{
            display: flex;
            gap: 16px;
            flex-wrap: wrap;
            margin: 10px 0 28px 0;
        }}
        .kpi-card {{
            position: relative;
            flex: 1;
            min-width: 170px;
            background: linear-gradient(160deg, #1E2330 0%, #171B24 100%);
            border: 1px solid #2A3040;
            border-radius: 16px;
            padding: 20px 22px 18px 22px;
            overflow: hidden;
            box-shadow: 0 6px 18px rgba(0,0,0,0.3);
            transition: transform 0.18s ease, box-shadow 0.18s ease;
        }}
        .kpi-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 12px 26px rgba(0,0,0,0.4);
        }}
        .kpi-topbar {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 5px;
            background: var(--accent);
        }}
        .kpi-label {{
            font-size: 12.5px;
            font-weight: 600;
            color: #9AA5B8;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin: 6px 0 10px 0;
        }}
        .kpi-value {{
            font-size: 30px;
            font-weight: 800;
            color: #F2F4F8;
            letter-spacing: -0.01em;
            line-height: 1.1;
        }}
        </style>
        <div class="kpi-row">{card_html}</div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Plotly chart builders
# --------------------------------------------------------------------------

def chart_price_by_marka(df: pd.DataFrame, top_n: int = 15):
    top_markas = df["Marka"].value_counts().head(top_n).index
    data = df[df["Marka"].isin(top_markas)]
    agg = data.groupby("Marka")["price_azn"].median().sort_values(ascending=False).reset_index()
    fig = px.bar(
        agg, x="Marka", y="price_azn",
        color="price_azn", color_continuous_scale="Blues",
        labels={"price_azn": "Median price (AZN)", "Marka": "Brand"},
        title=f"Median Price by Brand (Top {top_n} most listed)",
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, coloraxis_showscale=False)
    return fig


def chart_listings_by_city(df: pd.DataFrame, top_n: int = 12):
    agg = df["city"].value_counts().head(top_n).reset_index()
    agg.columns = ["city", "listings"]
    fig = px.bar(
        agg, x="listings", y="city", orientation="h",
        color="listings", color_continuous_scale="Teal",
        labels={"listings": "Number of listings", "city": "City"},
        title=f"Top {top_n} Cities by Number of Listings",
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
    return fig


def chart_price_distribution(df: pd.DataFrame):
    fig = px.histogram(
        df, x="price_azn", nbins=60,
        labels={"price_azn": "Price (AZN)"},
        title="Price Distribution",
        color_discrete_sequence=[PRIMARY_COLOR],
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, yaxis_title="Number of listings")
    return fig


def chart_price_vs_year(df: pd.DataFrame):
    agg = df.groupby("production_year")["price_azn"].median().reset_index()
    fig = px.line(
        agg, x="production_year", y="price_azn", markers=True,
        labels={"production_year": "Production year", "price_azn": "Median price (AZN)"},
        title="Median Price by Production Year",
    )
    fig.update_traces(line_color=ACCENT_COLOR)
    fig.update_layout(template=PLOTLY_TEMPLATE)
    return fig


def chart_price_vs_km(df: pd.DataFrame, sample_size: int = 5000):
    data = df.sample(min(sample_size, len(df)), random_state=42) if len(df) else df
    fig = px.scatter(
        data, x="kilometrage_num", y="price_azn", color="yanacaq_novu",
        labels={"kilometrage_num": "Mileage (km)", "price_azn": "Price (AZN)", "yanacaq_novu": "Fuel type"},
        title="Price vs Mileage",
        opacity=0.55,
    )
    fig.update_layout(template=PLOTLY_TEMPLATE)
    return fig


def chart_fuel_share(df: pd.DataFrame):
    agg = df["yanacaq_novu"].value_counts().reset_index()
    agg.columns = ["fuel", "count"]
    fig = px.pie(
        agg, names="fuel", values="count", hole=0.45,
        title="Fuel Type Share",
    )
    fig.update_layout(template=PLOTLY_TEMPLATE)
    return fig


def chart_transmission_share(df: pd.DataFrame):
    agg = df["transmission"].value_counts().reset_index()
    agg.columns = ["transmission", "count"]
    fig = px.pie(
        agg, names="transmission", values="count", hole=0.45,
        title="Transmission Type Share",
        color_discrete_sequence=px.colors.sequential.Sunset,
    )
    fig.update_layout(template=PLOTLY_TEMPLATE)
    return fig


def chart_body_type(df: pd.DataFrame, top_n: int = 10):
    agg = df["ban_type"].value_counts().head(top_n).reset_index()
    agg.columns = ["body_type", "count"]
    fig = px.bar(
        agg, x="body_type", y="count",
        color="count", color_continuous_scale="Purp",
        labels={"body_type": "Body type", "count": "Number of listings"},
        title=f"Top {top_n} Body Types",
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, coloraxis_showscale=False)
    return fig


def chart_correlation_heatmap(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    corr = df[numeric_cols].corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale="RdBu", zmid=0,
        text=np.round(corr.values, 2), texttemplate="%{text}",
    ))
    fig.update_layout(title="Correlation Between Numeric Features", template=PLOTLY_TEMPLATE, height=650)
    return fig


def chart_market_type_price(df: pd.DataFrame):
    fig = px.box(
        df, x="market_type", y="price_azn",
        labels={"market_type": "Market type", "price_azn": "Price (AZN)"},
        title="Price Spread by Market Type",
        color="market_type",
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, showlegend=False)
    return fig


# --------------------------------------------------------------------------
# Generic, column-selectable chart builders
# (let the user pick which categorical column to break a chart down by,
#  same idea as the y_val selectbox pattern used elsewhere in the app)
# --------------------------------------------------------------------------

# Columns that make sense to offer in a "choose a category" selectbox
CATEGORY_COLUMNS = [
    "Marka", "city", "ban_type", "yanacaq_novu", "transmission",
    "drivetrain", "market_type", "Rəng", "Qəzalı", "Sahiblər",
]


def chart_price_by_category(df: pd.DataFrame, category_col: str, top_n: int = 15):
    """Median price by the chosen category column (top N most frequent values)."""
    top_values = df[category_col].value_counts().head(top_n).index
    data = df[df[category_col].isin(top_values)]
    agg = data.groupby(category_col)["price_azn"].median().sort_values(ascending=False).reset_index()
    fig = px.bar(
        agg, x=category_col, y="price_azn",
        color="price_azn", color_continuous_scale="Blues",
        labels={"price_azn": "Median price (AZN)", category_col: category_col},
        title=f"Median Price by {category_col} (Top {top_n})",
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, coloraxis_showscale=False)
    return fig


def chart_count_by_category(df: pd.DataFrame, category_col: str, top_n: int = 12):
    """Number of listings by the chosen category column (top N most frequent values)."""
    agg = df[category_col].value_counts().head(top_n).reset_index()
    agg.columns = [category_col, "listings"]
    fig = px.bar(
        agg, x="listings", y=category_col, orientation="h",
        color="listings", color_continuous_scale="Teal",
        labels={"listings": "Number of listings", category_col: category_col},
        title=f"Top {top_n} {category_col} by Number of Listings",
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
    return fig


def chart_category_share(df: pd.DataFrame, category_col: str, top_n: int = 8):
    """Donut chart share of the chosen category column (top N values, rest grouped as 'Other')."""
    counts = df[category_col].value_counts()
    top = counts.head(top_n)
    other_total = counts.iloc[top_n:].sum()
    if other_total > 0:
        top = pd.concat([top, pd.Series({"Other": other_total})])
    agg = top.reset_index()
    agg.columns = [category_col, "count"]
    fig = px.pie(
        agg, names=category_col, values="count", hole=0.45,
        title=f"{category_col} Share",
    )
    fig.update_layout(template=PLOTLY_TEMPLATE)
    return fig


def chart_price_box_by_category(df: pd.DataFrame, category_col: str, top_n: int = 10):
    """Boxplot of price spread across the chosen category column (top N most frequent values)."""
    top_values = df[category_col].value_counts().head(top_n).index
    data = df[df[category_col].isin(top_values)]
    fig = px.box(
        data, x=category_col, y="price_azn",
        labels={category_col: category_col, "price_azn": "Price (AZN)"},
        title=f"Price Spread by {category_col}",
        color=category_col,
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, showlegend=False)
    return fig


def chart_marka_binary_breakdown(df: pd.DataFrame, y_val: str, top_n: int = 10):
    """Grouped bar: count of top brands split by a binary column (e.g. barter, vip, Yeni)."""
    top_markas = df["Marka"].value_counts().head(top_n).index
    data = df[df["Marka"].isin(top_markas)]
    grouped = data.groupby(["Marka", y_val])[y_val].count().reset_index(name="Count")
    fig = px.bar(
        grouped, x="Marka", y="Count", color=y_val,
        barmode="group", text_auto=True,
        title=f"Top {top_n} Brands — split by '{y_val}'",
        color_discrete_sequence=px.colors.qualitative.Prism,
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(template=PLOTLY_TEMPLATE, xaxis_title="Brand", yaxis_title="Number of listings")
    return fig


def chart_daily_trend(df: pd.DataFrame, y_val: str):
    """Line chart of a metric over time (median price or total views per day)."""
    if y_val == "price_azn":
        agg = df.groupby("day")["price_azn"].median().reset_index()
        y_label = "Median price (AZN)"
    else:
        agg = df.groupby("day")["views"].sum().reset_index()
        y_label = "Total views"

    fig = px.line(
        agg, x="day", y=y_val, markers=True,
        labels={"day": "Date", y_val: y_label},
        title=f"Daily Trend — {y_label}",
    )
    fig.update_traces(connectgaps=True, line_color=ACCENT_COLOR)
    fig.update_layout(template=PLOTLY_TEMPLATE)
    return fig


def chart_feature_importance(model, feature_columns):
    """Build a feature importance chart from the fitted RandomForest pipeline."""
    forest = model.named_steps["forestreg"]
    col_trans = model.named_steps["cleaning"]
    try:
        out_names = col_trans.get_feature_names_out()
    except Exception:
        out_names = [f"f{i}" for i in range(len(forest.feature_importances_))]

    importance = pd.Series(forest.feature_importances_, index=out_names)
    importance = importance.sort_values(ascending=False).head(15)
    fig = px.bar(
        importance[::-1], orientation="h",
        labels={"value": "Importance", "index": "Feature"},
        title="Top 15 Most Important Features (Random Forest)",
        color=importance[::-1].values, color_continuous_scale="Blues",
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, coloraxis_showscale=False, showlegend=False)
    return fig


def chart_price_gauge(predicted_price: float, low: float, high: float, dataset_max: float):
    """Speedometer-style gauge showing the predicted price against the market range."""
    gauge_max = max(dataset_max, high * 1.1)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=predicted_price,
        number={"suffix": " AZN", "valueformat": ",.0f", "font": {"size": 36}},
        title={"text": "Estimated Market Price", "font": {"size": 18}},
        gauge={
            "axis": {"range": [0, gauge_max], "tickformat": ",.0f"},
            "bar": {"color": "#5DADE2", "thickness": 0.28},
            "bgcolor": "#1B1F2A",
            "borderwidth": 1,
            "bordercolor": "#2A3040",
            "steps": [
                {"range": [0, low], "color": "#232838"},
                {"range": [low, high], "color": "#2E3648"},
                {"range": [high, gauge_max], "color": "#232838"},
            ],
            "threshold": {
                "line": {"color": "#F2B134", "width": 4},
                "thickness": 0.9,
                "value": predicted_price,
            },
        },
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=320,
        margin=dict(l=30, r=30, t=60, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#F2F4F8"},
    )
    return fig


# --------------------------------------------------------------------------
# Prediction helper
# --------------------------------------------------------------------------

def predict_price(model, input_dict: dict, feature_columns: list) -> float:
    """Build a single-row dataframe matching training columns and predict."""
    row = pd.DataFrame([{col: input_dict.get(col) for col in feature_columns}])
    prediction = model.predict(row)[0]
    return float(prediction)