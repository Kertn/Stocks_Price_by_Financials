import os
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

# ----------------------------
# App Config
# ----------------------------
st.set_page_config(
    page_title="Stock Predictions Dashboard",
    page_icon="📈",
    layout="wide",
)

# ----------------------------
# Helpers
# ----------------------------
def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize column names (in case)
    rename_map = {
        "Ticker": "Ticker",
        "Name": "Name",
        "Current Price": "Current Price",
        "Predicted Price": "Predicted Price",
    }
    df = df.rename(columns=rename_map)

    # Ensure numeric
    for col in ["Current Price", "Predicted Price"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Compute metrics
    df["Absolute Upside"] = df["Predicted Price"] - df["Current Price"]
    df["Percent Upside"] = np.where(
        df["Current Price"] > 0,
        (df["Predicted Price"] - df["Current Price"]) / df["Current Price"] * 100,
        np.nan,
    )

    # Direction and labels
    df["Direction"] = np.where(
        df["Percent Upside"] > 0, "Long",
        np.where(df["Percent Upside"] < 0, "Short", "Neutral")
    )

    def valuation_label(pct):
        if pd.isna(pct):
            return "N/A"
        if pct > 5:
            return "Undervalued"
        elif pct < -5:
            return "Overvalued"
        else:
            return "Fairly Valued"

    df["Valuation"] = df["Percent Upside"].apply(valuation_label)

    # Magnitudes for sorting
    df["Abs % Upside"] = df["Percent Upside"].abs()
    df["Abs $ Upside"] = df["Absolute Upside"].abs()

    # Round for display
    for c in ["Current Price", "Predicted Price", "Absolute Upside", "Percent Upside", "Abs % Upside", "Abs $ Upside"]:
        df[c] = df[c].round(2)

    return df


def load_data():
    default_path = "file_invest_name.csv"
    uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        return df, "Uploaded file"
    elif os.path.exists(default_path):
        df = pd.read_csv(default_path, index_col=None)
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
        return df, f"Loaded {default_path}"
    else:
        st.info("Please upload your CSV (with columns: Ticker, Name, Current Price, Predicted Price).")
        st.stop()


# ----------------------------
# Sidebar Controls
# ----------------------------
st.sidebar.title("Controls")

df_raw, source_msg = load_data()
df = compute_metrics(df_raw.copy())
st.sidebar.markdown(f"Data source: {source_msg}")

rank_metric = st.sidebar.radio(
    "Ranking metric",
    options=["Percent Upside", "Absolute Upside"],
    index=0,
    help="Choose how to rank the Top list.",
)

# Sort by magnitude toggle (makes +150% and -150% treated equally for ordering)
sort_by_magnitude = st.sidebar.checkbox(
    "Sort by magnitude (abs)",
    value=True,
    help="Order Top list by absolute value of the chosen metric so large positives and negatives rank together.",
)

# Session state for synced min price controls
if "min_price" not in st.session_state:
    st.session_state.min_price = 1.0

max_slider = float(max(1.0, np.nanmax(df["Current Price"]) if len(df) else 1.0))
min_slider = 0.0

def on_min_price_slider_change():
    st.session_state.min_price = st.session_state.min_price_slider

def on_min_price_input_change():
    st.session_state.min_price = st.session_state.min_price_input

st.sidebar.slider(
    "Minimum Current Price filter ($)",
    min_value=min_slider,
    max_value=max_slider,
    value=float(st.session_state.min_price),
    step=0.5,
    key="min_price_slider",
    on_change=on_min_price_slider_change,
    help="Exclude micro/penny stocks to reduce extreme % swings.",
)

direction_filter = st.sidebar.selectbox(
    "Direction filter",
    options=["All", "Long only", "Short only"],
    index=0,
    help="Filter by investment direction (derived from Predicted vs Current).",
)

top_n = st.sidebar.slider("Top N for charts/tables", min_value=5, max_value=50, value=10, step=1)

# ----------------------------
# Header + Quick Filters
# ----------------------------
st.title("📈 Stock Predictions Dashboard")
st.caption("Ranks by model-predicted upside and shows long/short direction. Not financial advice.")

st.markdown("### Quick Filters")
st.number_input(
    "Minimum Current Price ($)",
    min_value=0.0,
    max_value=max(100000.0, max_slider),
    value=float(st.session_state.min_price),
    step=0.5,
    key="min_price_input",
    on_change=on_min_price_input_change,
    help="Same as the sidebar slider—use whichever you prefer.",
)

min_price = float(st.session_state.min_price)

# ----------------------------
# Apply Filters
# ----------------------------
df_filtered = df[(df["Current Price"] >= min_price)].copy()

if direction_filter == "Long only":
    df_filtered = df_filtered[df_filtered["Direction"] == "Long"]
elif direction_filter == "Short only":
    df_filtered = df_filtered[df_filtered["Direction"] == "Short"]

# Decide the primary sort column + magnitude
sort_col = "Percent Upside" if rank_metric == "Percent Upside" else "Absolute Upside"
abs_col = "Abs % Upside" if rank_metric == "Percent Upside" else "Abs $ Upside"

if sort_by_magnitude:
    df_filtered["_sort_key"] = df_filtered[abs_col]
else:
    df_filtered["_sort_key"] = df_filtered[sort_col]

df_filtered = df_filtered.sort_values(by="_sort_key", ascending=False, na_position="last").drop(columns=["_sort_key"]).reset_index(drop=True)

# ----------------------------
# Top List (Magnitude-aware)
# ----------------------------
suffix = "" if direction_filter == "All" else f" ({direction_filter})"
mag_suffix = " | sorted by magnitude" if sort_by_magnitude else ""
st.subheader(f"Top {min(top_n, len(df_filtered))} by {rank_metric}{suffix}{mag_suffix}")

df_top = df_filtered.head(top_n).copy()
df_top.insert(0, "Rank", range(1, len(df_top) + 1))

st.dataframe(
    df_top[
        [
            "Rank",
            "Ticker",
            "Name",
            "Current Price",
            "Predicted Price",
            "Absolute Upside",
            "Percent Upside",
            "Direction",
            "Valuation",
        ]
    ],
    use_container_width=True,
    hide_index=True,
)

# Bar chart: keep signed values but order by magnitude
if not df_top.empty:
    category_order = df_top["Ticker"].tolist()
    fig = px.bar(
        df_top,
        x="Ticker",
        y=sort_col,  # signed values (so negatives show below x-axis)
        color="Direction",
        category_orders={"Ticker": category_order},
        color_discrete_map={"Long": "#16a34a", "Short": "#ef4444", "Neutral": "#9ca3af"},
        hover_data=["Name", "Current Price", "Predicted Price", "Percent Upside", "Absolute Upside"],
        title=f"{sort_col} for Top {min(top_n, len(df_top))}{suffix}{mag_suffix}",
    )
    fig.update_layout(yaxis_title=sort_col, xaxis_title="Ticker", bargap=0.25, height=420)
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Longs vs Shorts snapshot (magnitude-aware) — Direction column removed per request
# ----------------------------
st.markdown("### Longs and Shorts")
colL, colS = st.columns(2)

with colL:
    st.markdown("**Top Longs (by |% Upside|)**")
    df_longs = (
        df[(df["Percent Upside"] > 0) & (df["Current Price"] >= min_price)]
        .sort_values(by="Abs % Upside", ascending=False)
        .head(top_n)
    )
    st.dataframe(
        df_longs[["Ticker", "Name", "Current Price", "Predicted Price", "Percent Upside", "Absolute Upside"]],
        use_container_width=True,
        hide_index=True,
    )

with colS:
    st.markdown("**Top Shorts (by |% Upside|)**")
    df_shorts = (
        df[(df["Percent Upside"] < 0) & (df["Current Price"] >= min_price)]
        .sort_values(by="Abs % Upside", ascending=False)
        .head(top_n)
    )
    st.dataframe(
        df_shorts[["Ticker", "Name", "Current Price", "Predicted Price", "Percent Upside", "Absolute Upside"]],
        use_container_width=True,
        hide_index=True,
    )

# ----------------------------
# Full List (Expandable)
# ----------------------------
with st.expander("See full list (filtered)"):
    st.dataframe(
        df_filtered[
            [
                "Ticker",
                "Name",
                "Current Price",
                "Predicted Price",
                "Absolute Upside",
                "Percent Upside",
                "Abs $ Upside",
                "Abs % Upside",
                "Direction",
                "Valuation",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )
    csv = df_filtered.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered CSV", data=csv, file_name="stocks_filtered.csv", mime="text/csv")

# ----------------------------
# Search by Ticker (respects magnitude toggle)
# ----------------------------
st.subheader("Search by Ticker")
ticker_query = st.text_input("Enter a ticker (e.g., UBER, SATS, PUBM)").strip().upper()

if ticker_query:
    match = df[df["Ticker"].str.upper() == ticker_query]
    if match.empty:
        st.warning(f"No results for ticker: {ticker_query}")
    else:
        row = match.iloc[0]

        # Ranking by selected metric with magnitude if checked
        df_sort_all = df.copy()
        if sort_by_magnitude:
            df_sort_all = df_sort_all.assign(_key=df_sort_all[abs_col]).sort_values(by="_key", ascending=False, na_position="last")
        else:
            df_sort_all = df_sort_all.assign(_key=df_sort_all[sort_col]).sort_values(by="_key", ascending=False, na_position="last")
        df_sort_all = df_sort_all.reset_index(drop=True)
        df_sort_all["Rank (selected)"] = df_sort_all.index + 1

        row_rank = int(df_sort_all[df_sort_all["Ticker"].str.upper() == ticker_query]["Rank (selected)"].iloc[0])
        in_top_10 = row_rank <= 10

        label_metric = f"{'|' if sort_by_magnitude else ''}{sort_col}{'|' if sort_by_magnitude else ''}"
        if in_top_10:
            st.success(f"{ticker_query} is in the Top 10 by {label_metric}! Rank: #{row_rank}")
        else:
            st.info(f"{ticker_query} found. Rank by {label_metric}: #{row_rank}")

        col1, col2, col3, col4 = st.columns([1, 2, 1, 1])
        with col1:
            st.metric("Ticker", row["Ticker"])
        with col2:
            st.metric("Name", row["Name"])
        with col3:
            st.metric("Current Price", f"${row['Current Price']:.2f}")
        with col4:
            st.metric("Predicted Price", f"${row['Predicted Price']:.2f}")

        col5, col6, col7, col8 = st.columns(4)
        with col5:
            st.metric("Percent Upside", f"{row['Percent Upside']:.2f}%")
        with col6:
            st.metric("Absolute Upside", f"${row['Absolute Upside']:.2f}")
        with col7:
            st.metric("Direction", row["Direction"])
        with col8:
            st.metric("Valuation", row["Valuation"])

# ----------------------------
# Notes
# ----------------------------
st.markdown("""
- Percent Upside = (Predicted − Current) / Current × 100.
- Direction: Long if Percent Upside > 0, Short if < 0, Neutral if = 0.
- Sorting by magnitude treats +X and −X equally when ordering.
""")