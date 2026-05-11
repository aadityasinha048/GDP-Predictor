import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import numpy as np
from src.predict import predict_scenario

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="GDP Predictor", layout="wide")

st.title("🌍 Global GDP Scenario Simulator")

st.markdown("""
Simulate how economic changes impact GDP across countries.  
Adjust variables in the sidebar and analyze results instantly.
""")

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])
st.sidebar.header("⚙️ Scenario Controls")

# -----------------------------
# Available Features for Scenario
# -----------------------------
feature_labels = {
    "economic__Exports_of_goods_and_services_current_USusd": "Exports",
    "economic__Imports_of_goods_and_services_current_USusd": "Imports",
    "economic__Gross_capital_formation_current_USusd": "Investment",
    "social__Individuals_using_the_Internet_percent_of_population": "Internet Usage (%)",
    "social__Urban_population": "Urban Population",
    "economic__Agriculture_forestry_and_fishing_value_added_current_USusd": "Agriculture Output",
    "economic__Total_reserves_includes_gold_current_USusd": "Total Reserves"
}

all_features = list(feature_labels.keys())

st.markdown("## ⚙️ Scenario Controls")

selected_features = st.multiselect(
    "🎯 Select Features to Modify",
    all_features,
    default=[
        "economic__Exports_of_goods_and_services_current_USusd",
        "economic__Imports_of_goods_and_services_current_USusd"
    ]
)
scenario = {}

st.markdown("### 🎛️ Adjust Feature Changes (%)")

for feature in selected_features:
    clabel = feature_labels.get(feature, feature)
    change = st.slider(f"{clabel} (%)", -100, 100, 0)
    scenario[feature] = change / 100

if uploaded_file:
    df = load_data(uploaded_file)
    if "Time" in df.columns:
        year = st.selectbox("📅 Select Year", sorted(df["Time"].unique(), reverse=True))
        df = df[df["Time"] == year]
    # 🔥 Basic cleaning for user input
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.mean(numeric_only=True))

    if len(scenario) == 0:
        st.warning("⚠️ Please select at least one feature to modify")
        st.stop()

    try:
        with st.spinner("Running prediction..."):
            base, new = predict_scenario(df, scenario)
            st.success("✅ Scenario applied successfully!")
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

    with st.expander("📄 View Raw Data"):
        st.dataframe(df.head())
    
    # -----------------------------
    # Predictions
    # -----------------------------

    result_df = df.copy()
    result_df["Baseline GDP"] = base
    result_df["Scenario GDP"] = new
    result_df["GDP Change"] = new - base

    country_df = result_df.groupby("Country Name").agg({
        "Baseline GDP": "mean",
        "Scenario GDP": "mean",
        "GDP Change": "mean"
    }).reset_index()
    # -----------------------------
    # Format GDP (Readable)
    # -----------------------------
    def format_gdp(x):
        if x >= 1e12:
            return f"${x/1e12:.2f} Trillion"
        elif x >= 1e9:
            return f"${x/1e9:.2f} Billion"
        else:
            return f"${x:,.0f}"

    result_df["Formatted GDP"] = result_df["Scenario GDP"].apply(format_gdp)

    # -----------------------------
    # KPI Metrics
    # -----------------------------
    st.markdown("## 📊 Key Metrics")

    col1, col2, col3 = st.columns(3, gap="large")
    col1.metric("🌍 Baseline GDP(Avg)", format_gdp(np.mean(base)))
    col2.metric("🚀 Scenario GDP(Avg)", format_gdp(np.mean(new)))
    base_mean = np.mean(base)

    if base_mean != 0:
        delta_val = (np.mean(new - base) / base_mean) * 100
    else:
        delta_val = 0

    col3.metric(
        "📊 Net Change",
        format_gdp(np.mean(new - base)),
        delta=f"{delta_val:.2f}%",
        delta_color="inverse" if delta_val < 0 else "normal"
    )

    tab1, tab2, tab3 = st.tabs(["📊 Overview", "🌍 Country Analysis", "📈 Insights"])
    st.markdown(f"🌐 Countries analyzed: **{len(country_df)}**")
    # -----------------------------
    # Results Table
    # -----------------------------
    with tab1:
        st.subheader("📊 Scenario Results")

        display_df = country_df.copy()

        display_df["Baseline GDP"] = display_df["Baseline GDP"].apply(format_gdp)
        display_df["Scenario GDP"] = display_df["Scenario GDP"].apply(format_gdp)
        display_df["Change"] = display_df["GDP Change"].apply(format_gdp)

        if "Country Name" in result_df.columns:
            st.dataframe(display_df)
        else:
            st.dataframe(display_df[["Baseline GDP", "Scenario GDP", "Change"]])

    # -----------------------------
    # Chart
    # -----------------------------
    with tab2:
        st.subheader("🌍 Country-wise GDP Comparison")

        country_list = country_df["Country Name"].unique()

        selected_country = st.selectbox("Select Country", sorted(country_list))

        filtered = country_df[country_df["Country Name"] == selected_country]

        filtered_display = filtered.copy()

        filtered_display["Baseline GDP"] = filtered_display["Baseline GDP"].apply(format_gdp)
        filtered_display["Scenario GDP"] = filtered_display["Scenario GDP"].apply(format_gdp)
        filtered_display["GDP Change"] = filtered_display["GDP Change"].apply(format_gdp)

        st.dataframe(filtered_display)
        if country_df.empty:
            st.error("No data available after filtering")
            st.stop()
        sort_option = st.selectbox(
            "Sort by",
            ["GDP Change", "Scenario GDP", "Baseline GDP"],
            index=0
        )

        sorted_df = country_df.sort_values(sort_option, ascending=False)

        display_df = sorted_df.copy()

        display_df["Baseline GDP"] = display_df["Baseline GDP"].apply(format_gdp)
        display_df["Scenario GDP"] = display_df["Scenario GDP"].apply(format_gdp)
        display_df["Change"] = display_df["GDP Change"].apply(format_gdp)
        st.dataframe(display_df[[
            "Country Name",
            "Baseline GDP",
            "Scenario GDP",
            "Change"
        ]],use_container_width=True)

        st.subheader("📊 Country-wise GDP Change")

        chart_df = sorted_df.head(15).set_index("Country Name")[["GDP Change"]]

        st.bar_chart(chart_df)

    with tab3:
        top_gainers = country_df.sort_values("GDP Change", ascending=False).head(5)
        top_losers = country_df.sort_values("GDP Change").head(5)

        st.subheader("🚀 Top Gainers")
        top_gainers_display = top_gainers.copy()
        top_gainers_display["GDP Change"] = top_gainers_display["GDP Change"].apply(format_gdp)

        st.dataframe(top_gainers_display[["Country Name", "GDP Change"]])

        st.subheader("📉 Top Losers")
        top_losers_display = top_losers.copy()
        top_losers_display["GDP Change"] = top_losers_display["GDP Change"].apply(format_gdp)

        st.dataframe(top_losers_display[["Country Name", "GDP Change"]])