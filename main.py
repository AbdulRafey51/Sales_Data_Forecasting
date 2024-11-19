import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Define seasons for filtering
SEASON_MONTHS = {
    "Winter": ["Dec", "Jan", "Feb"],
    "Spring": ["Mar", "Apr", "May"],
    "Summer": ["Jun", "Jul", "Aug"],
    "Autumn": ["Sep", "Oct", "Nov"],
}


# %% Load Data
@st.cache_data
def load_sales_data(file_path):
    """Loads and preprocesses sales data."""
    sales = pd.read_excel(file_path)

    # Fix product descriptions and codes
    sales["PRODUCT_DESCRIPTION"] = sales["PRODUCT_DESCRIPTION"].str.replace(r"[^a-zA-Z0-9\s]", " ",
                                                                            regex=True).str.lower()
    sales["PRODUCT_CODE"] = sales["PRODUCT_CODE"].str.replace("-", "").astype(int)

    # Fix the MONTH column and ensure correct year
    sales["MONTH"] = pd.to_datetime(
        sales["MONTH"].str[:3] + "-01",  # Parse the first 3 letters as the month
        format='%b-%d',  # Use month-day format
        errors='coerce'
    )
    sales["YEAR"] = sales["MONTH"].dt.year  # Extract valid year

    # Correct invalid or missing years
    if sales["YEAR"].min() < 1900:
        sales["YEAR"] = pd.to_datetime(sales["MONTH"]).dt.year

    # Handle invalid or negative quantities
    sales["QUANTITY"] = np.where(sales["QUANTITY"] < 0, 0, sales["QUANTITY"])
    return sales


# %% Filter by Season
def filter_by_season(df, season):
    """Filters data for the specified season."""
    if season == "All":
        return df
    months = SEASON_MONTHS[season]
    return df[df["MONTH"].dt.strftime("%b").isin(months)]


# %% Forecast Function
def forecast_sales(df, product_code, forecast_periods=12):
    """Generates seasonal forecasts for a specific product code."""
    product_df = df[df["PRODUCT_CODE"] == product_code].copy()

    if product_df.empty:
        return None, None, None

    product_description = product_df["PRODUCT_DESCRIPTION"].iloc[0] if not product_df[
        "PRODUCT_DESCRIPTION"].empty else f"Product {product_code}"
    product_df.set_index("MONTH", inplace=True)
    product_df = product_df.resample("M").sum()  # Resample to monthly frequency

    # Check if sufficient data exists
    if len(product_df) < 24:
        model = ExponentialSmoothing(product_df["QUANTITY"], trend="add").fit()
    else:
        model = ExponentialSmoothing(
            product_df["QUANTITY"], trend="add", seasonal="add", seasonal_periods=12
        ).fit()

    forecast = model.forecast(forecast_periods)
    return product_df, forecast, product_description


# %% Visualization Function
def plot_forecasts(forecasts):
    """Plots forecasts and historical data for multiple product codes."""
    fig = go.Figure()

    for result in forecasts:
        product_code = result["product_code"]
        product_description = result["product_description"]
        historical_data = result["historical_data"]
        forecast = result["forecast"]

        # Plot historical data
        fig.add_trace(go.Scatter(
            x=historical_data.index,
            y=historical_data["QUANTITY"],
            mode="lines",
            name=f"{product_description} (Historical)"
        ))

        # Plot forecast data
        forecast_index = pd.date_range(
            start=historical_data.index[-1] + pd.DateOffset(months=1),
            periods=len(forecast),
            freq="M"
        )
        fig.add_trace(go.Scatter(
            x=forecast_index,
            y=forecast,
            mode="lines",
            name=f"{product_description} (Forecast)"
        ))

    # Customize layout
    fig.update_layout(
        title="Sales Forecasts for Selected Products",
        xaxis_title="Month",
        yaxis_title="Quantity",
        legend_title="Products",
    )
    return fig


# %% Streamlit UI
if __name__ == "_main_":
    st.title("Sales Forecasting Dashboard")

    # File upload
    uploaded_file = st.file_uploader("Upload Sales Data (Excel format)", type=["xls", "xlsx"])
    if uploaded_file:
        sales = load_sales_data(uploaded_file)

        # Display a sample of the data
        st.write("Sample Data:")
        st.dataframe(sales.sample(5))

        # Season selection
        season = st.selectbox("Select Season for Filtering", options=["All", "Winter", "Spring", "Summer", "Autumn"])
        filtered_sales = filter_by_season(sales, season)

        # Product selection
        product_codes = filtered_sales["PRODUCT_CODE"].unique().tolist()
        selected_product_codes = st.multiselect(
            "Select Product Codes for Forecasting",
            product_codes,
            default=product_codes[:3] if len(product_codes) >= 3 else product_codes
        )

        # Generate forecasts
        forecasts = []
        for product_code in selected_product_codes:
            historical_data, forecast, product_description = forecast_sales(filtered_sales, product_code)
            if historical_data is not None:
                forecasts.append({
                    "product_code": product_code,
                    "product_description": product_description,
                    "historical_data": historical_data,
                    "forecast": forecast
                })

        # Plot forecasts
        if forecasts:
            st.plotly_chart(plot_forecasts(forecasts))
        else:
            st.warning("No data available for the selected products.")