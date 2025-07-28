import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import plotly.express as px

# Page config
st.set_page_config(page_title="Daily Review Counts", layout="wide")

# Title
st.title("Amazon Customer Reviews Analysis")

# Load data
daily_counts = pd.read_csv('datasets_for_viz/daily_counts_pd_clean.csv')
daily_counts['review_date'] = pd.to_datetime(daily_counts['review_date'])
daily_counts = daily_counts.sort_values('review_date')

monthly_counts = pd.read_csv('datasets_for_viz/monthly_counts_pd_clean.csv')
monthly_counts['year_month'] = pd.to_datetime(monthly_counts['year_month'])
monthly_counts = monthly_counts.sort_values('year_month')

monthly_counts_top = pd.read_csv('datasets_for_viz/monthly_counts_pd_top.csv')
monthly_counts_top['year_month'] = pd.to_datetime(monthly_counts_top['year_month'])
monthly_counts_top = monthly_counts_top.sort_values('year_month')

# Calculate rolling average
daily_counts['rolling_avg'] = daily_counts['count'].rolling(window=7).mean()

# -------------------------------------------
# 🔧 Date range slider
min_date = daily_counts['review_date'].min().date()
max_date = daily_counts['review_date'].max().date()

start_date, end_date = st.slider(
    "Select date range:",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
    format="YYYY-MM-DD"
)

# Filter data based on slider
filtered_df_d = daily_counts[
    (daily_counts['review_date'].dt.date >= start_date) &
    (daily_counts['review_date'].dt.date <= end_date)
]

filtered_df_m = monthly_counts[
    (monthly_counts['year_month'].dt.date >= start_date) &
    (monthly_counts['year_month'].dt.date <= end_date)
]

filtered_df_m_t = monthly_counts_top[
    (monthly_counts_top['year_month'].dt.date >= start_date) &
    (monthly_counts_top['year_month'].dt.date <= end_date)
]

# -------------------------------------------
# 📈 Plot filtered rolling average

col5, col6 = st.columns(2)

with col5:
    fig = px.line(
        filtered_df_d,
        x="review_date",
        y="rolling_avg",
        title="Daily Review Counts (7-Day Rolling Average)",
        labels={"review_date": "Date", "rolling_avg": "7-Day Avg Review Count"},
        log_y=True
    )

    fig.update_layout(
        xaxis=dict(tickangle=270, tickformat="%Y-%m-%d"),
        margin=dict(l=40, r=40, t=60, b=120)
    )

    st.plotly_chart(fig, use_container_width=True)

# 📈 Plot filtered rolling average

with col6:
    fig6 = px.line(
        filtered_df_m,
        x="year_month",
        y="count",
        title="Monthly Review Counts",
        labels={"year_month": "Month-Year", "count": "Review Count"},
        log_y=True
    )

    fig6.update_layout(
        xaxis=dict(tickangle=270, tickformat="%Y-%m"),
        margin=dict(l=40, r=40, t=60, b=120)
    )

    st.plotly_chart(fig6, use_container_width=True)

fig7 = px.line(
    filtered_df_m_t,
    x="year_month",
    y="count",
    color = "short_title",
    title="Monthly Review Counts for Top Categories",
    labels={"year_month": "Month-Year", "count": "Review Count"},
    log_y=True
)

fig7.update_layout(
    xaxis=dict(tickangle=270, tickformat="%Y-%m"),
    margin=dict(l=40, r=40, t=60, b=120)
)

st.plotly_chart(fig7, use_container_width=True)

st.subheader("Review Counts Aggregated Analysis")

df2 = pd.read_csv("datasets_for_viz/reviews_group.csv")
df3 = pd.read_csv("datasets_for_viz/reviews_group_verified.csv")

col1, col2 = st.columns(2)

with col1:
    fig2 = px.bar(
        df2,
        x="rating",
        y="count",
        title="Review Counts by Rating",
        labels={"rating": "Rating", "count": "Review Count"},
        text="count",  # show count on bars
    )

    # Optional layout tweaks
    fig2.update_layout(
        xaxis_title="Rating",
        yaxis_title="Review Count",
        uniformtext_minsize=8,
        uniformtext_mode='hide'
    )

    # Show in Streamlit
    st.plotly_chart(fig2, use_container_width=True)

with col2:
    fig3 = px.bar(
        df3,
        x="verified_purchase",
        y="count",
        title="Review Counts by Verification Status",
        labels={"verified_purchase": "Verification Status", "count": "Review Count"},
        text="count",  # show count on bars
    )

    # Optional layout tweaks
    fig3.update_layout(
        xaxis_title="Verification Status",
        yaxis_title="Review Count",
        uniformtext_minsize=8,
        uniformtext_mode='hide'
    )

    # Show in Streamlit
    st.plotly_chart(fig3, use_container_width=True)

st.subheader("Products Analysis")

df4 = pd.read_csv("datasets_for_viz/category_group.csv")
df5 = pd.read_csv("datasets_for_viz/category_price_avg.csv")

df4['category'] = df4['category'].fillna('Not Available')

col3, col4 = st.columns(2)

with col3:
    fig4 = px.pie(
        df4,
        names="category",
        values="count",
        title="Category Distribution of Products",
        hole=0.4  # for a donut-style chart (optional)
    )

    # Customize layout (optional)
    fig4.update_traces(textinfo='percent+label')

    # Show in Streamlit
    st.plotly_chart(fig4, use_container_width=True)

with col4:
    fig5 = px.bar(
        df5,
        x="category",
        y="avg_price",
        title="Products and Average Price",
        labels={"category": "Category", "avg_price": "Average Price"},
        text="avg_price",  # show count on bars
    )

    # Optional layout tweaks
    fig5.update_layout(
        xaxis_title="Category",
        yaxis_title="Average Price",
        uniformtext_minsize=8,
        uniformtext_mode='hide'
    )

    # Show in Streamlit
    st.plotly_chart(fig5, use_container_width=True)

df6 = pd.read_csv('datasets_for_viz/correlation_data.csv')

col7, col8 = st.columns(2)

with col7:
    fig8 = px.scatter(
        df6,
        x="price",
        y="avg_rating",
        title="Rating vs Price",
        color='category',
        opacity = 0.6,
        labels={"price": "Price of Product", "avg_rating": "Average Rating"},
        size_max=10
    )

    fig8.update_layout(
        xaxis_title="Price of Product",
        yaxis_title="Average Rating",
        uniformtext_minsize=8,
        uniformtext_mode='hide',

        # 📌 Legend below the chart
        legend=dict(
            orientation="h",   # horizontal layout
            yanchor="bottom",
            y=-0.5,            # position below the x-axis (adjust if needed)
            x=0.5,
            xanchor="center",
            bgcolor="rgba(255,255,255,0.8)",  # optional styling
            bordercolor="gray",
            borderwidth=0.5
        ),
        margin=dict(l=40, r=40, t=60, b=120)  # add space at bottom for legend
    )

    st.plotly_chart(fig8, use_container_width=True)

with col8:
    fig9 = px.scatter(
        df6,
        x="price",
        y="review_volume",
        title="Number of Reviews vs Price",
        color='category',
        opacity = 0.6,
        labels={"price": "Price of Product", "review_volume": "Number of Reviews"},
        size_max=10
    )

    fig9.update_layout(
        xaxis_title="Price of Product",
        yaxis_title="Number of Reviews",
        uniformtext_minsize=8,
        uniformtext_mode='hide',

        # 📌 Legend below the chart
        legend=dict(
            orientation="h",   # horizontal layout
            yanchor="bottom",
            y=-0.5,            # position below the x-axis (adjust if needed)
            x=0.5,
            xanchor="center",
            bgcolor="rgba(255,255,255,0.8)",  # optional styling
            bordercolor="gray",
            borderwidth=0.5
        ),
        margin=dict(l=40, r=40, t=60, b=120)  # add space at bottom for legend
    )

    st.plotly_chart(fig9, use_container_width=True)