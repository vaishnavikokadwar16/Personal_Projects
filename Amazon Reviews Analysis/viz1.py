import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import plotly.express as px

# Page config
st.set_page_config(page_title="Daily Review Counts", layout="wide")

# Title
st.title("Daily Review Counts Over Time")

daily_counts = pd.read_csv('Datasets for Viz/daily_counts.csv')

# Show the data (optional)
st.write("Here's a quick preview of the data:")
st.dataframe(daily_counts.head())

daily_counts['review_date'] = pd.to_datetime(daily_counts['review_date'])

# Sort by date (important for rolling average)
daily_counts = daily_counts.sort_values('review_date')

# Calculate 7-day rolling average
daily_counts['rolling_avg'] = daily_counts['count'].rolling(window=7).mean()

# Plot rolling average line
fig = px.line(
    daily_counts,
    x="review_date",
    y="rolling_avg",
    title="Daily Review Counts (7-Day Rolling Average)",
    labels={"review_date": "Date", "rolling_avg": "7-Day Avg Review Count"},
    log_y=True  # Optional: still log-scaled
)

fig.update_layout(
    xaxis=dict(tickangle=90, tickformat="%Y-%m-%d"),
    margin=dict(l=40, r=40, t=60, b=120)
)

# Show plot
st.plotly_chart(fig, use_container_width=True)

st.subheader("Review Counts Aggregated Analysis")

df2 = pd.read_csv("Datasets for Viz/reviews_group.csv")
df3 = pd.read_csv("Datasets for Viz/reviews_group_verified.csv")

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