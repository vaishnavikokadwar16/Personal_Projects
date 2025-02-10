# Command to run the dashboard: streamlit run Chicago_Taxi_Trips_Dashboard.py

"""
#Pastel Color Pallete
"#FFB3BA",  # Light pink
"#FFDFBA",  # Light orange
"#FFFFBA",  # Light yellow
"#BAFFC9",  # Light green
"#BAE1FF",  # Light blue
"#D5BAFF",  # Light purple
- **Light Pink:** `#FFC1CC`
- **Peach:** `#FFDAB9`
- **Pale Yellow:** `#FFFACD`
- **Mint Green:** `#B0E57C`
- **Baby Blue:** `#ADD8E6`
- **Lavender:** `#E6E6FA`
- **Soft Coral:** `#FAD1C9`
- **Powder Blue:** `#B0E0E6`
- **Pale Turquoise:** `#AFEEEE`
- **Pastel Orange:** `#FFD1A9`
- **Soft Lilac:** `#D9B3FF`
- **Blush Pink:** `#FFB6C1`
- **Seafoam Green:** `#9FE2BF`
- **Light Sky Blue:** `#87CEFA`
- **Mauve:** `#E0B0FF`
- **Pale Peach:** `#FFE5B4`
- **Cream:** `#FFFDD0`
- **Pastel Red:** `#FF6961`
- **Pastel Green:** `#77DD77`
- **Soft Aqua:** `#ACE1AF`
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import os
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
from streamlit_lottie import st_lottie
warnings.filterwarnings('ignore')

def main():

    st.set_page_config("Chicago Taxi Trips Analysis", page_icon=":taxi:", layout="wide")
    st.title(":taxi: Chicago Taxi Trips Analysis")
    st.markdown('<style>div.block-container{padding-top:3rem;text-align:center}</style>', unsafe_allow_html=True)

    st.header("Trip Seconds Predictor")
    st_lottie("https://assets3.lottiefiles.com/packages/lf20_ydo1amjm.json", speed=1, height=100)

    os.chdir(r"/Users/vaishnavikokadwar/Documents/Projects/Chicago Taxi Dataset")
    df = pd.read_csv("chicago_taxi_trip_sample.csv")

    #Filter for Taxi Company
    st.sidebar.header("Select your filters: ")
    company = st.sidebar.multiselect("Select the taxi company",df["company"].unique())

    if not company:
        df2 = df
    else:
        df2 = df[df["company"].isin(company)]

    #Filter for Payment Type 
    payment = st.sidebar.multiselect("Select the payment type",df["payment_type"].unique())

    if not payment:
        df3 = df2
    else:
        df3 = df2[df2["payment_type"].isin(payment)]

    min_total, max_total = st.sidebar.slider(
        "Select the range for trip total",
        min_value=float(df3["trip_total"].min()), 
        max_value=float(df3["trip_total"].max()),
        value=(float(df3["trip_total"].min()), float(df3["trip_total"].max()))  # Default range
    )

    df4 = df3[df3["trip_total"].between(min_total, max_total)]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Trip Miles vs Trip Seconds")
        fig = px.scatter(data_frame=df4, x="trip_miles", y="trip_seconds", marginal_x="box", marginal_y="box", opacity=0.3,color_discrete_sequence=["#FFB380"])
        fig.update_layout(xaxis_title="Trip Miles",yaxis_title="Trip Seconds")
        st.plotly_chart(fig, use_container_width=True, height=200)

    with col2:
        st.subheader("Trip Seconds Distribution")
        fig = px.histogram(x='trip_seconds', data_frame=df4,color_discrete_sequence=["#FFB380"])
        st.plotly_chart(fig, use_container_width=True, height=200)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Trips Table")
        st.dataframe(data=df4.head(8))

    with col4:
        st.subheader("Trips Summary")
        st.dataframe(data=df4.describe())

    col5, col6 = st.columns(2)

    with col5:
        st.subheader("Trip Seconds vs Month")
        df_month = df4.groupby("month", as_index=False)["trip_seconds"].mean()
        fig = px.line(data_frame=df_month, x="month", y="trip_seconds",color_discrete_sequence=["#FFB380"])
        fig.update_layout(xaxis_title="Month",yaxis_title="Average Trip Seconds")
        st.plotly_chart(fig, use_container_width=True)


    with col6:
        st.subheader("Trip Seconds vs Day of the Week")
        df_week = df4.groupby("day_of_week", as_index=False)["trip_seconds"].mean()
        fig = px.line(data_frame=df_week, x="day_of_week", y="trip_seconds", color_discrete_sequence=["#FFB380"])
        fig.update_layout(xaxis_title="Day of the Week",yaxis_title="Average Trip Seconds")
        st.plotly_chart(fig, use_container_width=True)
    
    col16, col17, col18 = st.columns(3)
    with col16:
        st.subheader("Distribution of Payment Types")
        df_payment = df4['payment_type'].value_counts().reset_index()
        df_payment.columns = ['payment_type', 'count']
        pastel_palette = ["#FFB3BA",  "#FFDFBA", "#FFFFBA", "#BAFFC9", "#BAE1FF", "#D5BAFF"]
        fig = px.pie(data_frame=df_payment, names='payment_type', values='count', color_discrete_sequence=pastel_palette)
        st.plotly_chart(fig, use_container_width=True)
    
    with col17:
        st.subheader("Distribution of Pickup Areas")
        df_pickup = df4['pickup_area'].value_counts().reset_index()
        df_pickup.columns = ['pickup_area', 'count']
        fig = px.pie(data_frame=df_pickup, names='pickup_area', values='count', color_discrete_sequence=pastel_palette)
        st.plotly_chart(fig, use_container_width=True)
    
    with col18:
        st.subheader("Distribution of Dropoff Areas")
        df_dropoff = df4['dropoff_area'].value_counts().reset_index()
        df_dropoff.columns = ['dropoff_area', 'count']
        fig = px.pie(data_frame=df_dropoff, names='dropoff_area', values='count', color_discrete_sequence=pastel_palette)
        st.plotly_chart(fig, use_container_width=True)


    st.subheader("Choose your parameters for trip seconds prediction")
    col7, col8, col9 = st.columns(3)

    with col7:
        fare = st.number_input("Trip Fare", value=0.0, step =1.0)

    with col8:
        hour = st.selectbox("Hour of the Day", sorted(df4["hour"].unique()))

    with col9:
        month = st.selectbox("Month of the Year", sorted(df4["month"].unique()))

    col10, col11, col12 = st.columns(3)

    with col10:
        week = st.selectbox("Day of the Week", sorted(df4["day_of_week"].unique()))

    with col11:
        pickup_area = st.selectbox("Pickup area", sorted(df4["pickup_area"].unique()))

    with col12:
        dropoff_area = st.selectbox("Dropoff Area", sorted(df4["dropoff_area"].unique()))

    col13, col14, col15 = st.columns(3)

    with col13:
        tips = st.number_input("Tips", value=0.0, step =1.0)

    with col14:
        tips_percent = st.number_input("Tips Percent", value=0.0, step =1.0, max_value=100.0)

    with col15:
        extras = st.number_input("Extras", value=0.0, step =1.0)

    if st.button("Predict Trip Duration"):
        seconds = predict_seconds(df, fare, hour, month, week, pickup_area, dropoff_area, tips, tips_percent, extras)
        st.write(f"The predicted trip duration for the input values is: **{seconds[0]}** seconds or **{seconds[0]/60}** minutes. ")
    
def predict_seconds(df, f, h, m, dw, pa, da, t, tp, e):
    df2 = df[['trip_seconds','fare', 'hour','month','day_of_week','pickup_area','dropoff_area', 'tips','tips_percent','extras']]
    model = smf.ols(formula='trip_seconds ~ fare + hour + month + day_of_week + pickup_area * dropoff_area + tips + tips_percent + extras', data=df2).fit()
    new_data = pd.DataFrame({
        'fare': [f],             
        'hour': [h],              
        'month': [m],             
        'day_of_week': [dw],      
        'pickup_area': [pa],      
        'dropoff_area': [da],     
        'tips': [t],              
        'tips_percent': [tp],     
        'extras': [e]             
    })

    # Make predictions
    return model.predict(new_data)


if __name__ == "__main__":
    main()