# in case of conflicts: python3 -m pip install pandas or similar command
# To run the streamlit app: streamlit run file_name.py

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config("Walmart Sales EDA",page_icon=":bar_chart:",layout="wide")
st.title(":bar_chart: Sample Walmart Sales EDA")
st.markdown('<style>div.block-container{padding-top:2rem;}</style>',unsafe_allow_html=True)

os.chdir(r"C:\Users\Vaishanavi\Downloads\Refactored_Py_DS_ML_Bootcamp-master\Streamlit Project")
df = pd.read_csv("walmart.csv")

#Filter for Age
st.sidebar.header("Select your filters: ")
age = st.sidebar.multiselect("Select the age categories",df["Age"].unique())

if not age:
    df2 = df
else:
    df2 = df[df["Age"].isin(age)]

#Filter for City Category
st.sidebar.header("Select your filters: ")
city_cat = st.sidebar.multiselect("Select the city categories",df2["City_Category"].unique())

if not city_cat:
    df3 = df2
else:
    df3 = df2[df2["City_Category"].isin(city_cat)]

col1, col2 = st.columns(2)

grouped_df1 = df3.groupby(by="Age", as_index=False)["Purchase"].sum()
grouped_df2 = df3.groupby(by="City_Category", as_index=False)["Purchase"].sum()
grouped_df3 = df3.groupby(by="Product_Category",as_index=False)["Purchase"].mean()

with col1:
    st.subheader("Age Category wise sales")
    fig = px.bar(grouped_df1,x="Age",y="Purchase",text=['${:,.2f}'.format(x) for x in grouped_df1["Purchase"]], template="seaborn")
    st.plotly_chart(fig, use_container_width=True, height=200)
with col2:
    st.subheader("City Category wise sales")
    fig = px.pie(df3,values="Purchase",names="City_Category",hole=0.5)
    fig.update_traces(text=df3["City_Category"], textposition="outside")
    st.plotly_chart(fig, use_container_width=True, height=200)

col3, col4 = st.columns(2)

with col3:
    with st.expander("Age Category Wise Data"):
        st.write(grouped_df1.style.background_gradient(cmap="Blues"))
        csv = grouped_df1.to_csv(index=False).encode('utf-8')
        st.download_button("Download Data",data=csv,file_name="AgeCategoryData.csv",mime="text/csv")

with col4:
    with st.expander("City Category Wise Data"):
        st.write(grouped_df2.style.background_gradient(cmap="Oranges"))
        csv = grouped_df2.to_csv(index=False).encode('utf-8')
        st.download_button("Download Data",data=csv,file_name="CityCategoryData.csv",mime="text/csv")

col5, col6 = st.columns(2)

with col5:
    st.subheader("Gender wise order distribution")
    fig = px.histogram(df3,x="Gender",color="Marital_Status")
    st.plotly_chart(fig, use_container_width=True,height = 200)

with col6:
    st.subheader("Occupation wise order distribution")
    fig = px.histogram(df3,x="Occupation",color="Gender")
    st.plotly_chart(fig, use_container_width=True,height = 200)

col7, col8 = st.columns(2)

with col7:
    st.subheader("Mean Purchase for each Product Category")
    fig = px.scatter(grouped_df3,x="Product_Category",y="Purchase")
    st.plotly_chart(fig, use_container_width=True,height = 200)