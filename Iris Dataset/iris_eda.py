import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import os
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import GridSearchCV

def predict_species(df,pw,pl,sw,sl):
    features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    x_input = np.array([[sl,sw,pl,pw]])

    #scale features using Min-Max scaler
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])

    #encode the target variable using LabelEncoder
    label_encoder = LabelEncoder()
    df['Species'] = label_encoder.fit_transform(df['Species'])

    X = df.drop(["Species","Id"],axis=1)
    y = df["Species"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, shuffle = True)

    param_grid = {'C': [0.1, 1, 10]}

    grid_search = GridSearchCV(LogisticRegression(), param_grid=param_grid, cv = 5, scoring = 'f1')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(x_input)

    return y_pred


def main():
    st.set_page_config("Iris Dataset EDA",page_icon=":cherry_blossom:",layout="wide")
    st.title(":cherry_blossom: Iris Species Dataset EDA")
    st.markdown('<style>div.block-container{padding-top:2rem;text-align:center}</style>',unsafe_allow_html=True)

    os.chdir(r"C:\Users\Vaishanavi\Downloads\Refactored_Py_DS_ML_Bootcamp-master\Iris Dataset")
    df = pd.read_csv("Iris.csv")

    df_corr = df.drop("Id",axis=1).corr().round(1)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Correlation Heatmap")
        fig = px.imshow(df_corr,text_auto=True)
        st.plotly_chart(fig,use_container_width=True)

    with col4:
        st.subheader("Count of each Species")
        fig = px.histogram(data_frame=df,x="Species")
        st.plotly_chart(fig,use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Petal Width vs Petal Length")
        fig = px.scatter(data_frame=df,x="PetalWidthCm",y="PetalLengthCm",color="Species",marginal_x="box",marginal_y="box")
        st.plotly_chart(fig,use_container_width=True,height=200)

    with col2:
        st.subheader("Sepal Length vs Petal Length")
        fig = px.scatter(data_frame=df,x="SepalLengthCm",y="PetalLengthCm",color="Species",marginal_x="box",marginal_y="box")
        st.plotly_chart(fig,use_container_width=True,height=200)

    st.subheader("Iris Species Predictor")

    col5, col6, col7, col8 = st.columns(4)

    with col5:
        st.subheader("Select Petal Length")
        pl = st.number_input("Enter the desired petal length",min_value=0.0,step=0.1)
    with col6:
        st.subheader("Select Petal Width")
        pw = st.number_input("Enter the desired petal width",min_value=0.0,step=0.1)
    with col7:
        st.subheader("Select Sepal Length")
        sl = st.number_input("Enter the desired sepal length",min_value=0.0,step=0.1)
    with col8:
        st.subheader("Select Sepal Width")
        sw = st.number_input("Enter the desired sepal width",min_value=0.0,step=0.1)

    if st.button("Predict Species"):
        species = predict_species(df,pw,pl,sw,sl)
        st.write("The predicted species for the input values is: ")
        st.write(species)

if __name__=="__main__":
    main()