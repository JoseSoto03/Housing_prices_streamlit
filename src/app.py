import os
from utils import db_connect

if os.getenv('DATABASE_URL'):
    engine = db_connect()

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Predicción en Vivo", page_icon="🤖", layout="centered")

@st.cache_resource
def load_and_train_model():
    housing = fetch_california_housing()
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = pd.Series(housing.target, name='MedHouseVal')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    return model, X, score

model, X, model_score = load_and_train_model()

st.title("🤖 Predicción de Viviendas en California")
st.metric(label="Rendimiento del modelo (R²)", value=f"{model_score:.2f}")
st.sidebar.header("Parámetros de la Vivienda:")

def user_inputs():
    data = {}
    for feature in X.columns:
        min_val = float(X[feature].min())
        max_val = float(X[feature].max())
        mean_val = float(X[feature].mean())
        if min_val == max_val:
            max_val += 0.1
        data[feature] = st.sidebar.slider(
            label=feature,
            min_value=min_val,
            max_value=max_val,
            value=mean_val
        )
    return pd.DataFrame(data, index=[0])

input_df = user_inputs()
st.subheader('Parámetros Seleccionados:')
st.write(input_df)

if st.button('Predecir Valor de la Vivienda'):
    prediction = model.predict(input_df)
    predicted_price = prediction[0] * 100000
    st.metric(label="Precio Estimado", value=f"${predicted_price:,.2f}")
    st.balloons()
else:
    st.warning("Ajusta los parámetros y haz clic en el botón para obtener una predicción.")

st.write("---")
st.caption("El modelo se entrena al iniciar la app y su resultado se guarda en caché para mayor eficiencia.")