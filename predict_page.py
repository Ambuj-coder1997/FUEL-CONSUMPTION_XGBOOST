import streamlit as st
import pickle
import numpy as np
from PIL import Image
image = Image.open('FuelPredx_logo.jpg')

def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data


data = load_model()

regressor = data["model"]


def show_predict_page():
    st.title("FuelPredx : Software for Tractor Fuel Consumption Prediction")
    st.sidebar.image(image, use_column_width=True)
    st.write("""### We need some information to predict the Fuel Consumption""")

    Tractor_PTO = st.number_input('Enter PTO Power (kW)')
    Engine_Speed = st.number_input('Enter Engine Speed (RPM)')
    Speed_Depression = st.number_input('Enter Speed Depression (RPM)')

    ok = st.button("Calculate Fuel Consumption")
    if ok:
        X = np.array([[Tractor_PTO, Engine_Speed, Speed_Depression]])
        X = X.astype(float)

        salary = regressor.predict(X)
        st.subheader(f"The estimated Fuel Consumption(L/h) is {salary[0]:.2f}")



