import streamlit as st
import pandas as pd
import joblib

# Load the model
model = joblib.load('predictive_maintenance_model.pkl')

# Title and description
st.title('Predictive Maintenance in Oil & Gas Industry')
st.write('This app predicts equipment failures based on sensor data.')

# Sidebar for user input
st.sidebar.header('User Input Parameters')

def user_input_features():
    temperature = st.sidebar.slider('Temperature', 0, 100, 50)
    pressure = st.sidebar.slider('Pressure', 200, 500, 300)
    flow_rate = st.sidebar.slider('Flow Rate', 0, 100, 50)
    vibration = st.sidebar.slider('Vibration', 0.0, 5.0, 1.0)
    data = {'temperature': temperature,
            'pressure': pressure,
            'flow_rate': flow_rate,
            'vibration': vibration}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Main panel to display input parameters
st.subheader('User Input Parameters')
st.write(input_df)

# Predict using the model
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader('Prediction')
failure_status = 'Failure' if prediction[0] else 'No Failure'
st.write(failure_status)

st.subheader('Prediction Probability')
st.write(prediction_proba)

# Plot feature importance
st.subheader('Feature Importance')
import matplotlib.pyplot as plt
import numpy as np

# Use model coefficients for feature importance
importances = np.abs(model.coef_[0])
features = input_df.columns
indices = range(len(importances))

plt.figure(figsize=(10, 6))
plt.title('Feature Importances')
plt.barh(indices, importances, color='b', align='center')
plt.yticks(indices, features)
plt.xlabel('Relative Importance')
st.pyplot(plt)
