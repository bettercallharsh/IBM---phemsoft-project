import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# Load the model
model = joblib.load('predictive_maintenance_model.pkl')

# Title and description
st.set_page_config(page_title="Predictive Maintenance", page_icon=":oil_drum:")
st.title('Predictive Maintenance in Oil & Gas Industry')
st.write("""
## Predict Equipment Failures
This app predicts equipment failures based on sensor data. 
Adjust the parameters in the sidebar and click 'Predict' to see the results.
""")

# Sidebar for user input
st.sidebar.header('User Input Parameters')
st.sidebar.write('Adjust the parameters below:')

def user_input_features():
    temperature = st.sidebar.slider('Temperature (°C)', 0, 100, 50)
    pressure = st.sidebar.slider('Pressure (kPa)', 200, 500, 300)
    flow_rate = st.sidebar.slider('Flow Rate (L/min)', 0, 100, 50)
    vibration = st.sidebar.slider('Vibration (mm/s)', 0.0, 5.0, 1.0)
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

# Predict button
if st.button('Predict'):
    # Predict using the model
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.subheader('Prediction')
    failure_status = 'Failure' if prediction[0] else 'No Failure'
    st.write(failure_status)

    st.subheader('Prediction Probability')
    st.write(prediction_proba)

    # Display an image based on the prediction
    if failure_status == 'Failure':
        image = Image.open('failure.jpg')
        st.image(image, caption='Predicted: Failure', use_column_width=True)
    else:
        image = Image.open('no_failure.jpg')
        st.image(image, caption='Predicted: No Failure', use_column_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.write("Developed by [Your Name](https://github.com/your-profile)")
