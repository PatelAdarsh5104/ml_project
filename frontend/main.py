import streamlit as st
import requests

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    url = 'http://127.0.0.1:8000/predict/sms-spam'
    r = requests.post(url, json={'message': input_sms})
    output = r.json()
    if output['prediction'] == 'Spam':
        st.error(f'##### The Message is {output["prediction"]}')
    else:
        st.success(f'##### The Message is {output["prediction"]}')
    
