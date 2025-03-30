import streamlit as st
import requests
from dotenv import load_dotenv
import os
load_dotenv()

st.title("Email/SMS Spam Classifier")
url_api = os.getenv('url')
input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    
    url = f'{url_api}/predict/sms-spam'
    r = requests.post(url, json={'message': input_sms})
    output = r.json()
    if output['prediction'] == 'Spam':
        st.error(f'##### The Message is {output["prediction"]}')
    else:
        st.success(f'##### The Message is {output["prediction"]}')