import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import os

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')  # Add this line


ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


base_path = os.path.dirname(__file__)
vectorizer_path = os.path.join(base_path, 'email-spam-classifier', 'vectorizer.pkl')
model_path = os.path.join(base_path, 'email-spam-classifier', 'model.pkl')

with open(vectorizer_path, 'rb') as file:
    tfidf = pickle.load(file)
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# tfidf = pickle.load(open('email-spam-classifier\\vectorizer.pkl','rb'))
# model = pickle.load(open('email-spam-classifier\\model.pkl','rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
