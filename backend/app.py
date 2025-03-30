import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    message: str

ps = PorterStemmer()

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')


def transform_text(text:str):
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
vectorizer_path = os.path.join('email-spam-classifier', 'vectorizer.pkl')
model_path = os.path.join('email-spam-classifier', 'model.pkl')

with open(vectorizer_path, 'rb') as file:
    tfidf = pickle.load(file)
with open(model_path, 'rb') as file:
    model = pickle.load(file)





@app.get("/")
async def root():
    return {"message": "Welcome to the Machine Learning API"}


@app.get("/health-check")
async def health_check():
    return {"status": "healthy"}


@app.post("/predict/sms-spam")
async def predict(message: Message):
    transformed_sms = transform_text(message.message)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]
    return {"prediction": "Spam" if result == 1 else "Not Spam"}   



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv('PORT')))