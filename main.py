from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow all websites
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load model and vectorizer
model = pickle.load(open("model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

@app.get("/")
def home():
    return {"message": "SMS Spam Classifier API Running"}

@app.get("/predict")
def predict(message: str):
    
    transformed_message = vectorizer.transform([message])
    prediction = model.predict(transformed_message)[0]
    
    if prediction == 1:
        result = "Spam"
    else:
        result = "Not Spam"
        
    return {"prediction": result}