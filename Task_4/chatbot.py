import json
import pickle
import random

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

with open("intents.json") as file:
    data = json.load(file)

def get_response(message):
    X = vectorizer.transform([message])
    tag = model.predict(X)[0]

    for intent in data["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

    return "Sorry, I didn't understand."