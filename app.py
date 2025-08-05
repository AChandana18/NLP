# app.py

import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK resources (run only once)
nltk.download('stopwords')
nltk.download('punkt')

# Load saved model and vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Load stopwords
stop_words = set(stopwords.words('english'))

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Streamlit UI
st.title("🎬 Movie Review Sentiment Classifier")
st.write("Enter a movie review and find out if it's **Positive 😊** or **Negative 😠**!")

review_input = st.text_area("Enter your review here:", height=150)

if st.button("Predict Sentiment"):
    if review_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = preprocess_text(review_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        sentiment = "Positive 😊" if prediction == 1 else "Negative 😠"
        st.success(f"**Sentiment:** {sentiment}")
