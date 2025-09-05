import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load('bbc_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

st.title("BBC News Classifier")

# Text input
text = st.text_area("Enter news article:")

# Predict button
if st.button("Classify"):
    if text:
        # Transform and predict
        text_vector = vectorizer.transform([text])
        prediction = model.predict(text_vector)[0]
        
        # Show result
        categories = ['Business', 'Entertainment', 'Sport', 'Tech', 'Politics']
        st.success(f"Category: {categories[prediction]}")
    else:
        st.error("Please enter some text")