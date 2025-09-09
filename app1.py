import streamlit as st
import joblib
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# --- Preprocessing and Helper Functions ---

# Download NLTK data (only if not already downloaded)
try:
    stopwords.words('english')
except LookupError:
    st.info("Downloading NLTK resources (stopwords)...")
    nltk.download('stopwords')
    nltk.download('punkt')

# Load the trained model and vectorizer
try:
    model = joblib.load('model.pkl')
    vectorizer = joblib.load('vectorizer')
    st.success("Model and vectorizer loaded successfully!")
except FileNotFoundError:
    st.error("Model or vectorizer file not found. Please make sure 'model.pkl' and 'vectorizer' are in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the model files: {e}")
    st.stop()

# Define the text cleaning function based on the notebook
def clean_text(text):
    """
    Cleans the input text by removing punctuation, converting to lowercase,
    removing stopwords, and handling other text cleaning steps.
    """
    if not isinstance(text, str):
        return ""

    # Make text lowercase
    text = text.lower()
    # Remove square brackets and their content
    text = re.sub('\[.*?\]', '', text)
    # Remove URLs
    text = re.sub('https?://\S+|www\.\S+', '', text)
    # Remove HTML tags
    text = re.sub('<.*?>+', '', text)
    # Remove punctuation
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    # Remove newline characters
    text = re.sub('\n', '', text)
    # Remove words containing numbers
    text = re.sub('\w*\d\w*', '', text)

    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered_words = [word for word in tokens if word not in stop_words]

    return " ".join(filtered_words)

# --- Streamlit User Interface ---

st.set_page_config(page_title="Text Classifier", layout="wide")

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stApp {
        background-color: #f5f5f5;
    }
    h1 {
        color: #333;
        font-family: 'Arial', sans-serif;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 12px;
        padding: 10px 24px;
        border: none;
        font-size: 16px;
    }
    .stTextArea textarea {
        border: 2px solid #ccc;
        border-radius: 10px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)


st.title("📰 NLP Text Classification App")
st.write(
    "Enter a piece of text (e.g., a news article snippet), and this app will predict its category. "
    "The model was trained on a dataset of BBC news articles."
)

# Text area for user input
user_input = st.text_area("Enter text to classify:", height=200, placeholder="Type or paste your text here...")

# Classify button
if st.button("Classify Text"):
    if user_input:
        with st.spinner('Cleaning text and making a prediction...'):
            # 1. Clean the user's input
            cleaned_input = clean_text(user_input)

            # 2. Vectorize the cleaned text
            vectorized_input = vectorizer.transform([cleaned_input])

            # 3. Predict the category
            prediction = model.predict(vectorized_input)
            prediction_proba = model.predict_proba(vectorized_input)

            # Define category names (assuming these are the categories from your notebook)
            category_names = ['business', 'entertainment', 'politics', 'sport', 'tech']
            predicted_category = category_names[prediction[0]]

            st.subheader("Prediction Result")
            st.success(f"**Predicted Category:** {predicted_category.capitalize()}")

            # Display probabilities
            st.subheader("Prediction Confidence")
            probabilities = prediction_proba[0]
            for i, category in enumerate(category_names):
                st.write(f"{category.capitalize()}: {probabilities[i]:.2%}")

    else:
        st.warning("Please enter some text to classify.")
