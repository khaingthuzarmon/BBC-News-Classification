NLP Text Classification Streamlit App
This is a simple web application built with Streamlit that classifies text into one of the following categories: Business, Entertainment, Politics, Sport, or Tech. The app uses a pre-trained machine learning model.

🚀 How to Run This Application
Follow these steps to get the application running on your local machine.

Prerequisites

You need to have Python 3.7+ installed.

1. Clone the Repository

First, clone this repository to your local machine:

git clone <your-repository-url>
cd <your-repository-name>

2. Create a Virtual Environment (Recommended)

It's a good practice to create a virtual environment to manage project dependencies.

# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

3. Install Dependencies

Install all the required Python libraries using the requirements.txt file.

pip install -r requirements.txt

4. Run the Streamlit App

Now you can run the application with the following command:

streamlit run app.py

Your web browser should open a new tab with the running application.

📁 Project Files
app.py: The main Python script that contains the Streamlit application code.

model.pkl: The serialized, pre-trained machine learning model.

vectorizer: The serialized TF-IDF vectorizer used to transform text data.

requirements.txt: A list of all Python libraries required to run the application.

README.md: This file, providing instructions and information about the project.

🛠️ How It Works
User Input: You enter a piece of text into the text area.

Preprocessing: The application cleans the input text by converting it to lowercase, removing punctuation, URLs, and stopwords.

Vectorization: The cleaned text is transformed into a numerical representation using the pre-trained TF-IDF vectorizer.

Prediction: The machine learning model takes the vectorized text as input and predicts which category it belongs to.

Display Result: The predicted category and the model's confidence scores for each category are displayed on the screen.
