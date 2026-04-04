import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# print("Current Working Directory:", os.getcwd())  # Debug: Remove or replace with logging in production
# print("Files in this folder:", os.listdir())  # Debug: Remove or replace with logging in production

from flask import Flask, render_template, request, jsonify
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# 1. Initialize Flask and NLTK
app = Flask(__name__)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# 2. Load the NEW Colab Files (Ensure these are in your Models folder)
# If you didn't rename them, they are probably 'sentiment_model.pkl' and 'vectorizer.pkl'
model_path = os.path.join("Models", "sentiment_model.pkl")
vectorizer_path = os.path.join("Models", "vectorizer.pkl")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
if not os.path.exists(vectorizer_path):
    raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")

model = pickle.load(open(model_path, "rb"))
vectorizer = pickle.load(open(vectorizer_path, "rb"))

stemmer = PorterStemmer()
stop_words = None

def get_stop_words():
    global stop_words
    if stop_words is None:
        try:
            sw = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords')
            sw = set(stopwords.words('english'))
        # 'not' is removed from stop_words to preserve negation in sentiment analysis,
        # which is important for distinguishing phrases like "not good" from "good".
        if 'not' in sw:
            sw.remove('not')
        stop_words = sw
    return stop_words

def clean_text(text):
    text = str(text).lower()

    text = text.replace("did not", "not")
    text = text.replace("didn't", "not")
    text = text.replace("does not", "not")
    text = text.replace("doesn't", "not")
    text = text.replace("is not", "not")
    text=text.replace("did not like", "not")
    text=text.replace("didn't like", "not")

    


    # Handle negation phrases to match your Colab logic
    text = text.replace("not good", "not_good")
    text = text.replace("not bad", "not_bad")
    text = text.replace("not working", "not_working")
    text = text.replace("not worth", "not_worth")
    text = text.replace("not satisfy", "not_satisfy")
    
    # Keep underscores, remove other punctuation
    text = re.sub('[^a-zA-Z_]', ' ', text)
    text = text.split()
    
    # Stemming and Stopword removal
    sw = get_stop_words()
    text = [stemmer.stem(word) for word in text if word not in sw]
    return " ".join(text)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get text from the frontend
        data = request.get_json()
        user_input = data.get("text", "")

        if not user_input:
            return jsonify({"prediction": "Please enter some text."})

        # Process: Clean -> Vectorize -> Predict
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]

        # Map 1 to Positive and 0 to Negative
        result = "Positive" if prediction == 1 else "Negative"
        return jsonify({"prediction": result})

    except KeyError as ke:
        return jsonify({"error": f"Missing key: {str(ke)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)