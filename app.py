from flask import Flask, render_template, request
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import SnowballStemmer
import joblib
import os
import nltk

# Use pre-downloaded nltk data
nltk.data.path.append(os.path.join(os.path.dirname(__file__), 'nltk_data'))

# Paths to model files
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

#load models
voting_classifier = joblib.load(os.path.join(MODEL_DIR, 'voting_classifier_model_Disease_pred_97_percent_acc.pkl'))
tfidf_vectorizer = joblib.load(os.path.join(MODEL_DIR, 'tfidf_vectorizer_disease_nlp.joblib'))
label_encoder = joblib.load(os.path.join(MODEL_DIR, 'label_encoder_disease_nlp.joblib'))

app = Flask(__name__)


tokenizer = TreebankWordTokenizer()

def preprocess_text(text):
    tokens = tokenizer.tokenize(text)
    stemmer = SnowballStemmer('english')
    tokens = [stemmer.stem(token.lower()) for token in tokens if token.isalpha()]
    return ' '.join(tokens)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_disease', methods=['POST'])
def detect_disease():
    user_input = request.form['symptoms']
    processed = preprocess_text(user_input)
    transformed = tfidf_vectorizer.transform([processed])
    prediction = voting_classifier.predict(transformed)
    predicted_label = label_encoder.inverse_transform(prediction)[0]
    return render_template('result.html', disease=predicted_label)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
