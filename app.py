
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from collections import Counter as ctr
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

df = pd.read_csv('Symptom2Disease.csv')
df.head(10)

df.drop(['Unnamed: 0'], axis=1, inplace=True)

ctr(df['label'])

df.sample(10)

def preprocess_text(text):
    tokens = word_tokenize(text)
    snowball_stemmer = SnowballStemmer('english')
    tokens = [snowball_stemmer.stem(token.lower()) for token in tokens if token.isalpha()]
    return ' '.join(tokens)

df['text'] = df['text'].apply(preprocess_text)

df.sample(5)

tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(df['text'])

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

base_models = [
    ('nb', MultinomialNB()),
    ('rf', RandomForestClassifier()),
    ('lr', LogisticRegression()),
    ('svm', SVC(kernel='linear', probability=True))
]

voting_classifier = VotingClassifier(estimators=base_models, voting='hard')

voting_classifier.fit(X_train, y_train)

accuracy = voting_classifier.score(X_test, y_test)
print("Accuracy:", accuracy)

import joblib

joblib.dump(voting_classifier, 'voting_classifier_model_Disease_pred_97_percent_acc.pkl')

loaded_model = joblib.load('voting_classifier_model_Disease_pred_97_percent_acc.pkl')

# Sample text
sample_text = "I have been experiencing a skin rash on my arm for the past few weeks."
sample_text_processed = preprocess_text(sample_text)
sample_text_transformed = tfidf_vectorizer.transform([sample_text_processed])
predicted_label = label_encoder.inverse_transform(voting_classifier.predict(sample_text_transformed))
print("Predicted Label:", predicted_label)

text = 'i been realli weari and ill i been suffer from..'
sample_text = text
sample_text_processed = preprocess_text(sample_text)
sample_text_transformed = tfidf_vectorizer.transform([sample_text_processed])
predicted_label = label_encoder.inverse_transform(voting_classifier.predict(sample_text_transformed))
print("Predicted Label:", predicted_label)

df.sample()

# --- UPDATED EVALUATION METRICS SECTION ---
# Compute predictions on the test set
y_pred = voting_classifier.predict(X_test)

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Compute precision
precision = precision_score(y_test, y_pred, average='macro')
print("Precision:", precision)

# Compute recall
recall = recall_score(y_test, y_pred, average='macro')
print("Recall:", recall)

# Compute F1-score
f1 = f1_score(y_test, y_pred, average='macro')
print("F1-score:", f1)

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)
# --- END OF UPDATED SECTION ---

joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer_disease_nlp.joblib')
joblib.dump(label_encoder, 'label_encoder_disease_nlp.joblib')

import joblib
voting_classifier = joblib.load('voting_classifier_model_Disease_pred_97_percent_acc.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer_disease_nlp.joblib')
label_encoder = joblib.load('label_encoder_disease_nlp.joblib')

from flask import Flask, render_template, request, redirect, url_for
import joblib
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

app = Flask(__name__)

# Load the model and vectorizer
voting_classifier = joblib.load('voting_classifier_model_Disease_pred_97_percent_acc.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer_disease_nlp.joblib')
label_encoder = joblib.load('label_encoder_disease_nlp.joblib')

# Preprocess the text
def preprocess_text(text):
    tokens = word_tokenize(text)
    snowball_stemmer = SnowballStemmer('english')
    tokens = [snowball_stemmer.stem(token.lower()) for token in tokens if token.isalpha()]
    return ' '.join(tokens)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_disease', methods=['POST'])
def detect_disease():
    user_input = request.form['symptoms']
    user_input_processed = preprocess_text(user_input)
    user_input_transformed = tfidf_vectorizer.transform([user_input_processed])
    predicted_label_encoded = voting_classifier.predict(user_input_transformed)
    predicted_label = label_encoder.inverse_transform(predicted_label_encoded)[0]
    return render_template('result.html', disease=predicted_label)

if __name__ == '__main__':
    app.run(debug=True)
