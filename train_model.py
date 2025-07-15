import nltk
import pandas as pd
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import os

# Optional: download NLTK to local directory
nltk.download('punkt', download_dir='nltk_data')
nltk.download('averaged_perceptron_tagger', download_dir='nltk_data')

nltk.data.path.append('nltk_data')

df = pd.read_csv('Symptom2Disease.csv')
df.drop(['Unnamed: 0'], axis=1, inplace=True)

def preprocess_text(text):
    tokens = word_tokenize(text)
    stemmer = SnowballStemmer('english')
    tokens = [stemmer.stem(token.lower()) for token in tokens if token.isalpha()]
    return ' '.join(tokens)

df['text'] = df['text'].apply(preprocess_text)

X = TfidfVectorizer().fit_transform(df['text'])
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(df['text'])

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

models = [
    ('nb', MultinomialNB()),
    ('rf', RandomForestClassifier()),
    ('lr', LogisticRegression()),
    ('svm', SVC(kernel='linear', probability=True))
]

voting_classifier = VotingClassifier(estimators=models, voting='hard')
voting_classifier.fit(X_train, y_train)

os.makedirs('models', exist_ok=True)

# Save models
joblib.dump(voting_classifier, 'models/voting_classifier_model_Disease_pred_97_percent_acc.pkl')
joblib.dump(tfidf_vectorizer, 'models/tfidf_vectorizer_disease_nlp.joblib')
joblib.dump(label_encoder, 'models/label_encoder_disease_nlp.joblib')
