# 🩺 Disease Prediction App (Flask + ML)

This is a Flask web application that predicts possible diseases based on user-entered symptoms using a trained machine learning ensemble model.

## 🚀 Demo

🌐 [Live App on Render](https://disease-prediction-nlp.onrender.com/)  


---

## 🧠 Model Overview

The model is a `VotingClassifier` combining:
- Multinomial Naive Bayes
- Random Forest
- Logistic Regression
- Support Vector Machine

It is trained on a symptom-to-disease mapping dataset.

---

## 📂 Project Structure

```
├── app.py                        # Flask app (inference only)
├── train_model.py               # (Optional) model training script
├── models/                      # Trained model + vectorizer + encoder
│   ├── voting_classifier_model_Disease_pred_97_percent_acc.pkl
│   ├── tfidf_vectorizer_disease_nlp.joblib
│   └── label_encoder_disease_nlp.joblib
├── nltk_data/                   # Bundled NLTK tokenizer/tagger files
├── templates/
│   ├── index.html               # Home form UI
│   └── result.html              # Result display UI
├── requirements.txt             # Dependencies
├── render.yaml                  # Render deployment config
└── Symptom2Disease.csv          # Training data (optional)
```

---

## 🛠 How to Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/Sagar-Pariyar/disease-prediction
cd disease-prediction
```

### 2. Set up environment
```bash
pip install -r requirements.txt
```

### 3. Train the model (optional, only needed once)
```bash
python train_model.py
```

### 4. Run the app
```bash
python app.py
```

Visit `http://localhost:5000` in your browser.

---


## 🧠 Technologies Used

- Python, Flask
- scikit-learn
- Pandas, NLTK
- HTML/CSS (Jinja2 templating)
- Render for deployment

---

## 👤 Author

**Sagar Pariyar**  
GitHub: [@Sagar-Pariyar](https://github.com/Sagar-Pariyar)

This project was developed as a part of Machine Learning Course work in PCCOE, Pune
## Screenshots
<img width="1575" height="855" alt="image" src="https://github.com/user-attachments/assets/9f7deec9-ecab-4f72-9a07-f68f6137f774" />

