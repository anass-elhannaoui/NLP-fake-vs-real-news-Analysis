# Fake News Classification with SVM

This repository contains a Jupyter notebook that builds a machine-learning pipeline to classify news articles into **Factual News** and **Fake News** using Natural Language Processing (NLP) techniques and a Support Vector Machine (SVM) classifier.

The project covers data loading, text cleaning, feature extraction with TF–IDF, model training, and evaluation using accuracy, precision, recall, and F1-score.

---

## 📌 Project Objectives

- Preprocess and clean raw news text (tokenization, lowercasing, stopword removal, etc.)
- Convert text into numerical features using TF–IDF
- Train and evaluate an SVM classifier for fake-news detection
- Analyze precision/recall trade-offs and discuss model limitations

---

## 📂 Dataset

The notebook expects a labeled dataset with two classes:

- **Factual News**
- **Fake News**

Expected columns:

- **text** — content of the article  
- **label** — ground-truth category  

If your dataset is private or cannot be uploaded:

- Use any public fake-news dataset (e.g., Kaggle)
- Or describe the dataset structure in a `DATASET.md`

---

## 🔧 Methods and Pipeline

### 1. Exploratory Data Analysis (EDA)
- Inspect distribution of classes  
- Preview example articles  

### 2. Text Preprocessing
- Lowercasing  
- Removing punctuation and stopwords  
- Tokenization  
- Optional lemmatization or stemming  

### 3. Feature Extraction
- TF–IDF vectorizer fitted on training data  
- Transformation into sparse matrices for model input  

### 4. Model Training
- Linear SVM (`LinearSVC` or `SVC(kernel="linear")`)  
- 80/20 train–test split  

### 5. Evaluation
- Accuracy (≈0.82 in a reference run)  
- Classification report:  
  - **Factual News** → high recall  
  - **Fake News** → high precision but lower recall  

### 6. Discussion
- Observations on recall/precision imbalance  
- Potential effects of dataset quality and size  
- Suggestions for improving generalization  

---

## 📊 Results

Reference performance:

- **Accuracy:** ~0.82  

### Factual News
- Very high recall (almost all factual articles correctly classified)

### Fake News
- Excellent precision (few false positives)
- Lower recall (some fake articles not detected)

Performance may vary depending on dataset, preprocessing, and training conditions.

---

## 🚀 Possible Improvements

To extend or strengthen the model:

### Hyperparameter Tuning
- Grid search on `C`, n-grams, class weights

### Alternative Models
- Logistic Regression  
- Random Forest  
- XGBoost  
- Neural architectures such as LSTM or Transformers  

### Better NLP Features
- N-grams  
- Character-level features  
- Word embeddings  

### Handle Class Imbalance
- Oversampling / undersampling  
- SMOTE  
- Adjusted SVM class weights  

---

## 📄 License

Choose a license (MIT, Apache-2.0, GPL-3.0…) and include it as a `LICENSE` file.

---

## 👤 Author

This project was created as part of a machine-learning study exercise, focusing on end-to-end NLP classification using Python, scikit-learn, and Jupyter notebooks.

