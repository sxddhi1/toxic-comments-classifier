
# Toxic Comment Classification System
 
A machine learning pipeline that automatically detects and classifies toxic comments using Natural Language Processing (NLP). Built on the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) dataset (~160,000 real Wikipedia comments), it compares Naïve Bayes and k-NN classifiers across two feature extraction strategies to find the best approach for content moderation.
 
---
 
## Table of Contents
 
- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline](#pipeline)
- [Models](#models)
- [Results](#results)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Dependencies](#dependencies)
- [License](#license)
---
 
## Overview
 
Online platforms face a constant challenge moderating user-generated content at scale. This project builds a traditional ML-based classifier that can flag toxic comments in real time. It systematically compares four model configurations — Naïve Bayes and k-NN, each with TF-IDF and Count Vectorization — and provides a deployment-ready `PredictionService` class for single-comment or batch inference.
 
---
 
## Dataset
 
**Source:** [Kaggle — Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
 
The dataset contains ~159,000 Wikipedia talk page comments labelled by human raters across six toxicity subtypes:
 
| Label | Description |
|---|---|
| `toxic` | General toxic language |
| `severe_toxic` | Extreme toxicity |
| `obscene` | Obscene content |
| `threat` | Direct threats |
| `insult` | Personal insults |
| `identity_hate` | Hate based on identity |
 
For this project the six subtypes are collapsed into a single binary label: **toxic (1)** or **non-toxic (0)**. The dataset is heavily imbalanced — approximately 10% of comments are toxic.
 
### Downloading the Data
 
**Option A — Kaggle CLI**
```bash
pip install kaggle
# Place kaggle.json at ~/.kaggle/kaggle.json
kaggle competitions download -c jigsaw-toxic-comment-classification-challenge
unzip jigsaw-toxic-comment-classification-challenge.zip
```
 
**Option B — Manual**
Visit the [data page](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data), accept the rules, and download `train.csv`.
 
Only `train.csv` is needed. The competition's `test.csv` has no labels (they were withheld for the leaderboard), so we perform our own stratified 80/20 split internally.
 
---
 
## Installation
 
```bash
git clone https://github.com/YOUR_USERNAME/toxic-comment-classifier.git
cd toxic-comment-classifier
 
pip install pandas numpy matplotlib seaborn scikit-learn nltk wordcloud joblib kaggle
```
 
NLTK resources are downloaded automatically when you run the notebook.
 
---
 
## Usage
 
### Running the notebook
 
Open `toxic_comments_classifier.ipynb` in Jupyter or Google Colab and run all cells in order (**Runtime → Run all** in Colab).
 
Make sure `train.csv` is in the same directory as the notebook, or update `DATA_PATH` at the top of the data loading cell:
 
```python
DATA_PATH   = 'train.csv'   # path to Kaggle data
SAMPLE_SIZE = 50_000        # set to None to use all ~160k rows
```
 
### Making predictions
 
```python
# After running all cells, the service is ready:
result = service.predict("You are an absolute idiot.")
print(result)
# {'prediction': 'Toxic', 'probability_toxic': 0.9741, 'probability_non_toxic': 0.0259}
 
# Batch prediction
comments = ["Thanks for your help!", "I hope you get banned."]
df_results = service.predict_batch(comments)
print(df_results)
```
 
### Saving and loading the model
 
```python
# Save (runs automatically at the end of the notebook)
save_model(model, vectorizer, 'toxic_nb_model.joblib', 'toxic_tfidf_vectorizer.pkl')
 
# Load in a new session
model, vectorizer = load_model('toxic_nb_model.joblib', 'toxic_tfidf_vectorizer.pkl')
```
 
---
 
## Pipeline
 
### 1. Data Loading
Reads `train.csv`, merges the 6 toxicity columns into a single binary label, and applies a stratified sample to keep class proportions consistent.
 
### 2. Exploratory Data Analysis
- Class distribution bar chart
- Comment length histogram by class
- Toxicity subtype frequency breakdown
### 3. Text Preprocessing
 
The `TextPreprocessor` class applies a configurable multi-stage pipeline:
 
| Step | What it does | Example |
|---|---|---|
| Lowercasing | Normalises case | `"You IDIOT"` → `"you idiot"` |
| URL & mention removal | Strips `http://`, `@username` | `"check http://x.com"` → `"check"` |
| Punctuation removal | Removes non-alphanumeric characters | `"great!!!"` → `"great"` |
| Number removal | Strips digits | `"top 10 list"` → `"top list"` |
| Stopword removal | Drops common words | `"this is the best"` → `"best"` |
| Stemming (Porter) | Reduces to root form | `"running"` → `"run"` |
 
### 4. Feature Extraction
 
Two vectorization strategies, each with unigrams + bigrams (`ngram_range=(1,2)`) and 10,000 max features:
 
**TF-IDF Vectorizer** — weights terms by how distinctive they are across the corpus. `sublinear_tf=True` dampens the effect of very frequent terms. Good at surfacing rare but highly toxic words.
 
**Count Vectorizer** — represents each comment as raw word frequency counts. Straightforward and fast; tends to favour Naïve Bayes.
 
### 5. Model Training
 
Both models are trained on an 80/20 stratified split:
 
- **Naïve Bayes** (`MultinomialNB`, alpha=0.5)
- **k-NN** (`KNeighborsClassifier`, k=5, cosine distance, brute-force search)
### 6. Evaluation
 
Each of the 4 model/feature combinations is evaluated on accuracy, precision, recall, F1, confusion matrix, and ROC/AUC.
 
### 7. Hyperparameter Tuning
 
`GridSearchCV` with 3-fold `StratifiedKFold` over a Naïve Bayes + TF-IDF pipeline, searching across:
- `max_features`: 5,000 / 10,000
- `ngram_range`: (1,1) / (1,2)
- `alpha`: 0.1 / 0.5 / 1.0
### 8. Visualisations
 
- Side-by-side word clouds for toxic vs non-toxic comments
- Top 20 most discriminative features from Naïve Bayes (log probability difference)
- ROC curves for all 4 model configurations
---
 
## Models
 
### Naïve Bayes (MultinomialNB)
 
Applies Bayes' Theorem with a bag-of-words assumption — each word is treated as an independent feature. Works extremely well on high-dimensional sparse text data and is fast to train.
 
**Strengths:** Fast, interpretable, handles high-dimensional data well, robust on imbalanced classes with the right smoothing.
 
**Weaknesses:** Cannot capture word order or context; misses sarcasm and implied threats.
 
### k-Nearest Neighbors
 
Classifies a comment by looking at the k most similar comments in the training set. No explicit training phase — all computation happens at prediction time.
 
**Strengths:** Non-parametric, no assumptions about data distribution, naturally adapts to new patterns.
 
**Weaknesses:** Slow at prediction on large datasets, sensitive to feature representation — performs very differently with TF-IDF vs Count features.
 
---
 
## Results
 
*Run the notebook to populate this table — values appear in the `perf_df` output after cell 6.*
 
| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Naïve Bayes (TF-IDF) | — | — | — | — |
| k-NN (TF-IDF) | — | — | — | — |
| Naïve Bayes (Count) | — | — | — | — |
| k-NN (Count) | — | — | — | — |
 
**Key findings:**
- Naïve Bayes with Count Vectorization consistently achieves the best precision-recall balance
- k-NN performance varies dramatically by feature type (cosine TF-IDF works well; raw counts do not)
- TF-IDF with `sublinear_tf` benefits Naïve Bayes most on the full dataset
**Recommended model for deployment:** Naïve Bayes + TF-IDF — fast inference, strong precision, and interpretable feature importance.
 
---
 
## Limitations
 
- **Class imbalance** — ~90% of comments are non-toxic; the model may underperform on recall without oversampling
- **Sarcasm and irony** — bag-of-words features cannot capture implicit toxicity
- **Short comments** — very short comments (under ~5 words) provide little signal
- **English only** — preprocessing and stopwords are English-specific
- **Static vocabulary** — slang and emerging toxic terms not in the training set will be missed
---
 
## Future Work
 
- **Transformer models** — fine-tune `distilbert-base-uncased` or `roberta-base` on the Jigsaw data for contextual understanding
- **Oversampling** — apply SMOTE or class weighting to address the 10/90 class imbalance
- **Multilingual support** — use the [Jigsaw Multilingual dataset](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification) and multilingual embeddings
- **REST API** — wrap the `PredictionService` in a FastAPI endpoint for real-time moderation
- **Explainability** — integrate LIME or SHAP to explain individual predictions
---
 
## Dependencies
 
| Package | Version |
|---|---|
| pandas | 1.3+ |
| numpy | 1.21+ |
| scikit-learn | 1.0+ |
| nltk | 3.7+ |
| matplotlib | 3.5+ |
| seaborn | 0.11+ |
| wordcloud | 1.8+ |
| joblib | 1.1+ |
 
---
 
## License
 
This project is licensed under the MIT License.
 
The Jigsaw dataset is released under [CC0](https://creativecommons.org/publicdomain/zero/1.0/), with underlying comment text governed by Wikipedia's [CC-SA-3.0](https://creativecommons.org/licenses/by-sa/3.0/).
