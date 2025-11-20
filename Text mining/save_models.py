
"""
save_models.py

Run:
    python save_models.py

What it does:
- Loads "book_details.csv" (expects columns "description" and "genres")
- Cleans and preprocesses text (basic cleaning + lemmatization)
- Extracts a single "primary_genre" label (first genre in genres list)
- Trains three models (Logistic Regression, MultinomialNB, LinearSVC) as sklearn Pipelines
  that include TF-IDF vectorization.
- Evaluates each model and saves each pipeline to a pickle file:
    - model_logistic.pkl
    - model_nb.pkl
    - model_svm.pkl
- Demonstrates how to load back a saved model and predict.

Adjust paths / parameters at the top if needed.
"""
import re
import ast
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# sklearn imports
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# nltk (for text cleaning)
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ------------- Config -------------
DATA_PATH = Path("book_details.csv")   # change if your CSV is elsewhere
OUTPUT_DIR = Path("models")
OUTPUT_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42
TEST_SIZE = 0.2
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)
# ----------------------------------

# make sure NLTK data is available
nltk_packages = ["punkt", "stopwords", "wordnet", "omw-1.4"]
for pkg in nltk_packages:
    try:
        nltk.data.find(pkg)
    except Exception:
        print(f"Downloading NLTK package: {pkg}")
        nltk.download(pkg)

stop_words = set(stopwords.words("english"))
lemm = WordNetLemmatizer()

def clean_text(text):
    """Lowercase, remove URLs, keep only letters/spaces, tokenize, remove stopwords (len>2), lemmatize."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)        # remove URLs
    text = re.sub(r"[^a-z\s]", " ", text)              # keep only letters and spaces
    tokens = nltk.word_tokenize(text)
    tokens = [lemm.lemmatize(tok) for tok in tokens if tok not in stop_words and len(tok) > 2]
    return " ".join(tokens)

def parse_genres(g):
    """Convert string representations like \"['Classics','Fiction']\" to Python lists safely."""
    if isinstance(g, list):
        return g
    if pd.isna(g):
        return []
    try:
        # ast.literal_eval is safe for literals
        parsed = ast.literal_eval(g)
        if isinstance(parsed, (list, tuple)):
            return list(parsed)
        else:
            # if it's a single string, return as a single-item list
            return [str(parsed)]
    except Exception:
        # fallback: try splitting by comma
        try:
            return [p.strip().strip("[]'\"") for p in str(g).split(",") if p.strip()]
        except Exception:
            return []

def load_and_prepare(path):
    """Load CSV, drop NaNs, parse genres, keep only first genre as label, clean descriptions."""
    print(f"Loading data from: {path}")
    df = pd.read_csv(path)
    # Drop rows missing description or genres
    df = df.dropna(subset=["description", "genres"]).reset_index(drop=True)
    # parse genres
    df["genres_parsed"] = df["genres"].apply(parse_genres)
    # extract primary (first) genre; drop rows with empty genres_parsed
    primary_genres = []
    indices_to_drop = []
    for idx, g_list in enumerate(df["genres_parsed"]):
        if len(g_list) > 0:
            primary_genres.append(g_list[0])
        else:
            indices_to_drop.append(idx)
    if indices_to_drop:
        df = df.drop(index=indices_to_drop).reset_index(drop=True)
    df["primary_genre"] = primary_genres
    # clean descriptions
    df["clean_description"] = df["description"].apply(clean_text)
    # remove empty cleaned descriptions
    df = df[df["clean_description"].str.strip() != ""].reset_index(drop=True)
    return df

def build_pipelines():
    """Return three sklearn Pipelines (tfidf + classifier)."""
    tfidf = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, ngram_range=TFIDF_NGRAM_RANGE)
    pipe_lr = Pipeline([("tfidf", tfidf), ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))])
    pipe_nb = Pipeline([("tfidf", tfidf), ("clf", MultinomialNB())])
    pipe_svm = Pipeline([("tfidf", tfidf), ("clf", LinearSVC(random_state=RANDOM_STATE))])
    return {"logistic": pipe_lr, "naive_bayes": pipe_nb, "svm": pipe_svm}

def train_and_evaluate(pipelines, X_train, X_test, y_train, y_test):
    """Train each pipeline, evaluate, and save to pickle."""
    results = {}
    for name, pipe in pipelines.items():
        print(f"\nTraining: {name} ...")
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"{name} Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred, zero_division=0))
        # save pipeline
        out_path = OUTPUT_DIR / f"model_{name}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(pipe, f)
        print(f"Saved {name} pipeline to: {out_path}")
        results[name] = {"accuracy": acc, "path": str(out_path)}
    return results

def load_model(path):
    """Load a pickled pipeline and return it."""
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model

def main():
    if not DATA_PATH.exists():
        print(f"ERROR: Data file not found at {DATA_PATH.resolve()}")
        print("Place book_details.csv in the same folder or change DATA_PATH variable.")
        sys.exit(1)

    df = load_and_prepare(DATA_PATH)
    print(f"Prepared dataset with {len(df)} samples and {df['primary_genre'].nunique()} classes.")

    X = df["clean_description"].values
    y = df["primary_genre"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    pipelines = build_pipelines()
    results = train_and_evaluate(pipelines, X_train, X_test, y_train, y_test)
    print("\nAll models trained and saved. Summary:")
    for k, v in results.items():
        print(f" - {k}: acc={v['accuracy']:.4f}, file={v['path']}")

    # Demonstrate loading one model and using it
    example_path = OUTPUT_DIR / "model_logistic.pkl"
    if example_path.exists():
        loaded = load_model(example_path)
        print("\nLoaded logistic model, example predictions:")
        sample_texts = X_test[:5]
        preds = loaded.predict(sample_texts)
        for t, p in zip(sample_texts, preds):
            print(f"pred={p} | text={t[:120]}...")

if __name__ == "__main__":
    main()
