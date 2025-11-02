"""
TF-IDF + Logistic Regression baseline for clinical note classification.
This is always my first model on a text classification problem — it's fast,
interpretable, and sets a surprisingly high bar for neural approaches.
"""

import json
import os
import sys
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import preprocess_pipeline


def load_clinical_notes(data_path: str) -> tuple[list[str], list[int]]:
    with open(data_path, "r") as f:
        dataset = json.load(f)
    texts = [d["text"] for d in dataset]
    labels = [d["label_id"] for d in dataset]
    return texts, labels


def show_top_features(vectorizer: TfidfVectorizer, model: LogisticRegression, n_top: int = 20):
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefficients = model.coef_[0]

    top_urgent_idx = np.argsort(coefficients)[-n_top:][::-1]
    top_routine_idx = np.argsort(coefficients)[:n_top]

    print(f"\nTop {n_top} features for URGENT:")
    for idx in top_urgent_idx:
        print(f"  {feature_names[idx]:30s} coef={coefficients[idx]:.4f}")

    print(f"\nTop {n_top} features for ROUTINE:")
    for idx in top_routine_idx:
        print(f"  {feature_names[idx]:30s} coef={coefficients[idx]:.4f}")


def train_tfidf_baseline(data_path: str, output_dir: str = "models"):
    print("=" * 60)
    print("TF-IDF + Logistic Regression Baseline")
    print("=" * 60)

    texts, labels = load_clinical_notes(data_path)

    print(f"\nPreprocessing {len(texts)} clinical notes...")
    processed_texts = [preprocess_pipeline(clinical_note) for clinical_note in texts]

    X_train, X_test, y_train, y_test = train_test_split(
        processed_texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # TF-IDF with unigrams and bigrams — bigrams help capture phrases like
    # "chest pain" and "follow up" that carry meaning as a unit
    tfidf_vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )

    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    print(f"TF-IDF matrix shape: {X_train_tfidf.shape}")

    lr_model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    lr_model.fit(X_train_tfidf, y_train)

    cv_scores = cross_val_score(lr_model, X_train_tfidf, y_train, cv=5, scoring="f1")
    print(f"\n5-fold CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    y_pred = lr_model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"\nTest Accuracy: {accuracy:.4f}")
    print(f"Test F1 (weighted): {f1:.4f}")
    print("\nClassification Report:")
    report = classification_report(y_test, y_pred, target_names=["routine", "urgent"])
    print(report)

    show_top_features(tfidf_vectorizer, lr_model, n_top=20)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(tfidf_vectorizer, f)
    with open(os.path.join(output_dir, "lr_model.pkl"), "wb") as f:
        pickle.dump(lr_model, f)

    results = {
        "model": "TF-IDF + Logistic Regression",
        "accuracy": float(accuracy),
        "f1_weighted": float(f1),
        "cv_f1_mean": float(cv_scores.mean()),
        "cv_f1_std": float(cv_scores.std()),
        "report": classification_report(y_test, y_pred, target_names=["routine", "urgent"], output_dict=True),
    }
    with open(os.path.join(output_dir, "tfidf_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_dir}/tfidf_results.json")

    return results


if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    data_path = os.path.join(base_dir, "data", "clinical_notes.json")
    output_dir = os.path.join(base_dir, "models")

    if not os.path.exists(data_path):
        print("Data not found. Generating dataset first...")
        from data import generate_dataset, save_dataset
        dataset = generate_dataset(n_samples=500)
        save_dataset(dataset, os.path.join(base_dir, "data"))

    train_tfidf_baseline(data_path, output_dir)
