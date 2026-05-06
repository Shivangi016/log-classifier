# train.py
# Run this file to train your ML model.
# Command: python train.py

import sys
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Add current directory to path so we can import from app/
sys.path.insert(0, '.')
from app.classifier import get_bert_embedding, BERT_AVAILABLE


def train():

    # ── STEP 1: Check BERT is available ──
    if not BERT_AVAILABLE:
        print("ERROR: BERT is not available. Check your transformers install.")
        return

    # ── STEP 2: Load your CSV data ──
    print("\n STEP 1: Loading training data...")
    data_path = "data/logs.csv"

    if not os.path.exists(data_path):
        print(f"ERROR: {data_path} not found. Did you complete Phase 2?")
        return

    df = pd.read_csv(data_path)
    print(f"   Loaded {len(df)} log samples")
    print(f"   Categories found:")
    for label, count in df['label'].value_counts().items():
        print(f"      {label}: {count} samples")

    # ── STEP 3: Convert each log to BERT embeddings ──
    # This is the slow part — BERT processes each log one by one
    print(f"\n STEP 2: Generating BERT embeddings...")
    print(f"   Converting {len(df)} logs to 768-dimensional vectors...")
    print(f"   (This takes 2-4 minutes — BERT is thinking hard!)\n")

    embeddings = []
    labels = []

    for i, row in df.iterrows():
        log_text = row['log_message']
        label = row['label']

        # Convert this log to 768 numbers
        embedding = get_bert_embedding(log_text)
        embeddings.append(embedding)
        labels.append(label)

        # Progress update every 10 logs
        if (i + 1) % 10 == 0:
            pct = int(((i + 1) / len(df)) * 100)
            bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
            print(f"   [{bar}] {pct}% — {i+1}/{len(df)} logs processed")

    print(f"\n   All embeddings generated!")

    # Convert lists to numpy arrays
    # X shape: (80, 768) — 80 logs, each with 768 numbers
    # y shape: (80,)     — 80 labels
    X = np.array(embeddings)
    y = np.array(labels)
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")

    # ── STEP 4: Split into train and test sets ──
    # 80% for training, 20% for testing
    # stratify=y means each category is proportionally split
    print(f"\n STEP 3: Splitting data (80% train / 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,       # 20% = 16 logs for testing
        random_state=42,     # fixed seed = reproducible results
        stratify=y           # keeps category proportions balanced
    )
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples:  {len(X_test)}")

    # ── STEP 5: Train Logistic Regression ──
    print(f"\n STEP 4: Training Logistic Regression model...")
    print(f"   (This is fast — the hard work was the embeddings!)")

    clf = LogisticRegression(
        max_iter=1000,    # max optimization steps
        C=1.0,            # regularization (1.0 = balanced)
        solver='lbfgs',   # optimization algorithm
        random_state=42
    )
    clf.fit(X_train, y_train)
    print(f"   Model trained!")

    # ── STEP 6: Evaluate on test set ──
    print(f"\n STEP 5: Evaluating model on test data...")
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n   Overall Accuracy: {accuracy:.0%}")
    print(f"\n   Per-category breakdown:")
    print(classification_report(y_test, y_pred))

    # ── STEP 7: Save the trained model ──
    print(f"\n STEP 6: Saving model...")
    os.makedirs("models", exist_ok=True)
    model_path = "models/log_classifier.pkl"
    joblib.dump(clf, model_path)

    size_kb = os.path.getsize(model_path) / 1024
    print(f"   Saved to: {model_path}")
    print(f"   File size: {size_kb:.1f} KB")

    print(f"\n Training complete!")
    print(f" Your model is ready at: models/log_classifier.pkl")
    print(f" Now run Phase 5 to build the FastAPI backend!")


if __name__ == "__main__":
    train()