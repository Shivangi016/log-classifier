# app/classifier.py

import re
import os
import joblib
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────
# LAYER 1: REGEX
# Works like a lookup table of known patterns.
# If a log matches a pattern → instant answer.
# ─────────────────────────────────────────

REGEX_PATTERNS = [
    (re.compile(r'(GET|POST|PUT|DELETE|PATCH).*\s(200|201|202|204)\s', re.IGNORECASE), "HTTP_SUCCESS"),
    (re.compile(r'(GET|POST|PUT|DELETE|PATCH).*(404|500|503|403|401|400)', re.IGNORECASE), "HTTP_ERROR"),
    (re.compile(r'HTTP [45]\d{2}', re.IGNORECASE), "HTTP_ERROR"),
    (re.compile(r'(NullPointerException|ArrayIndexOutOfBoundsException|StackOverflowError|OutOfMemoryError|ClassNotFoundException|RuntimeException|IllegalArgumentException|NumberFormatException|IndexOutOfBoundsException)', re.IGNORECASE), "APPLICATION_ERROR"),
    (re.compile(r'(Connection refused|Timeout expired|pool exhausted|server has gone away|deadlock detected|disk quota exceeded|Replication lag|Failed to connect to|does not exist|connection timeout)', re.IGNORECASE), "DATABASE_ERROR"),
    (re.compile(r'(logged in successfully|authenticated successfully|key validated successfully|authentication passed|login successful|logged in successfully|password changed successfully)', re.IGNORECASE), "AUTH_SUCCESS"),
    (re.compile(r'(Failed login|token expired|Rate limit exceeded|Account locked|Invalid API key|token validation failed|reset link expired|Unauthorized access|signature verification failed)', re.IGNORECASE), "AUTH_FAILURE"),
    (re.compile(r'(CPU usage|Memory usage|Disk space.*low|certificate expires|network latency|file descriptors|Swap memory|Load average|Thread pool.*exhausted|exception rate spiked)', re.IGNORECASE), "SYSTEM_WARNING"),
    (re.compile(r'(started successfully|loaded.*successfully|completed successfully|Cache cleared|migration.*applied|restarted successfully|Health check passed|Deployment.*completed|Log rotation completed|joined the cluster)', re.IGNORECASE), "SYSTEM_INFO"),
]

def classify_with_regex(log: str):
    """
    Tries each pattern one by one.
    Returns (label, 1.0) if matched, or (None, 0.0) if not.
    Confidence is always 1.0 — regex is either right or doesn't match.
    """
    for pattern, label in REGEX_PATTERNS:
        if pattern.search(log):
            return label, 1.0
    return None, 0.0


# ─────────────────────────────────────────
# LAYER 2: BERT + LOGISTIC REGRESSION
# BERT = converts text to 768 numbers
# Logistic Regression = classifies those numbers
# ─────────────────────────────────────────

# We load BERT once here (slow to load, fast to use after)
# Using a try/except so the app doesn't crash if torch isn't available
try:
    from transformers import AutoTokenizer, AutoModel
    import torch

    BERT_MODEL_NAME = "distilbert-base-uncased"
    print("Loading BERT tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    bert_model = AutoModel.from_pretrained(BERT_MODEL_NAME)
    bert_model.eval()  # evaluation mode = no gradient tracking
    BERT_AVAILABLE = True
    print("BERT loaded successfully!")
except Exception as e:
    print(f"BERT not available: {e}")
    BERT_AVAILABLE = False


def get_bert_embedding(text: str) -> np.ndarray:
    """
    Converts a text string into 768 numbers using BERT.
    These numbers capture the MEANING of the text.

    Example:
      "Connection refused" → [0.23, -0.11, 0.87, ...]  (768 numbers)
      "Database timeout"   → [0.21, -0.09, 0.84, ...]  (similar! both DB errors)
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",    # PyTorch format
        truncation=True,        # cut if too long
        max_length=128,         # max 128 tokens
        padding="max_length"    # pad if too short
    )

    with torch.no_grad():       # don't track gradients (saves memory)
        outputs = bert_model(**inputs)

    # The [CLS] token at position 0 summarises the whole sentence
    # Shape: [1, 128, 768] → we take [0, 0, :] = first batch, CLS token, all 768 dims
    embedding = outputs.last_hidden_state[0, 0, :].numpy()
    return embedding


def classify_with_ml(log: str, model_path: str = "models/log_classifier.pkl"):
    """
    Uses the saved Logistic Regression model to classify.
    Returns (label, confidence) or (None, 0.0) if model not ready.
    """
    if not BERT_AVAILABLE:
        return None, 0.0

    if not os.path.exists(model_path):
        return None, 0.0  # Model hasn't been trained yet

    clf = joblib.load(model_path)
    embedding = get_bert_embedding(log).reshape(1, -1)  # shape: [1, 768]

    label = clf.predict(embedding)[0]
    # predict_proba returns probabilities for all classes
    # .max() gives us the highest one (the model's best guess confidence)
    confidence = float(clf.predict_proba(embedding).max())

    return label, confidence


# ─────────────────────────────────────────
# LAYER 3: LLM via Groq
# When REGEX and ML both fail or aren't confident,
# we ask a powerful LLM to classify the log.
# ─────────────────────────────────────────

def classify_with_llm(log: str):
    """
    Sends the log to Deepseek (via Groq Cloud) for classification.
    Groq is free and very fast — perfect for this project.
    Returns (label, confidence, reason).
    """
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key or api_key == "your_groq_api_key_here":
        return "UNKNOWN", 0.0, "No Groq API key configured"

    try:
        from groq import Groq
        client = Groq(api_key=api_key)

        prompt = f"""You are an expert log analysis system. Classify the following log message into EXACTLY one of these categories:

HTTP_SUCCESS    - Successful HTTP requests (2xx status codes)
HTTP_ERROR      - Failed HTTP requests (4xx, 5xx status codes)
APPLICATION_ERROR - Code exceptions, crashes, runtime errors
DATABASE_ERROR  - Database connection or query failures
AUTH_SUCCESS    - Successful logins or authentication
AUTH_FAILURE    - Failed logins, expired tokens, rate limits
SYSTEM_WARNING  - Resource warnings (CPU, memory, disk)
SYSTEM_INFO     - General informational messages
UNKNOWN         - Cannot be classified

Log: "{log}"

Respond in EXACTLY this format (no extra text):
LABEL: <category>
CONFIDENCE: <decimal between 0.0 and 1.0>
REASON: <one short sentence>"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,   # Low = more consistent answers
            max_tokens=100
        )

        content = response.choices[0].message.content.strip()

        # Parse the structured response
        label = "UNKNOWN"
        confidence = 0.5
        reason = "Classified by LLM"

        for line in content.split('\n'):
            line = line.strip()
            if line.startswith("LABEL:"):
                label = line.replace("LABEL:", "").strip()
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.replace("CONFIDENCE:", "").strip())
                except ValueError:
                    confidence = 0.5
            elif line.startswith("REASON:"):
                reason = line.replace("REASON:", "").strip()

        return label, confidence, reason

    except Exception as e:
        return "UNKNOWN", 0.0, f"LLM error: {str(e)}"


# ─────────────────────────────────────────
# HYBRID ORCHESTRATOR
# Tries all 3 layers in order.
# Stops at the first confident answer.
# ─────────────────────────────────────────

def classify_log(log: str, ml_threshold: float = 0.80):
    """
    Master function that runs the full pipeline.

    Flow:
      1. Try REGEX    → if match, return immediately
      2. Try ML       → if confidence >= 80%, return
      3. Try LLM      → always returns something

    Returns a dict with all details about the classification.
    """

    # ── Step 1: REGEX ──
    label, confidence = classify_with_regex(log)
    if label:
        return {
            "log": log,
            "label": label,
            "confidence": confidence,
            "method": "REGEX",
            "reason": "Matched a known log pattern"
        }

    # ── Step 2: BERT + ML ──
    label, confidence = classify_with_ml(log)
    if label and confidence >= ml_threshold:
        return {
            "log": log,
            "label": label,
            "confidence": confidence,
            "method": "ML (BERT + LR)",
            "reason": f"ML model confidence: {confidence:.0%}"
        }

    # ── Step 3: LLM ──
    label, confidence, reason = classify_with_llm(log)
    return {
        "log": log,
        "label": label,
        "confidence": confidence,
        "method": "LLM (Deepseek via Groq)",
        "reason": reason
    }