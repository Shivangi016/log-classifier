# app/main.py
# This is your FastAPI backend server.
# Run it with: uvicorn app.main:app --reload

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import sys
import os

sys.path.insert(0, '.')
from app.utils import *               # loads .env
from app.classifier import classify_log

# ── Create the FastAPI app ──
app = FastAPI(
    title="Log Classifier API",
    description="Hybrid log classification: REGEX → BERT+ML → LLM",
    version="1.0.0"
)

# ── CORS Middleware ──
# This allows your Streamlit frontend (on a different port)
# to talk to this FastAPI backend without being blocked.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # allow all origins (fine for local dev)
    allow_methods=["*"],      # allow GET, POST, etc.
    allow_headers=["*"],
)


# ─────────────────────────────────────────
# REQUEST / RESPONSE MODELS
# Pydantic models define the exact shape of
# data coming IN and going OUT of your API.
# FastAPI validates them automatically.
# ─────────────────────────────────────────

class LogRequest(BaseModel):
    """Shape of a single classify request."""
    log: str

    class Config:
        json_schema_extra = {
            "example": {
                "log": "GET /api/users HTTP/1.1 404 Not Found"
            }
        }

class BatchLogRequest(BaseModel):
    """Shape of a batch classify request."""
    logs: List[str]

    class Config:
        json_schema_extra = {
            "example": {
                "logs": [
                    "GET /api/users HTTP/1.1 404 Not Found",
                    "NullPointerException at Main.java:42",
                    "User admin logged in successfully"
                ]
            }
        }

class ClassificationResult(BaseModel):
    """Shape of every classification response."""
    log: str
    label: str
    confidence: float
    method: str
    reason: str


# ─────────────────────────────────────────
# ROUTES
# Each route is a URL endpoint your frontend
# can call. Think of them as API functions.
# ─────────────────────────────────────────

@app.get("/")
def root():
    """Health check — visit this to confirm API is running."""
    return {
        "status": "running",
        "message": "Log Classifier API is live!",
        "docs": "Visit /docs to test the API interactively"
    }


@app.get("/health")
def health():
    """Simple health check for deployment platforms."""
    return {"status": "healthy"}


@app.post("/classify", response_model=ClassificationResult)
def classify_single(request: LogRequest):
    """
    Classify a single log message.

    Tries REGEX → ML → LLM in order.
    Returns label, confidence, method used, and reason.
    """
    # Validate input
    if not request.log.strip():
        raise HTTPException(
            status_code=400,
            detail="Log message cannot be empty"
        )

    if len(request.log) > 2000:
        raise HTTPException(
            status_code=400,
            detail="Log message too long (max 2000 characters)"
        )

    # Run the hybrid classifier
    result = classify_log(request.log)
    return result


@app.post("/classify/batch", response_model=List[ClassificationResult])
def classify_batch(request: BatchLogRequest):
    """
    Classify multiple log messages at once.
    Maximum 50 logs per request.
    """
    if not request.logs:
        raise HTTPException(
            status_code=400,
            detail="Logs list cannot be empty"
        )

    if len(request.logs) > 50:
        raise HTTPException(
            status_code=400,
            detail="Maximum 50 logs per batch request"
        )

    # Classify each log and collect results
    results = []
    for log in request.logs:
        if log.strip():                    # skip empty lines
            result = classify_log(log)
            results.append(result)

    return results


@app.get("/categories")
def get_categories():
    """
    Returns all supported log categories with descriptions.
    Useful for the frontend to show labels dynamically.
    """
    return {
        "categories": [
            {"label": "HTTP_SUCCESS",      "icon": "green",  "description": "Successful HTTP requests (2xx)"},
            {"label": "HTTP_ERROR",        "icon": "red",    "description": "Failed HTTP requests (4xx, 5xx)"},
            {"label": "APPLICATION_ERROR", "icon": "red",    "description": "Code exceptions and crashes"},
            {"label": "DATABASE_ERROR",    "icon": "orange", "description": "Database connection/query failures"},
            {"label": "AUTH_SUCCESS",      "icon": "green",  "description": "Successful logins and authentication"},
            {"label": "AUTH_FAILURE",      "icon": "red",    "description": "Failed logins, expired tokens"},
            {"label": "SYSTEM_WARNING",    "icon": "yellow", "description": "Resource warnings (CPU, memory, disk)"},
            {"label": "SYSTEM_INFO",       "icon": "blue",   "description": "General informational messages"},
        ]
    }