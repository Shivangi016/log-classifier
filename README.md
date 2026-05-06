\# 🔍 Log Classification System



A hybrid AI system that classifies application logs using 3 layers:



\*\*REGEX → BERT + ML → LLM\*\*



\## How It Works



| Layer | Technology | Used When |

|-------|-----------|-----------|

| ⚡ REGEX | Pattern matching | Known patterns (HTTP codes, exceptions) |

| 🧠 BERT + LR | DistilBERT + Scikit-learn | Variable patterns with training data |

| 🤖 LLM | Llama 3.3 via Groq | Complex or ambiguous logs |



\## Tech Stack

\- \*\*Backend:\*\* FastAPI + Python

\- \*\*Frontend:\*\* Streamlit

\- \*\*ML:\*\* DistilBERT embeddings + Logistic Regression

\- \*\*LLM:\*\* Llama 3.3-70b via Groq Cloud (free)



\## Log Categories

\- HTTP\_SUCCESS / HTTP\_ERROR

\- APPLICATION\_ERROR

\- DATABASE\_ERROR

\- AUTH\_SUCCESS / AUTH\_FAILURE

\- SYSTEM\_WARNING / SYSTEM\_INFO



\## Setup



```bash

\# Clone the repo

git clone https://github.com/Shivangi016/log-classifier.git

cd log-classifier



\# Create virtual environment

python -m venv venv

venv\\Scripts\\activate  # Windows



\# Install dependencies

pip install -r requirements.txt



\# Add your Groq API key

echo GROQ\_API\_KEY=your\_key\_here > .env



\# Train the ML model

python train.py



\# Start the backend (Terminal 1)

uvicorn app.main:app --reload



\# Start the frontend (Terminal 2)

streamlit run streamlit\_app.py

```



\## Project Structure

```

log-classifier/

├── app/

│   ├── classifier.py   # 3-layer hybrid classifier

│   ├── main.py         # FastAPI backend

│   └── utils.py        # Environment loader

├── data/

│   └── logs.csv        # 80 labeled training samples

├── streamlit\_app.py    # Streamlit frontend

└── train.py            # ML training script

```

