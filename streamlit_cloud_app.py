import streamlit as st
import re
import os
import numpy as np
import pandas as pd

st.set_page_config(page_title="Log Classifier", page_icon="🔍", layout="wide")

LABEL_ICONS = {
    "HTTP_SUCCESS": "🟢", "HTTP_ERROR": "🔴",
    "APPLICATION_ERROR": "🔴", "DATABASE_ERROR": "🟠",
    "AUTH_SUCCESS": "🟢", "AUTH_FAILURE": "🔴",
    "SYSTEM_WARNING": "🟡", "SYSTEM_INFO": "🔵", "UNKNOWN": "⚪"
}
LABEL_COLORS = {
    "HTTP_SUCCESS": "#e8f5e9", "HTTP_ERROR": "#ffebee",
    "APPLICATION_ERROR": "#ffebee", "DATABASE_ERROR": "#fff3e0",
    "AUTH_SUCCESS": "#e8f5e9", "AUTH_FAILURE": "#ffebee",
    "SYSTEM_WARNING": "#fffde7", "SYSTEM_INFO": "#e3f2fd", "UNKNOWN": "#f5f5f5"
}
METHOD_INFO = {
    "REGEX": ("⚡", "#e8f5e9"),
    "ML (BERT + LR)": ("🧠", "#e3f2fd"),
    "LLM (Llama via Groq)": ("🤖", "#f3e5f5"),
}

REGEX_PATTERNS = [
    (re.compile(r'(GET|POST|PUT|DELETE|PATCH).*\s(200|201|202|204)\s', re.IGNORECASE), "HTTP_SUCCESS"),
    (re.compile(r'(GET|POST|PUT|DELETE|PATCH).*(404|500|503|403|401|400)', re.IGNORECASE), "HTTP_ERROR"),
    (re.compile(r'HTTP [45]\d{2}', re.IGNORECASE), "HTTP_ERROR"),
    (re.compile(r'(NullPointerException|ArrayIndexOutOfBoundsException|StackOverflowError|OutOfMemoryError|ClassNotFoundException|RuntimeException|IllegalArgumentException|NumberFormatException|IndexOutOfBoundsException)', re.IGNORECASE), "APPLICATION_ERROR"),
    (re.compile(r'(Connection refused|Timeout expired|pool exhausted|server has gone away|deadlock detected|disk quota exceeded|Replication lag|Failed to connect to|does not exist|connection timeout)', re.IGNORECASE), "DATABASE_ERROR"),
    (re.compile(r'(logged in successfully|authenticated successfully|key validated successfully|authentication passed|login successful|password changed successfully)', re.IGNORECASE), "AUTH_SUCCESS"),
    (re.compile(r'(Failed login|token expired|Rate limit exceeded|Account locked|Invalid API key|token validation failed|reset link expired|Unauthorized access|signature verification failed)', re.IGNORECASE), "AUTH_FAILURE"),
    (re.compile(r'(CPU usage|Memory usage|Disk space.*low|certificate expires|network latency|file descriptors|Swap memory|Load average|Thread pool.*exhausted|exception rate spiked)', re.IGNORECASE), "SYSTEM_WARNING"),
    (re.compile(r'(started successfully|loaded.*successfully|completed successfully|Cache cleared|migration.*applied|restarted successfully|Health check passed|Deployment.*completed|Log rotation completed|joined the cluster)', re.IGNORECASE), "SYSTEM_INFO"),
]

def classify_with_regex(log):
    for pattern, label in REGEX_PATTERNS:
        if pattern.search(log):
            return label, 1.0
    return None, 0.0

@st.cache_resource
def load_models():
    try:
        from transformers import AutoTokenizer, AutoModel
        import torch
        import joblib
        tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        mdl = AutoModel.from_pretrained("distilbert-base-uncased")
        mdl.eval()
        clf = joblib.load("models/log_classifier.pkl") if os.path.exists("models/log_classifier.pkl") else None
        return tok, mdl, clf, True
    except Exception as e:
        return None, None, None, False

def get_embedding(text, tok, mdl):
    import torch
    inp = tok(text, return_tensors="pt", truncation=True, max_length=128, padding="max_length")
    with torch.no_grad():
        out = mdl(**inp)
    return out.last_hidden_state[0, 0, :].numpy()

def classify_with_ml(log, tok, mdl, clf):
    if clf is None or tok is None:
        return None, 0.0
    try:
        emb = get_embedding(log, tok, mdl).reshape(1, -1)
        return clf.predict(emb)[0], float(clf.predict_proba(emb).max())
    except:
        return None, 0.0

def classify_with_llm(log):
    try:
        try:
            key = st.secrets["GROQ_API_KEY"]
        except:
            key = os.getenv("GROQ_API_KEY")
        if not key:
            return "UNKNOWN", 0.0, "No API key configured"
        from groq import Groq
        client = Groq(api_key=key)
        prompt = f"""Classify this log into exactly one category:
HTTP_SUCCESS, HTTP_ERROR, APPLICATION_ERROR, DATABASE_ERROR,
AUTH_SUCCESS, AUTH_FAILURE, SYSTEM_WARNING, SYSTEM_INFO, UNKNOWN

Log: "{log}"

Reply ONLY in this format:
LABEL: <category>
CONFIDENCE: <0.0 to 1.0>
REASON: <one sentence>"""
        r = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=100
        )
        content = r.choices[0].message.content.strip()
        label, conf, reason = "UNKNOWN", 0.5, "Classified by LLM"
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith("LABEL:"):
                label = line.replace("LABEL:", "").strip()
            elif line.startswith("CONFIDENCE:"):
                try:
                    conf = float(line.replace("CONFIDENCE:", "").strip())
                except:
                    pass
            elif line.startswith("REASON:"):
                reason = line.replace("REASON:", "").strip()
        return label, conf, reason
    except Exception as e:
        return "UNKNOWN", 0.0, str(e)

def classify_log(log, tok, mdl, clf):
    label, conf = classify_with_regex(log)
    if label:
        return {"log": log, "label": label, "confidence": conf,
                "method": "REGEX", "reason": "Matched a known pattern"}
    label, conf = classify_with_ml(log, tok, mdl, clf)
    if label and conf >= 0.80:
        return {"log": log, "label": label, "confidence": conf,
                "method": "ML (BERT + LR)", "reason": f"ML confidence: {conf:.0%}"}
    label, conf, reason = classify_with_llm(log)
    return {"log": log, "label": label, "confidence": conf,
            "method": "LLM (Llama via Groq)", "reason": reason}

with st.spinner("Loading BERT model... (first load ~30s)"):
    tok, mdl, clf, bert_ok = load_models()

with st.sidebar:
    st.title("Log Classifier")
    st.markdown("---")
    if bert_ok:
        st.success("BERT model loaded")
    else:
        st.success("System ready (REGEX + LLM)")
    st.markdown("---")
    st.markdown("### How it works")
    st.markdown("""
**Layer 1 - REGEX**
Pattern matching. Instant.

**Layer 2 - BERT + ML**
Deep learning + Logistic Regression.

**Layer 3 - LLM**
Llama 3.3 via Groq for complex logs.
    """)
    st.markdown("---")
    st.markdown("### Categories")
    for label, icon in LABEL_ICONS.items():
        if label != "UNKNOWN":
            st.markdown(f"{icon} `{label}`")

st.title("Log Classification System")
st.markdown("*Hybrid classifier: REGEX -> BERT+ML -> LLM*")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["Single Log", "Batch Classification", "About"])

with tab1:
    st.header("Classify a Single Log")
    EXAMPLES = {
        "Select an example...": "",
        "HTTP 404 error": "GET /api/users HTTP/1.1 404 Not Found",
        "HTTP success": "POST /api/login HTTP/1.1 200 OK",
        "Java exception": "NullPointerException at com.app.Service.process(Service.java:42)",
        "Database error": "Connection refused: database host unreachable",
        "Successful login": "User admin logged in successfully from 192.168.1.10",
        "Failed login": "Failed login attempt for user admin from 192.168.1.99",
        "CPU warning": "CPU usage at 95% for the last 5 minutes on server web-01",
        "App started": "Application started successfully on port 8080",
        "Ambiguous (LLM)": "Anomalous packet sequence detected on interface eth0",
    }
    selected = st.selectbox("Try an example:", list(EXAMPLES.keys()))
    log_input = st.text_area("Log message:", value=EXAMPLES[selected],
                              height=80, placeholder="Paste any log message here...")
    if st.button("Classify", type="primary"):
        if not log_input.strip():
            st.warning("Please enter a log message.")
        else:
            with st.spinner("Classifying..."):
                result = classify_log(log_input, tok, mdl, clf)
            st.markdown("---")
            icon = LABEL_ICONS.get(result['label'], '?')
            bg = LABEL_COLORS.get(result['label'], '#f5f5f5')
            m_icon, m_bg = METHOD_INFO.get(result['method'], ("?", "#f5f5f5"))
            conf_pct = int(result['confidence'] * 100)
            conf_color = "#e8f5e9" if conf_pct >= 80 else "#fff3e0" if conf_pct >= 50 else "#ffebee"
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(
                    f"<div style='background:{bg};padding:16px;border-radius:10px;text-align:center'>"
                    f"<div style='font-size:13px;color:#666'>Label</div>"
                    f"<div style='font-size:20px;font-weight:600'>{icon} {result['label']}</div>"
                    f"</div>", unsafe_allow_html=True)
            with c2:
                st.markdown(
                    f"<div style='background:{conf_color};padding:16px;border-radius:10px;text-align:center'>"
                    f"<div style='font-size:13px;color:#666'>Confidence</div>"
                    f"<div style='font-size:20px;font-weight:600'>{conf_pct}%</div>"
                    f"</div>", unsafe_allow_html=True)
            with c3:
                st.markdown(
                    f"<div style='background:{m_bg};padding:16px;border-radius:10px;text-align:center'>"
                    f"<div style='font-size:13px;color:#666'>Method</div>"
                    f"<div style='font-size:18px;font-weight:600'>{m_icon} {result['method']}</div>"
                    f"</div>", unsafe_allow_html=True)
            st.markdown(
                f"<div style='background:#f8f9fa;border-left:4px solid #667;"
                f"padding:12px 16px;border-radius:0 8px 8px 0;margin-top:12px'>"
                f"<b>Reason:</b> {result['reason']}</div>", unsafe_allow_html=True)

with tab2:
    st.header("Batch Classification")
    st.markdown("One log per line, up to 50 logs.")
    BATCH_EXAMPLE = """GET /index.html HTTP/1.1 200 OK
POST /api/login HTTP/1.1 500 Internal Server Error
NullPointerException at com.app.Main.run(Main.java:15)
Connection refused: database host unreachable
User john_doe logged in successfully from 10.0.0.5
Failed login attempt for user unknown_user: invalid credentials
CPU usage at 95% for the last 5 minutes on server web-01
Application started successfully on port 8080"""
    if st.button("Load example logs"):
        st.session_state["batch_text"] = BATCH_EXAMPLE
    batch_text = st.text_area("Logs:", value=st.session_state.get("batch_text", ""),
                               height=200, placeholder="One log per line...")
    logs_list = [l.strip() for l in batch_text.split('\n') if l.strip()]
    st.caption(f"{len(logs_list)} log(s) entered")
    if st.button("Classify All", type="primary"):
        if not logs_list:
            st.warning("Please enter at least one log.")
        else:
            results = []
            progress = st.progress(0)
            for i, log in enumerate(logs_list):
                results.append(classify_log(log, tok, mdl, clf))
                progress.progress((i + 1) / len(logs_list))
            rows = []
            for r in results:
                icon = LABEL_ICONS.get(r['label'], '?')
                m_icon = METHOD_INFO.get(r['method'], ("?", ""))[0]
                rows.append({
                    "Log": r['log'][:60] + "..." if len(r['log']) > 60 else r['log'],
                    "Label": f"{icon} {r['label']}",
                    "Confidence": f"{int(r['confidence'] * 100)}%",
                    "Method": f"{m_icon} {r['method']}",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            raw_df = pd.DataFrame(results)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**By label**")
                st.bar_chart(raw_df['label'].value_counts())
            with c2:
                st.markdown("**By method**")
                st.bar_chart(raw_df['method'].value_counts())

with tab3:
    st.header("About")
    st.markdown("""
### Architecture
| Layer | Technology | Speed | Used When |
|-------|-----------|-------|-----------|
| REGEX | Pattern matching | Instant | Known patterns |
| BERT + LR | Transformers + Scikit-learn | Fast | Variable patterns |
| LLM | Llama 3.3 via Groq | Slower | Complex logs |

### Tech Stack
Python, Streamlit, HuggingFace Transformers, Scikit-learn, Groq API

### GitHub
github.com/Shivangi016/log-classifier
    """)