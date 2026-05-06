# streamlit_app.py
# Run with: streamlit run streamlit_app.py

import streamlit as st
import requests
import pandas as pd

# ── Page configuration ──
st.set_page_config(
    page_title="Log Classifier",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = "http://localhost:8000"

# ── Label styling ──
LABEL_ICONS = {
    "HTTP_SUCCESS":      "🟢",
    "HTTP_ERROR":        "🔴",
    "APPLICATION_ERROR": "🔴",
    "DATABASE_ERROR":    "🟠",
    "AUTH_SUCCESS":      "🟢",
    "AUTH_FAILURE":      "🔴",
    "SYSTEM_WARNING":    "🟡",
    "SYSTEM_INFO":       "🔵",
    "UNKNOWN":           "⚪"
}

LABEL_COLORS = {
    "HTTP_SUCCESS":      "#e8f5e9",
    "HTTP_ERROR":        "#ffebee",
    "APPLICATION_ERROR": "#ffebee",
    "DATABASE_ERROR":    "#fff3e0",
    "AUTH_SUCCESS":      "#e8f5e9",
    "AUTH_FAILURE":      "#ffebee",
    "SYSTEM_WARNING":    "#fffde7",
    "SYSTEM_INFO":       "#e3f2fd",
    "UNKNOWN":           "#f5f5f5"
}

METHOD_INFO = {
    "REGEX":                 ("⚡", "#e8f5e9", "Instant pattern match"),
    "ML (BERT + LR)":        ("🧠", "#e3f2fd", "Machine learning model"),
    "LLM (Deepseek via Groq)":("🤖", "#f3e5f5", "AI language model"),
}

# ── Helper: call the API ──
def call_classify(log: str):
    try:
        r = requests.post(
            f"{API_URL}/classify",
            json={"log": log},
            timeout=30
        )
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "connection_error"
    except Exception as e:
        return None, str(e)

def call_batch(logs: list):
    try:
        r = requests.post(
            f"{API_URL}/classify/batch",
            json={"logs": logs},
            timeout=60
        )
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "connection_error"
    except Exception as e:
        return None, str(e)
st.success("✅ System ready")

    st.markdown("---")
    st.markdown("### How it works")
    st.markdown("""
**Layer 1 — REGEX** ⚡
Pattern matching. Instant, 100% confident.

**Layer 2 — BERT + ML** 🧠
Converts log to numbers, ML classifies.

**Layer 3 — LLM** 🤖
AI handles complex/rare logs.
    """)

    st.markdown("---")
    st.markdown("### Categories")
    for label, icon in LABEL_ICONS.items():
        if label != "UNKNOWN":
            st.markdown(f"{icon} `{label}`")

# ── Main content ──
st.title("Log Classification System")
st.markdown("*Hybrid classifier: REGEX → BERT+ML → LLM*")
st.markdown("---")

tab1, tab2, tab3 = st.tabs([
    "🔍 Single Log",
    "📋 Batch Classification",
    "📊 About & Examples"
])

# ════════════════════════════════
# TAB 1 — Single Log
# ════════════════════════════════
with tab1:
    st.header("Classify a Single Log")

    # Example logs
    EXAMPLES = {
        "Select an example...": "",
        "HTTP 404 error":           "GET /api/users HTTP/1.1 404 Not Found",
        "HTTP success":             "POST /api/login HTTP/1.1 200 OK",
        "Java exception":           "NullPointerException at com.app.Service.process(Service.java:42)",
        "Database error":           "Connection refused: database host unreachable",
        "Successful login":         "User admin logged in successfully from 192.168.1.10",
        "Failed login":             "Failed login attempt for user admin from 192.168.1.99",
        "CPU warning":              "CPU usage at 95% for the last 5 minutes on server web-01",
        "App started":              "Application started successfully on port 8080",
        "Ambiguous log (→ LLM)":    "Unusual memory pattern detected in worker process after batch job",
    }

    selected = st.selectbox("Try a quick example:", list(EXAMPLES.keys()))

    default_text = EXAMPLES[selected]
    log_input = st.text_area(
        "Log message:",
        value=default_text,
        height=80,
        placeholder="Paste any log message here..."
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        classify_btn = st.button("🚀 Classify", type="primary", use_container_width=True)
    with col2:
        if st.button("🗑️ Clear", use_container_width=False):
            st.rerun()

    if classify_btn:
        if not log_input.strip():
            st.warning("⚠️ Please enter a log message.")
        else:
            with st.spinner("Classifying..."):
                result, error = call_classify(log_input)

            if error == "connection_error":
                st.error("❌ Cannot connect to API. Is `uvicorn app.main:app --reload` running?")
            elif error:
                st.error(f"❌ Error: {error}")
            else:
                # ── Result display ──
                st.markdown("---")
                st.subheader("Result")

                icon = LABEL_ICONS.get(result['label'], '⚪')
                bg   = LABEL_COLORS.get(result['label'], '#f5f5f5')
                m_icon, m_bg, m_desc = METHOD_INFO.get(
                    result['method'],
                    ("🤖", "#f5f5f5", result['method'])
                )

                # Three metric columns
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(
                        f"<div style='background:{bg};padding:16px;border-radius:10px;text-align:center'>"
                        f"<div style='font-size:13px;color:#666'>Label</div>"
                        f"<div style='font-size:22px;font-weight:600'>{icon} {result['label']}</div>"
                        f"</div>", unsafe_allow_html=True
                    )
                with c2:
                    conf_pct = int(result['confidence'] * 100)
                    conf_color = "#e8f5e9" if conf_pct >= 80 else "#fff3e0" if conf_pct >= 50 else "#ffebee"
                    st.markdown(
                        f"<div style='background:{conf_color};padding:16px;border-radius:10px;text-align:center'>"
                        f"<div style='font-size:13px;color:#666'>Confidence</div>"
                        f"<div style='font-size:22px;font-weight:600'>{conf_pct}%</div>"
                        f"</div>", unsafe_allow_html=True
                    )
                with c3:
                    st.markdown(
                        f"<div style='background:{m_bg};padding:16px;border-radius:10px;text-align:center'>"
                        f"<div style='font-size:13px;color:#666'>Method</div>"
                        f"<div style='font-size:18px;font-weight:600'>{m_icon} {result['method']}</div>"
                        f"</div>", unsafe_allow_html=True
                    )

                # Reason box
                st.markdown(
                    f"<div style='background:#f8f9fa;border-left:4px solid #667;padding:12px 16px;"
                    f"border-radius:0 8px 8px 0;margin-top:12px'>"
                    f"<b>Reason:</b> {result['reason']}</div>",
                    unsafe_allow_html=True
                )


# ════════════════════════════════
# TAB 2 — Batch Classification
# ════════════════════════════════
with tab2:
    st.header("Batch Classification")
    st.markdown("Enter one log per line — up to 50 logs at once.")

    BATCH_EXAMPLE = """GET /index.html HTTP/1.1 200 OK
POST /api/login HTTP/1.1 500 Internal Server Error
NullPointerException at com.app.Main.run(Main.java:15)
Connection refused: database host unreachable
User john_doe logged in successfully from 10.0.0.5
Failed login attempt for user unknown_user: invalid credentials
CPU usage at 95% for the last 5 minutes on server web-01
Application started successfully on port 8080"""

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("Load example logs"):
            st.session_state["batch_text"] = BATCH_EXAMPLE

    batch_text = st.text_area(
        "Logs (one per line):",
        value=st.session_state.get("batch_text", ""),
        height=220,
        placeholder="Paste your logs here, one per line..."
    )

    logs_list = [l.strip() for l in batch_text.split('\n') if l.strip()]
    st.caption(f"{len(logs_list)} log(s) entered")

    if st.button("🚀 Classify All", type="primary"):
        if not logs_list:
            st.warning("⚠️ Please enter at least one log.")
        elif len(logs_list) > 50:
            st.warning("⚠️ Maximum 50 logs. Please reduce the number.")
        else:
            with st.spinner(f"Classifying {len(logs_list)} logs..."):
                results, error = call_batch(logs_list)

            if error == "connection_error":
                st.error("❌ Cannot connect to API.")
            elif error:
                st.error(f"❌ Error: {error}")
            else:
                st.markdown("---")
                st.subheader(f"Results — {len(results)} logs classified")

                # Build display DataFrame
                rows = []
                for r in results:
                    icon = LABEL_ICONS.get(r['label'], '⚪')
                    m_icon = METHOD_INFO.get(r['method'], ("🤖","",""))[0]
                    rows.append({
                        "Log":        r['log'][:65] + "..." if len(r['log']) > 65 else r['log'],
                        "Label":      f"{icon} {r['label']}",
                        "Confidence": f"{int(r['confidence']*100)}%",
                        "Method":     f"{m_icon} {r['method']}",
                    })

                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True, hide_index=True)

                # Summary charts side by side
                st.markdown("---")
                st.subheader("Summary")
                c1, c2 = st.columns(2)

                raw_df = pd.DataFrame(results)
                with c1:
                    st.markdown("**By label**")
                    st.bar_chart(raw_df['label'].value_counts())
                with c2:
                    st.markdown("**By method used**")
                    st.bar_chart(raw_df['method'].value_counts())


# ════════════════════════════════
# TAB 3 — About
# ════════════════════════════════
with tab3:
    st.header("About This Project")

    st.markdown("""
    ### Architecture

    This system uses a **3-layer hybrid approach**:

    | Layer | Technology | Speed | Used When |
    |-------|-----------|-------|-----------|
    | 1 — REGEX | Pattern matching | ⚡ Instant | Known patterns (HTTP codes, exceptions) |
    | 2 — BERT + LR | Transformers + Scikit-learn | 🔄 Fast | Variable patterns with training data |
    | 3 — LLM | Deepseek via Groq | 🤖 Slower | Complex or ambiguous logs |

    ### Tech Stack
    - **Backend:** FastAPI + Python
    - **Frontend:** Streamlit
    - **ML Model:** DistilBERT embeddings + Logistic Regression
    - **LLM:** Deepseek-R1 via Groq Cloud (free tier)
    - **Training data:** 80 labeled log samples across 8 categories
    """)

    st.markdown("---")
    st.subheader("Test Logs to Try")

    test_logs = {
        "Will use REGEX ⚡":     [
            "GET /dashboard HTTP/1.1 404 Not Found",
            "POST /api/auth HTTP/1.1 200 OK",
            "NullPointerException at Main.java:10",
            "User root logged in successfully",
        ],
        "Will use ML 🧠":        [
            "Database query took 45 seconds to complete",
            "Worker process recycled after memory threshold",
            "Retry attempt 3 of 5 for payment gateway",
        ],
        "Will use LLM 🤖":       [
            "Anomalous packet sequence detected on interface eth0",
            "Service mesh sidecar reported unexpected handshake",
            "Batch processor stalled after checkpoint mismatch",
        ],
    }

    for section, logs in test_logs.items():
        st.markdown(f"**{section}**")
        for log in logs:
            st.code(log, language=None)