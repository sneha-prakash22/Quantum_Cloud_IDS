"""
Quantum-Inspired Machine Learning for Cyber Intrusion Detection
Professional Streamlit research dashboard.

Run from the project root:
    streamlit run dashboard.py

Primary dashboard model:
    models/extra_trees_20_features_model.pkl
    models/extra_trees_20_features_label_encoder.pkl

Required dataset and evaluation assets:
    data/balanced_ids_dataset.csv
    results/extra_trees_20_features_metrics.json
    results/extra_trees_78_features_metrics.json
    results/extra_trees_20_features_feature_importance.csv
    results/extra_trees_78_features_feature_importance.csv
    results/extra_trees_20_features_confusion_matrix.csv
    results/extra_trees_78_features_confusion_matrix.csv
    results/mininet_validation/
"""

from pathlib import Path
import json
import warnings
import hashlib
import urllib.request

import joblib
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore", category=FutureWarning)

# -------------------------------------------------------------------
# PAGE + PROJECT PATHS
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Quantum Cloud IDS | Security Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "balanced_ids_dataset.csv"
MODEL_URL = "https://github.com/sneha-prakash22/Quantum_Cloud_IDS/releases/download/v1.0.0/extra_trees_20_features_model.pkl"
MODEL_SHA256 = "C2D3466333A7D1C0826C109774E60FE779677173EE4E7CECFBDC27A0E19262F5"
MODEL_PATH = BASE_DIR / "models" / "extra_trees_20_features_model.pkl"
ENCODER_PATH = BASE_DIR / "models" / "extra_trees_20_features_label_encoder.pkl"
METRICS_PATH = BASE_DIR / "results" / "extra_trees_20_features_metrics.json"
BASELINE_METRICS_PATH = BASE_DIR / "results" / "extra_trees_78_features_metrics.json"
IMPORTANCE_20_PATH = BASE_DIR / "results" / "extra_trees_20_features_feature_importance.csv"
IMPORTANCE_78_PATH = BASE_DIR / "results" / "extra_trees_78_features_feature_importance.csv"
CONFUSION_20_PATH = BASE_DIR / "results" / "extra_trees_20_features_confusion_matrix.csv"
CONFUSION_78_PATH = BASE_DIR / "results" / "extra_trees_78_features_confusion_matrix.csv"
MININET_DIR = BASE_DIR / "results" / "mininet_validation"
MININET_FLOWS_PATH = MININET_DIR / "mininet_benign_tcp_paced_001_flows.csv"
MININET_PREDICTIONS_PATH = MININET_DIR / "mininet_benign_tcp_paced_001_predictions.csv"
MININET_MANIFEST_PATH = MININET_DIR / "mininet_validation_sha256_manifest.csv"
HISTORY_PATH = BASE_DIR / "results" / "quantum_feature_selection_history.csv"

# -------------------------------------------------------------------
# DESIGN SYSTEM
# -------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --bg: #080d18;
        --panel: #101827;
        --panel-2: #121e30;
        --line: rgba(148, 163, 184, .16);
        --text: #e8eef8;
        --muted: #91a2ba;
        --cyan: #4de2d0;
        --blue: #6a9cff;
        --green: #5ee0a0;
        --amber: #ffc56d;
        --red: #ff7185;
    }

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp {
        background:
            radial-gradient(ellipse at 8% 0%, rgba(39, 91, 180, .15), transparent 32%),
            radial-gradient(ellipse at 94% 7%, rgba(20, 184, 166, .09), transparent 26%),
            var(--bg);
        color: var(--text);
    }
    [data-testid="stHeader"] { background: rgba(8,13,24,.75); }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0c1423 0%, #0a101c 100%);
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] > div { padding-top: 1.15rem; }
    [data-testid="stSidebar"] * { color: #dce6f5; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        border-radius: 10px;
        padding: .35rem .55rem;
    }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -.035em; }
    h1 { font-weight: 700 !important; }
    h2, h3 { color: #edf4ff; }
    p, li { color: #c1cde0; }
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(19,31,49,.96), rgba(13,22,36,.96));
        border: 1px solid var(--line);
        padding: 17px 18px;
        border-radius: 16px;
        min-height: 112px;
        box-shadow: 0 10px 28px rgba(0,0,0,.12);
    }
    [data-testid="stMetricLabel"] { color: #9fb0c8 !important; font-size: .82rem; }
    [data-testid="stMetricValue"] { color: #f4f8ff !important; font-family: 'Space Grotesk', sans-serif; }
    [data-testid="stMetricDelta"] { font-size: .78rem; }
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--line) !important;
        border-radius: 16px !important;
    }
    div[data-testid="stAlert"] { border-radius: 12px; }
    div.stButton > button, div.stDownloadButton > button,
    div.stFormSubmitButton > button {
        border-radius: 11px;
        min-height: 2.65rem;
        font-weight: 700;
        border: 1px solid rgba(77,226,208,.35);
        transition: all .18s ease;
    }
    div.stButton > button:hover, div.stDownloadButton > button:hover,
    div.stFormSubmitButton > button:hover {
        border-color: var(--cyan);
        transform: translateY(-1px);
        box-shadow: 0 8px 22px rgba(77,226,208,.10);
    }
    [data-testid="stDataFrame"], [data-testid="stTable"] {
        border: 1px solid var(--line);
        border-radius: 12px;
        overflow: hidden;
    }
    [data-baseweb="tab-list"] { gap: 8px; }
    [data-baseweb="tab"] { border-radius: 10px 10px 0 0; }
    hr { border-color: var(--line); }
    .brand {
        border: 1px solid rgba(106,156,255,.25);
        border-radius: 17px;
        padding: 17px 15px;
        background: linear-gradient(135deg, rgba(39,91,180,.24), rgba(20,184,166,.08));
        margin: 0 0 18px 0;
    }
    .brand-icon { font-size: 1.5rem; }
    .brand-name { font: 700 1.08rem 'Space Grotesk',sans-serif; margin-top: 7px; color: #fff; }
    .brand-sub { color: #9fb0c8; font-size: .78rem; margin-top: 4px; line-height: 1.45; }
    .eyebrow {
        color: var(--cyan); text-transform: uppercase; letter-spacing: .16em;
        font-size: .72rem; font-weight: 700; margin-bottom: 9px;
    }
    .hero {
        padding: 27px 30px; border-radius: 22px; margin: 4px 0 24px;
        border: 1px solid rgba(106,156,255,.24);
        background: linear-gradient(115deg, rgba(17,29,48,.98), rgba(18,43,72,.97) 62%, rgba(7,92,103,.72));
        box-shadow: 0 18px 44px rgba(0,0,0,.18);
        position: relative; overflow: hidden;
    }
    .hero:after {
        content: ""; position: absolute; width: 220px; height: 220px; right: -65px; top: -110px;
        border-radius: 50%; border: 28px solid rgba(77,226,208,.10);
    }
    .hero h1 { color: #fff; font-size: clamp(1.65rem, 3vw, 2.45rem); line-height: 1.16; margin: 0 0 11px; }
    .hero p { color: #c6d7ed; max-width: 820px; line-height: 1.6; margin: 0; }
    .pill-row { display:flex; flex-wrap:wrap; gap:8px; margin-top:18px; }
    .pill { border:1px solid rgba(191,219,254,.2); background:rgba(4,13,27,.3); color:#d8eaff;
        border-radius:999px; padding:6px 10px; font-size:.75rem; }
    .section-kicker { color: var(--muted); font-size:.82rem; margin-top:-7px; margin-bottom:12px; }
    .panel {
        border: 1px solid var(--line); background: rgba(16,24,39,.78);
        padding: 17px 18px; border-radius: 15px; height: 100%;
    }
    .panel-label { color:#94a8c2; font-size:.77rem; text-transform:uppercase; letter-spacing:.09em; }
    .panel-value { color:#f4f8ff; font:700 1.45rem 'Space Grotesk',sans-serif; margin-top:8px; }
    .panel-note { color:#9aabc2; font-size:.8rem; margin-top:5px; }
    .status-dot { display:inline-block; width:8px; height:8px; border-radius:50%; background:var(--green); margin-right:7px; }
    .footer { text-align:center; color:#71839c; font-size:.77rem; padding:20px 0 5px; }
    .muted { color: #91a2ba; }

    /* ---------------------------------------------------------------
       LIGHT THEME OVERRIDES
       --------------------------------------------------------------- */
    :root {
        --bg: #ffffff;
        --panel: #ffffff;
        --panel-2: #f7f9fc;
        --line: #e2e8f0;
        --text: #172b4d;
        --muted: #64748b;
        --cyan: #0891b2;
        --blue: #2563eb;
        --green: #16a34a;
        --amber: #d97706;
        --red: #dc2626;
    }

    .stApp,
    [data-testid="stAppViewContainer"] {
        background: #ffffff !important;
        color: #172b4d !important;
    }

    [data-testid="stHeader"] {
        background: rgba(255,255,255,.96) !important;
    }

    [data-testid="stSidebar"] {
        background: #f5f7fb !important;
        border-right: 1px solid #e2e8f0 !important;
    }

    [data-testid="stSidebar"] * {
        color: #26364b;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        color: #26364b !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background: #e8eef8;
    }

    h1, h2, h3, h4, h5, h6,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3 {
        color: #172b4d !important;
    }

    p, li, [data-testid="stMarkdownContainer"] {
        color: #334155;
    }

    label, .stCaption, [data-testid="stCaptionContainer"] {
        color: #64748b !important;
    }

    [data-testid="stMetric"] {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 3px 12px rgba(15,23,42,.045) !important;
    }

    [data-testid="stMetricLabel"] {
        color: #64748b !important;
    }

    [data-testid="stMetricValue"] {
        color: #172b4d !important;
    }

    [data-testid="stMetricDelta"] {
        color: #475569 !important;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #e2e8f0 !important;
        background: #ffffff !important;
    }

    .brand {
        background: linear-gradient(135deg, #ffffff, #edf5ff) !important;
        border: 1px solid #dbeafe !important;
    }

    .brand-name {
        color: #172b4d !important;
    }

    .brand-sub {
        color: #64748b !important;
    }

    .eyebrow {
        color: #087e9b !important;
    }

    .hero {
        background: linear-gradient(115deg, #ffffff, #f0f6ff 72%, #e9fafa) !important;
        border: 1px solid #dbeafe !important;
        box-shadow: 0 8px 26px rgba(15,23,42,.055) !important;
    }

    .hero h1 {
        color: #172b4d !important;
    }

    .hero p {
        color: #475569 !important;
    }

    .pill {
        background: #ffffff !important;
        border: 1px solid #dbeafe !important;
        color: #1e40af !important;
    }

    .section-kicker {
        color: #64748b !important;
    }

    .panel {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 3px 12px rgba(15,23,42,.035);
    }

    .panel-label {
        color: #64748b !important;
    }

    .panel-value {
        color: #172b4d !important;
    }

    .panel-note, .muted {
        color: #64748b !important;
    }

    .footer {
        color: #64748b !important;
    }

    div.stButton > button,
    div.stDownloadButton > button,
    div.stFormSubmitButton > button {
        background: #2563eb !important;
        color: #ffffff !important;
        border: 1px solid #2563eb !important;
    }

    div.stButton > button:hover,
    div.stDownloadButton > button:hover,
    div.stFormSubmitButton > button:hover {
        background: #1d4ed8 !important;
        border-color: #1d4ed8 !important;
        box-shadow: 0 5px 14px rgba(37,99,235,.16) !important;
    }

    input, textarea,
    [data-baseweb="select"] > div,
    [data-baseweb="input"] > div {
        background: #ffffff !important;
        color: #172b4d !important;
        border-color: #cbd5e1 !important;
    }

    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
        border: 1px solid #e2e8f0 !important;
        background: #ffffff !important;
    }

    [data-baseweb="tab"] {
        color: #475569 !important;
    }

    [data-baseweb="tab"][aria-selected="true"] {
        color: #1d4ed8 !important;
    }

    hr {
        border-color: #e2e8f0 !important;
    }

    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# DATA ACCESS
# -------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained model…")

def calculate_sha256(file_path):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def ensure_model_available():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Check whether a valid model already exists
    if MODEL_PATH.exists():
        existing_hash = calculate_sha256(MODEL_PATH)

        if existing_hash.lower() == MODEL_SHA256.lower():
            return MODEL_PATH

    # Download the model from GitHub
    temporary_path = MODEL_PATH.with_suffix(
        MODEL_PATH.suffix + ".part"
    )

    try:
        request = urllib.request.Request(
            MODEL_URL,
            headers={"User-Agent": "Quantum-Cloud-IDS"}
        )

        sha256 = hashlib.sha256()

        with urllib.request.urlopen(request, timeout=120) as response:
            with open(temporary_path, "wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)

                    if not chunk:
                        break

                    output.write(chunk)
                    sha256.update(chunk)

        # Verify downloaded model
        downloaded_hash = sha256.hexdigest()

        if downloaded_hash.lower() != MODEL_SHA256.lower():
            raise ValueError("Model SHA-256 verification failed!")

        # Save verified model
        temporary_path.replace(MODEL_PATH)

        return MODEL_PATH

    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()

        raise

def load_model(path_string: str):
    path = Path(path_string)
    if not path.exists():
        return None
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def load_json(path_string: str):
    path = Path(path_string)
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
        return result if isinstance(result, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


@st.cache_data(show_spinner=False)
def load_features(path_string: str):
    path = Path(path_string)
    if not path.exists():
        return []
    try:
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except OSError:
        return []


@st.cache_data(show_spinner="Loading CIC-IDS2017 dataset…")
def load_dataset(path_string: str):
    path = Path(path_string)
    if not path.exists():
        return pd.DataFrame()
    try:
        frame = pd.read_csv(path, low_memory=False)
        frame.columns = frame.columns.astype(str).str.strip()
        if "Label" in frame.columns:
            frame["Label"] = frame["Label"].astype(str).str.strip()
        return frame
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        return pd.DataFrame()



model = None
model_error = None

try:
    verified_model_path = ensure_model_available()
    model = load_model(str(verified_model_path))

except Exception as exc:
    model_error = str(exc)


metrics = load_json(str(METRICS_PATH))
baseline_metrics = load_json(str(BASELINE_METRICS_PATH))
selected_features = metrics.get("selected_features", [])
comparison = {}
label_encoder = None
encoder_error = None
try:
    label_encoder = load_model(str(ENCODER_PATH))
except Exception as exc:
    encoder_error = str(exc)
dataset = load_dataset(str(DATA_PATH))

# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------
def metric_value(key, default=None):
    value = metrics.get(key, default)
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def pct(value, digits=2):
    if value is None or not np.isfinite(value):
        return "N/A"
    # Most saved metrics are fractions; support percentage values too.
    normalized = value * 100 if abs(value) <= 1 else value
    return f"{normalized:.{digits}f}%"


def get_prediction_frame(frame: pd.DataFrame):
    """Clean selected feature columns and return usable rows + original indices."""
    missing = [col for col in selected_features if col not in frame.columns]
    if missing:
        raise ValueError("Required model features are missing: " + ", ".join(missing))
    clean = frame[selected_features].copy()
    for col in selected_features:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")
    clean = clean.replace([np.inf, -np.inf], np.nan)
    valid_mask = clean.notna().all(axis=1)
    return clean.loc[valid_mask], valid_mask


def safe_predict_proba(estimator, frame):
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(frame)
    return None


def render_hero(title, description, tags):
    pills = "".join(f'<span class="pill">{tag}</span>' for tag in tags)
    st.markdown(
        f"""
        <div class="hero">
          <div class="eyebrow">Security analytics / Research environment</div>
          <h1>{title}</h1>
          <p>{description}</p>
          <div class="pill-row">{pills}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_heading(title, subtitle):
    st.title(title)
    st.markdown(f'<div class="section-kicker">{subtitle}</div>', unsafe_allow_html=True)


def show_missing_assets():
    missing = []
    if model is None:
        missing.append(str(MODEL_PATH.relative_to(BASE_DIR)))
    if label_encoder is None:
        missing.append(str(ENCODER_PATH.relative_to(BASE_DIR)))
    if not selected_features:
        missing.append(str(METRICS_PATH.relative_to(BASE_DIR)) + " (selected_features)")
    if missing:
        st.warning("Some required project assets could not be loaded: " + ", ".join(missing))
        st.caption("Keep dashboard.py in the project root and confirm the paths above. The dashboard will still show available dataset and result information.")
    if model_error:
        st.error("Model loading error: " + model_error)
    if encoder_error:
        st.error("Label encoder loading error: " + encoder_error)


def download_csv(frame, label, filename, key):
    st.download_button(
        label=label,
        data=frame.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
        key=key,
    )


def class_label(value):
    text = str(value).strip()
    return "BENIGN" if text.upper() == "BENIGN" else text


# -------------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="brand">
          <div class="brand-icon">🛡️</div>
          <div class="brand-name">QUANTUM <span style="color:#4de2d0">CLOUD IDS</span></div>
          <div class="brand-sub">Enterprise intrusion detection<br>Research & model analytics</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    page = st.radio(
        "WORKSPACE",
        [
            "Security Overview",
            "Detection Lab",
            "Batch Analysis",
            "Feature Intelligence",
            "Model Evaluation",
            "Dataset Explorer",
            "Mininet Validation",
        ],
        label_visibility="visible",
    )
    st.divider()
    st.markdown("**SYSTEM STATUS**")
    if model is not None and label_encoder is not None and selected_features:
        st.markdown('<span class="status-dot"></span> Model assets loaded', unsafe_allow_html=True)
    else:
        st.markdown("🟠 Model assets need attention")
    st.caption(f"Model: {MODEL_PATH.name}")
    st.caption(f"Features: {len(selected_features) if selected_features else 'not loaded'}")
    st.divider()
    st.caption("Research prototype · Offline dataset analysis")
    st.caption("Not connected to a live network sensor.")

# -------------------------------------------------------------------
# TOP-LEVEL HERO
# -------------------------------------------------------------------
render_hero(
    "Quantum-Inspired Cyber Intrusion Detection",
    "A focused security analytics workspace for exploring CIC-IDS2017 traffic, "
    "testing a trained multiclass classifier, and examining the impact of "
    "quantum-inspired feature selection on enterprise-cloud intrusion detection.",
    [
        "⚛ Quantum-inspired feature search",
        "🧠 Extra Trees classifier",
        "📊 CIC-IDS2017",
        f"🧩 {len(selected_features)} selected features" if selected_features else "🧩 Feature list unavailable",
    ],
)
show_missing_assets()

# -------------------------------------------------------------------
# SECURITY OVERVIEW
# -------------------------------------------------------------------
if page == "Security Overview":
    page_heading("Security overview", "A high-level view of the model, dataset, and saved evaluation results.")

    total_records = len(dataset)
    class_count = dataset["Label"].nunique() if not dataset.empty and "Label" in dataset.columns else 0
    attack_records = 0
    if not dataset.empty and "Label" in dataset.columns:
        attack_records = int((dataset["Label"].astype(str).str.upper() != "BENIGN").sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset records", f"{total_records:,}" if total_records else "N/A")
    c2.metric("Traffic classes", f"{class_count:,}" if class_count else "N/A")
    c3.metric("Selected features", str(len(selected_features)) if selected_features else "N/A")
    c4.metric("Feature reduction", "74.36%" if selected_features else "N/A", help="Calculated from 78 original features reduced to 20 selected features: (78 - 20) / 78 * 100.")

    st.markdown("### Model evaluation snapshot")
    a, b, c, d = st.columns(4)
    a.metric("Test accuracy", pct(metric_value("accuracy")))
    b.metric("Macro F1-score", f"{metric_value('macro_f1'):.4f}" if metric_value("macro_f1") is not None else "N/A")
    d.metric("Weighted F1-score", f"{metric_value('weighted_f1'):.4f}" if metric_value("weighted_f1") is not None else "N/A")
    d2 = "Saved evaluation artifact" if metrics else "Metrics file not found"
    c.metric("Evaluation data", "Available" if metrics else "Missing", help=d2)

    left, right = st.columns([1.35, 1])
    with left:
        st.markdown("### Traffic class landscape")
        if not dataset.empty and "Label" in dataset.columns:
            counts = dataset["Label"].value_counts().rename_axis("Traffic class").to_frame("Records")
            st.bar_chart(counts, use_container_width=True)
            st.caption("Distribution reflects the loaded CSV file, not live traffic.")
        else:
            st.info("Traffic class distribution is unavailable until the dataset CSV is present.")
    with right:
        st.markdown("### Detection pipeline")
        st.markdown(
            """
            <div class="panel">
              <div class="panel-label">01 · Ingest</div><div class="panel-value">CIC-IDS2017</div>
              <div class="panel-note">Load and inspect labelled network-flow records.</div>
              <hr>
              <div class="panel-label">02 · Reduce</div><div class="panel-value">Feature search</div>
              <div class="panel-note">Use the saved quantum-inspired selected feature subset.</div>
              <hr>
              <div class="panel-label">03 · Classify</div><div class="panel-value">Extra Trees</div>
              <div class="panel-note">Predict a traffic class from the selected input features.</div>
              <hr>
              <div class="panel-label">04 · Evaluate</div><div class="panel-value">Metrics & errors</div>
              <div class="panel-note">Review saved metrics and inspect batch prediction mistakes.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.info("Quantum-inspired refers to the classical optimization approach used for feature selection. This dashboard does not claim to execute on quantum hardware or monitor a live enterprise network.")

# -------------------------------------------------------------------
# DETECTION LAB
# -------------------------------------------------------------------
elif page == "Detection Lab":
    page_heading("Detection lab", "Submit one network-flow feature vector to the saved model.")
    if model is None or not selected_features:
        st.error("Detection is unavailable. Confirm the trained model and selected-feature file are present.")
    else:
        st.markdown("#### Input network-flow features")
        st.caption("Enter values using the same feature definitions, units, and preprocessing expected during model training.")
        with st.form("single_detection_form", border=True):
            cols = st.columns(2)
            values = []
            for idx, feature in enumerate(selected_features):
                with cols[idx % 2]:
                    values.append(
                        st.number_input(
                            feature,
                            value=0.0,
                            format="%.6f",
                            key=f"single_feature_{idx}",
                            help="Enter a numeric value for this model feature.",
                        )
                    )
            submitted = st.form_submit_button("Run intrusion classification", type="primary", use_container_width=True)

        if submitted:
            input_df = pd.DataFrame([values], columns=selected_features)
            try:
                prediction_id = int(model.predict(input_df)[0])
                predicted_label = str(label_encoder.inverse_transform([prediction_id])[0])
                probability = safe_predict_proba(model, input_df)
                confidence = float(np.max(probability[0])) if probability is not None else None
                st.divider()
                st.markdown("### Classification result")
                if predicted_label.strip().upper() == "BENIGN":
                    st.success(f"BENIGN traffic predicted · {predicted_label}")
                else:
                    st.error(f"Potential intrusion class predicted · {predicted_label}")
                r1, r2, r3 = st.columns(3)
                r1.metric("Predicted class", predicted_label)
                r2.metric("Model confidence", f"{confidence * 100:.2f}%" if confidence is not None else "Not available")
                r3.metric("Input features", len(selected_features))
                if probability is not None and hasattr(model, "classes_"):
                    class_ids = np.asarray(model.classes_, dtype=int)
                    class_names = label_encoder.inverse_transform(class_ids)
                    prob_df = pd.DataFrame({
                        "Traffic class": class_names,
                        "Probability (%)": probability[0] * 100,
                    }).sort_values("Probability (%)", ascending=False)
                    st.markdown("### Class probability profile")
                    st.bar_chart(prob_df.set_index("Traffic class"), use_container_width=True)
                    st.dataframe(prob_df.round(3), use_container_width=True, hide_index=True)
                else:
                    st.info("This model does not expose predict_proba, so a probability profile cannot be displayed.")
                st.markdown("### Submitted feature vector")
                st.dataframe(input_df, use_container_width=True, hide_index=True)
                st.caption("Model confidence is a predicted probability, not a guarantee of correctness. This manual form is a demonstration; use realistic, correctly scaled values for meaningful output.")
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

# -------------------------------------------------------------------
# BATCH ANALYSIS
# -------------------------------------------------------------------
elif page == "Batch Analysis":
    page_heading("Batch threat analysis", "Run the classifier on labelled records from the project dataset and review errors transparently.")
    if dataset.empty:
        st.error(f"Dataset not found or could not be read: {DATA_PATH}")
    elif model is None or label_encoder is None or not selected_features:
        st.error("Batch prediction requires the trained model, label encoder, and selected-feature list.")
    elif "Label" not in dataset.columns:
        st.error("The dataset does not contain the expected 'Label' column.")
    else:
        with st.container(border=True):
            x1, x2, x3 = st.columns([1, 1, 1])
            max_records = min(5000, len(dataset))
            record_count = x1.slider("Records to evaluate", 1, max(1, max_records), min(500, max_records))
            sample_mode = x2.selectbox("Record selection", ["Random sample", "First records"], index=0)
            only_errors = x3.checkbox("Show misclassified rows only", value=False)
            seed = st.number_input("Random seed (for reproducible sampling)", min_value=0, max_value=999999, value=42, step=1)
            run_batch = st.button("Run batch analysis", type="primary", use_container_width=True)

        if run_batch:
            with st.spinner("Preparing records and running predictions…"):
                try:
                    working = dataset.copy()
                    if sample_mode == "Random sample":
                        working = working.sample(n=min(record_count, len(working)), random_state=int(seed))
                    else:
                        working = working.head(record_count)

                    valid_X, valid_mask = get_prediction_frame(working)
                    valid_rows = working.loc[valid_X.index].copy()
                    valid_rows["Actual Label"] = valid_rows["Label"].astype(str).str.strip()
                    if valid_X.empty:
                        st.error("No usable rows remained after numeric and missing-value checks.")
                    else:
                        predicted = model.predict(valid_X)
                        predicted_labels = label_encoder.inverse_transform(np.asarray(predicted, dtype=int))
                        results = pd.DataFrame({
                            "Dataset index": valid_X.index,
                            "Actual Label": valid_rows["Actual Label"].to_numpy(),
                            "Predicted Label": predicted_labels.astype(str),
                        })
                        results["Correct"] = results["Actual Label"].str.casefold() == results["Predicted Label"].str.casefold()
                        total = len(results)
                        correct = int(results["Correct"].sum())
                        errors = total - correct
                        accuracy = correct / total * 100 if total else 0.0
                        st.session_state["batch_results"] = results
                        st.session_state["batch_accuracy"] = accuracy
                        st.session_state["batch_total"] = total
                        st.session_state["batch_errors"] = errors
                        st.session_state["batch_input_rows"] = len(working)
                        st.session_state["batch_dropped"] = len(working) - len(valid_X)
                except Exception as exc:
                    st.error(f"Batch analysis failed: {exc}")

        if "batch_results" in st.session_state:
            results = st.session_state["batch_results"]
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Rows evaluated", f"{st.session_state['batch_total']:,}")
            k2.metric("Correct predictions", f"{st.session_state['batch_total'] - st.session_state['batch_errors']:,}")
            k3.metric("Misclassified", f"{st.session_state['batch_errors']:,}")
            k4.metric("Sample accuracy", f"{st.session_state['batch_accuracy']:.2f}%")
            dropped = st.session_state.get("batch_dropped", 0)
            if dropped:
                st.caption(f"{dropped:,} selected rows were excluded because selected features contained missing, infinite, or non-numeric values.")
            if st.session_state["batch_errors"]:
                st.warning("Misclassifications are retained for inspection. Sample accuracy applies only to this selected sample and is not the saved held-out test metric.")
            else:
                st.success("No misclassifications occurred in this sample. This does not guarantee performance on unseen traffic.")
            show_df = results.loc[~results["Correct"]] if only_errors else results
            st.markdown("### Prediction results")
            st.dataframe(show_df, use_container_width=True, hide_index=True)
            if not results.empty:
                with st.expander("Per-class summary"):
                    class_summary = results.groupby("Actual Label", dropna=False).agg(
                        Records=("Correct", "size"),
                        Correct=("Correct", "sum"),
                    ).reset_index()
                    class_summary["Incorrect"] = class_summary["Records"] - class_summary["Correct"]
                    class_summary["Accuracy (%)"] = (class_summary["Correct"] / class_summary["Records"] * 100).round(2)
                    st.dataframe(class_summary, use_container_width=True, hide_index=True)
            download_csv(results, "Download all batch predictions (CSV)", "quantum_cloud_ids_batch_predictions.csv", "download_batch_predictions")

# -------------------------------------------------------------------
# FEATURE INTELLIGENCE
# -------------------------------------------------------------------
elif page == "Feature Intelligence":
    page_heading("Feature intelligence", "Inspect the selected feature subset and the saved optimizer search history.")
    if selected_features:
        f1, f2, f3 = st.columns(3)
        f1.metric("Original feature count", "78")
        f2.metric("Selected feature count", len(selected_features))
        reduction = (1 - len(selected_features) / 78) * 100
        f3.metric("Calculated reduction", f"{reduction:.2f}%")
        feature_df = pd.DataFrame({"Rank": range(1, len(selected_features) + 1), "Selected feature": selected_features})
        st.markdown("### Selected feature set")
        st.dataframe(feature_df, use_container_width=True, hide_index=True)
        download_csv(feature_df, "Export selected features", "selected_model_features.csv", "download_features")
    else:
        st.warning("Selected features were not loaded. Check the feature file path.")

    st.markdown("### Extra Trees feature importance")
    if IMPORTANCE_20_PATH.exists() and selected_features:
        try:
            saved_importance = pd.read_csv(IMPORTANCE_20_PATH)
            required_columns = {"feature", "importance"}
            if not required_columns.issubset(saved_importance.columns):
                st.warning("The saved Extra Trees feature-importance file has an unexpected column structure.")
            elif set(saved_importance["feature"].astype(str)) != set(map(str, selected_features)):
                st.warning("The saved Extra Trees feature-importance records do not match the selected 20-feature list.")
            else:
                importance_df = saved_importance[["feature", "importance"]].copy()
                importance_df.columns = ["Feature", "Importance"]
                importance_df = importance_df.sort_values("Importance", ascending=False)
                st.bar_chart(importance_df.set_index("Feature"), use_container_width=True)
                st.dataframe(importance_df.round(6), use_container_width=True, hide_index=True)
                download_csv(importance_df, "Export Extra Trees feature importance", "extra_trees_20_feature_importance.csv", "download_importance_20")
        except (OSError, pd.errors.ParserError, KeyError, ValueError) as exc:
            st.error(f"Could not read saved Extra Trees feature importance: {exc}")
    else:
        st.info("The saved Extra Trees feature-importance file or selected feature list is unavailable.")

    st.markdown("### Optimizer search history")
    if HISTORY_PATH.exists():
        try:
            history = pd.read_csv(HISTORY_PATH)
            if history.empty:
                st.info("The search history file exists but contains no rows.")
            else:
                if "best_macro_f1_so_far" in history.columns:
                    st.markdown("**Best macro F1 found during search**")
                    st.line_chart(history["best_macro_f1_so_far"], use_container_width=True)
                if "macro_f1_mean" in history.columns:
                    st.markdown("**Candidate cross-validation macro F1**")
                    st.line_chart(history["macro_f1_mean"], use_container_width=True)
                st.dataframe(history, use_container_width=True, hide_index=True)
                download_csv(history, "Download optimizer history", "quantum_feature_selection_history.csv", "download_history")
        except (OSError, pd.errors.ParserError) as exc:
            st.error(f"Could not read optimizer history: {exc}")
    else:
        st.info("Search history has not been generated yet. The dashboard will show it automatically when the CSV is available.")
    st.caption("These values describe the fitted 20-feature Extra Trees classifier's feature importance. They are not quantum-inspired selection probabilities or causal explanations. The quantum-inspired search used a Random Forest candidate evaluator with 3-fold cross-validation; Extra Trees is the final classifier.")

# -------------------------------------------------------------------
# MODEL EVALUATION
# -------------------------------------------------------------------
elif page == "Model Evaluation":
    page_heading("Model evaluation","Compare saved experiment results. ")
    if baseline_metrics and metrics:
        st.markdown("### Primary experiment comparison")
        st.caption("Both entries are Extra Trees classifiers evaluated using the saved held-out test results. The comparison isolates the effect of reducing the input feature set from 78 to 20.")

        comparison_df = pd.DataFrame([
            {
                "Model": "Extra Trees - 78 features",
                "Features": baseline_metrics["features"],
                "Accuracy (%)": float(baseline_metrics["accuracy"]) * 100,
                "Macro F1 (%)": float(baseline_metrics["macro_f1"]) * 100,
                "Weighted F1 (%)": float(baseline_metrics["weighted_f1"]) * 100,
                "Training time (s)": float(baseline_metrics["training_time_seconds"]),
            },
            {
                "Model": "Extra Trees - 20 features",
                "Features": metrics["features"],
                "Accuracy (%)": float(metrics["accuracy"]) * 100,
                "Macro F1 (%)": float(metrics["macro_f1"]) * 100,
                "Weighted F1 (%)": float(metrics["weighted_f1"]) * 100,
                "Training time (s)": float(metrics["training_time_seconds"]),
            },
        ])

        st.dataframe(comparison_df.round(4), use_container_width=True, hide_index=True)
        download_csv(comparison_df, "Export Extra Trees comparison", "extra_trees_78_vs_20_comparison.csv", "download_extra_trees_comparison")

        accuracy_delta = (float(metrics["accuracy"]) - float(baseline_metrics["accuracy"])) * 100
        macro_delta = (float(metrics["macro_f1"]) - float(baseline_metrics["macro_f1"])) * 100
        weighted_delta = (float(metrics["weighted_f1"]) - float(baseline_metrics["weighted_f1"])) * 100
        feature_reduction = (1 - float(metrics["features"]) / float(baseline_metrics["features"])) * 100
        training_time_reduction = (1 - float(metrics["training_time_seconds"]) / float(baseline_metrics["training_time_seconds"])) * 100

        st.markdown("### Change from the 78-feature baseline")
        d1, d2, d3, d4, d5 = st.columns(5)
        d1.metric("Accuracy change", f"{accuracy_delta:+.4f} pp")
        d2.metric("Macro F1 change", f"{macro_delta:+.4f} pp")
        d3.metric("Weighted F1 change", f"{weighted_delta:+.4f} pp")
        d4.metric("Feature reduction", f"{feature_reduction:.2f}%")
        d5.metric("Training-time reduction", f"{training_time_reduction:.2f}%")

        st.markdown("### Held-out test metric comparison")
        chart_tab1, chart_tab2, chart_tab3 = st.tabs(["Accuracy", "Macro F1", "Weighted F1"])
        with chart_tab1:
            st.bar_chart(comparison_df.set_index("Model")[["Accuracy (%)"]], use_container_width=True)
        with chart_tab2:
            st.bar_chart(comparison_df.set_index("Model")[["Macro F1 (%)"]], use_container_width=True)
        with chart_tab3:
            st.bar_chart(comparison_df.set_index("Model")[["Weighted F1 (%)"]], use_container_width=True)

        st.markdown("### Saved confusion matrices")
        matrix_tab1, matrix_tab2 = st.tabs(["Extra Trees - 78 features", "Extra Trees - 20 features"])
        for matrix_tab, matrix_path in [
            (matrix_tab1, CONFUSION_78_PATH),
            (matrix_tab2, CONFUSION_20_PATH),
        ]:
            with matrix_tab:
                if matrix_path.exists():
                    try:
                        matrix_df = pd.read_csv(matrix_path, index_col=0)
                        st.dataframe(matrix_df, use_container_width=True)
                    except (OSError, pd.errors.ParserError, ValueError) as exc:
                        st.error(f"Could not read confusion matrix: {exc}")
                else:
                    st.info(f"Confusion matrix file is unavailable: {matrix_path.name}")

        st.caption("The reported changes are differences between the two saved Extra Trees experiments. Accuracy and F1 changes are percentage-point differences, not relative percentage gains. The 20-feature model shows a modest metric improvement alongside fewer features and lower recorded training time.")
    else:
        st.warning("The saved Extra Trees 78-feature baseline or 20-feature metrics are unavailable. Check the two configured metrics files.")

    if metrics:
        st.markdown("### Saved test metrics")
        metric_df = pd.DataFrame([{"Metric": key.replace("_", " ").title(), "Value": value} for key, value in metrics.items()])
        st.dataframe(metric_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download saved metrics (JSON)",
            data=json.dumps(metrics, indent=2).encode("utf-8"),
            file_name="quantum_multiclass_test_metrics.json",
            mime="application/json",
            use_container_width=True,
        )
    else:
        st.info("Saved test metrics are not available yet.")
    #st.info("Use the saved held-out test metrics for formal reporting. Batch sample accuracy is a separate, sample-specific diagnostic.")

# -------------------------------------------------------------------
# DATASET EXPLORER
# -------------------------------------------------------------------
elif page == "Dataset Explorer":
    page_heading("Dataset explorer", "Explore the prepared CIC-IDS2017 dataset, filter records, and export a sample.")
    if dataset.empty:
        st.error(f"Dataset file was not found or could not be read: {DATA_PATH}")
    else:
        duplicate_count = int(dataset.duplicated().sum())
        st.caption(
            f"Source: {DATA_PATH.name}. The prepared dataset retains all {len(dataset):,} records, "
            f"including {duplicate_count:,} duplicate rows. Class counts are not perfectly equal."
        )
        label_classes = dataset["Label"].nunique() if "Label" in dataset.columns else 0
        a, b, c, d = st.columns(4)
        a.metric("Records", f"{len(dataset):,}")
        b.metric("Columns", f"{len(dataset.columns):,}")
        c.metric("Traffic classes", f"{label_classes:,}" if label_classes else "N/A")
        missing_cells = int(dataset.isna().sum().sum())
        d.metric("Missing cells", f"{missing_cells:,}")

        if "Label" in dataset.columns:
            st.markdown("### Traffic class distribution")
            distribution = dataset["Label"].value_counts().rename_axis("Traffic class").to_frame("Records")
            st.bar_chart(distribution, use_container_width=True)

        st.markdown("### Filter and inspect records")
        filtered = dataset
        if "Label" in dataset.columns:
            classes = sorted(dataset["Label"].dropna().astype(str).unique().tolist())
            selected_classes = st.multiselect("Traffic classes", classes, default=classes, help="Choose one or more labels to filter the table.")
            filtered = filtered[filtered["Label"].astype(str).isin(selected_classes)]
        search_text = st.text_input("Optional text search", placeholder="Search visible string values…")
        if search_text.strip():
            string_cols = filtered.select_dtypes(include=["object", "string"]).columns
            if len(string_cols):
                mask = filtered[string_cols].astype(str).apply(
                    lambda col: col.str.contains(search_text, case=False, na=False, regex=False)
                ).any(axis=1)
                filtered = filtered[mask]

        max_rows = min(1000, max(5, len(filtered)))
        display_count = st.slider("Rows to display", min_value=5, max_value=max_rows, value=min(25, max_rows))
        st.caption(f"Showing {min(display_count, len(filtered)):,} of {len(filtered):,} filtered rows.")
        st.dataframe(filtered.head(display_count), use_container_width=True, hide_index=True)
        download_csv(filtered.head(display_count), "Download displayed records (CSV)", "cic_ids2017_filtered_sample.csv", "download_dataset_sample")

        with st.expander("Dataset quality snapshot"):
            quality = pd.DataFrame({
                "Column": dataset.columns,
                "Data type": [str(dtype) for dtype in dataset.dtypes],
                "Missing values": [int(dataset[col].isna().sum()) for col in dataset.columns],
                "Unique values": [int(dataset[col].nunique(dropna=True)) for col in dataset.columns],
            })
            st.dataframe(quality, use_container_width=True, hide_index=True)

# -------------------------------------------------------------------
# MININET VALIDATION
# -------------------------------------------------------------------
elif page == "Mininet Validation":
    page_heading(
        "Mininet validation",
        "Review the saved evidence from the controlled benign TCP-flow experiment."
    )

    st.warning(
        "Offline experimental evidence only. This page displays the saved prediction "
        "record; it does not rerun the classifier or monitor a live network."
    )

    required_files = [
        MININET_FLOWS_PATH,
        MININET_PREDICTIONS_PATH,
        MININET_MANIFEST_PATH,
    ]
    missing_files = [str(file.relative_to(BASE_DIR)) for file in required_files if not file.exists()]

    if missing_files:
        st.error("Required Mininet evidence files are missing: " + ", ".join(missing_files))
    else:
        try:
            mininet_flows = pd.read_csv(MININET_FLOWS_PATH)
            mininet_predictions = pd.read_csv(MININET_PREDICTIONS_PATH)
            mininet_manifest = pd.read_csv(MININET_MANIFEST_PATH)

            prediction_columns = {
                "Ground Truth",
                "Predicted Label",
                "Prediction Confidence",
                "Flow ID",
            }
            flow_columns = {
                "Flow ID",
                "Source IP",
                "Source Port",
                "Destination IP",
                "Destination Port",
                "Protocol",
                "Total Fwd Packets",
                "Total Backward Packets",
                "Total Length of Fwd Packets",
            }
            manifest_columns = {"File", "SizeBytes", "SHA256"}

            if not prediction_columns.issubset(mininet_predictions.columns):
                st.error("The saved Mininet prediction file has an unexpected column structure.")
            elif not flow_columns.issubset(mininet_flows.columns):
                st.error("The saved Mininet flow file has an unexpected column structure.")
            elif not manifest_columns.issubset(mininet_manifest.columns):
                st.error("The Mininet SHA256 manifest has an unexpected column structure.")
            elif mininet_predictions.empty or mininet_flows.empty:
                st.warning("The saved Mininet flow or prediction file contains no records.")
            else:
                saved_prediction = mininet_predictions.iloc[0]
                confidence = float(saved_prediction["Prediction Confidence"]) * 100

                st.markdown("### Recorded experiment summary")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Captured flow records", f"{len(mininet_flows):,}")
                m2.metric("Saved prediction records", f"{len(mininet_predictions):,}")
                m3.metric("Ground truth", str(saved_prediction["Ground Truth"]))
                m4.metric("Prediction confidence", f"{confidence:.1f}%")

                st.markdown("### Saved prediction evidence")
                st.dataframe(
                    mininet_predictions[
                        [
                            "Flow ID",
                            "Source IP",
                            "Source Port",
                            "Destination IP",
                            "Destination Port",
                            "Ground Truth",
                            "Predicted Label",
                            "Prediction Confidence",
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

                st.caption(
                    "The prediction and confidence above are read directly from the saved "
                    "experiment CSV. They are not newly generated by this dashboard."
                )

                st.markdown("### Captured flow details")
                visible_flow_columns = [
                    "Flow ID",
                    "Source IP",
                    "Source Port",
                    "Destination IP",
                    "Destination Port",
                    "Protocol",
                    "Total Fwd Packets",
                    "Total Backward Packets",
                    "Total Length of Fwd Packets",
                ]
                st.dataframe(
                    mininet_flows[visible_flow_columns],
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption(
                    "The captured experiment used a controlled TCP transfer with 500,000 "
                    "forward payload bytes. The flow record is presented as captured evidence."
                )

                st.markdown("### Evidence integrity manifest")
                st.dataframe(
                    mininet_manifest,
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption(
                    f"The saved manifest contains {len(mininet_manifest):,} file records "
                    "with their recorded sizes and SHA256 hashes."
                )

                st.markdown("### Experimental scope and limitations")
                st.warning(
                    "This experiment contains one controlled benign TCP flow only. "
                    "It does not test attack traffic, establish statistically meaningful "
                    "independent accuracy, or demonstrate model generalization."
                )
                st.markdown(
                    "- Ground truth and prediction are shown as recorded in the saved evidence.\n"
                    "- A single matching benign prediction is not an independent accuracy evaluation.\n"
                    "- No live network monitoring or continuous detection is performed.\n"
                    "- The extracted flow uses `Fwd Header Length`, whereas the selected model "
                    "feature is `Fwd Header Length.1`; therefore this page does not rerun "
                    "the model against the flow CSV."
                )

        except (OSError, pd.errors.ParserError, KeyError, ValueError, TypeError) as exc:
            st.error(f"Could not read or display the saved Mininet evidence: {exc}")

# -------------------------------------------------------------------
# FOOTER
# -------------------------------------------------------------------
st.divider()
st.markdown(
    """
    <div class="footer">
      <strong>Quantum-Inspired Machine Learning for Cyber Intrusion Detection in Enterprise Cloud Platforms</strong><br>
      Research prototype · Offline dataset analysis · Not a live network sensor
    </div>
    """,
    unsafe_allow_html=True,
)
