import os
import sys
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Ensure local src/ directory is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.ml_engine import (
    detect_identifier_columns,
    detect_task_type,
    separate_feature_types,
    find_semantic_columns,
    get_available_algorithms,
    train_supervised_models,
    predict_single_student,
    predict_batch_students,
    get_feature_importances,
    generate_academic_suggestions
)

# =========================================================
# 1. PAGE CONFIGURATION & CLEAN ACADEMIC STYLING
# =========================================================

st.set_page_config(
    page_title="Dynamic Student Performance Prediction System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional styling: White background, subtle indigo/blue accents, dark readable text
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #0B1120;
        --surface: #111A2E;
        --surface-2: #16213A;
        --border: #24304D;
        --text: #E2E8F0;
        --muted: #94A3B8;
        --primary: #818CF8;
        --primary-strong: #6366F1;
        --teal: #2DD4BF;
        --ok: #34D399;
        --warn: #FBBF24;
        --danger: #F87171;
    }

    /* ---------- Base ---------- */
    .stApp {
        background:
            radial-gradient(1100px 520px at 8% -8%, rgba(99, 102, 241, 0.16), transparent 60%),
            radial-gradient(900px 480px at 100% 0%, rgba(45, 212, 191, 0.08), transparent 55%),
            var(--bg);
    }
    .stApp, .stApp p, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3,
    .stApp h4, .stApp h5, .stApp h6, .stApp button, .stApp input, .stApp textarea {
        font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif;
    }
    .block-container { padding-top: 2.2rem; max-width: 1400px; }
    h2, h3, h4 { letter-spacing: -0.01em; }
    hr { border-color: var(--border) !important; }

    /* ---------- Header ---------- */
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .main-header .hdr-text {
        background: linear-gradient(90deg, #C7D2FE 0%, #818CF8 45%, #2DD4BF 100%);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        color: transparent;
    }
    .sub-header {
        font-size: 0.98rem;
        color: var(--muted);
        margin-bottom: 1.4rem;
        max-width: 900px;
        line-height: 1.55;
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0E1629 0%, #0B1120 100%);
        border-right: 1px solid var(--border);
    }

    /* ---------- Cards (bordered containers) ---------- */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--surface);
        border-color: var(--border) !important;
        border-radius: 14px !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.28);
    }

    /* ---------- Native metrics as cards ---------- */
    div[data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 14px 16px;
    }
    div[data-testid="stMetricLabel"] p {
        color: var(--muted);
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.4px;
    }
    div[data-testid="stMetricValue"] { font-weight: 800; }

    /* ---------- Tabs as pills ---------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: var(--surface);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid var(--border);
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        padding: 0 16px;
        border-radius: 8px;
        color: var(--muted);
        font-weight: 600;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"]:hover { color: var(--text); background: var(--surface-2); }
    .stTabs [aria-selected="true"] {
        color: #FFFFFF !important;
        background: linear-gradient(135deg, var(--primary-strong), #7C3AED) !important;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35);
    }
    .stTabs [data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tab-border"] { display: none; }

    /* ---------- Buttons ---------- */
    .stButton > button, .stDownloadButton > button {
        border-radius: 10px;
        font-weight: 600;
        border: 1px solid var(--border);
        transition: all 0.15s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        border-color: var(--primary);
        color: var(--primary);
        transform: translateY(-1px);
    }
    .stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"],
    button[data-testid="stBaseButton-primary"] {
        background: linear-gradient(135deg, var(--primary-strong), #7C3AED);
        border: none;
        color: #FFFFFF;
        box-shadow: 0 6px 18px rgba(99, 102, 241, 0.35);
    }
    .stButton > button[kind="primary"]:hover, .stDownloadButton > button[kind="primary"]:hover,
    button[data-testid="stBaseButton-primary"]:hover {
        color: #FFFFFF;
        filter: brightness(1.1);
    }

    /* ---------- Inputs, alerts, tables ---------- */
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div,
    div[data-baseweb="textarea"] { border-radius: 10px; }
    div[data-testid="stAlert"] { border-radius: 12px; border: 1px solid var(--border); }
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
    }
    div[data-testid="stFileUploaderDropzone"] {
        background: var(--surface);
        border: 1.5px dashed var(--primary-strong);
        border-radius: 14px;
    }
    div[data-testid="stExpander"] {
        border: 1px solid var(--border);
        border-radius: 12px;
        background: var(--surface);
    }
    footer { visibility: hidden; }

    /* ---------- Prediction badges ---------- */
    .badge-top, .badge-mid, .badge-risk {
        padding: 10px 20px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 1.25rem;
        display: inline-block;
    }
    .badge-top {
        background: rgba(52, 211, 153, 0.12);
        color: var(--ok);
        border: 1px solid rgba(52, 211, 153, 0.55);
        box-shadow: 0 0 22px rgba(52, 211, 153, 0.18);
    }
    .badge-mid {
        background: rgba(251, 191, 36, 0.12);
        color: var(--warn);
        border: 1px solid rgba(251, 191, 36, 0.55);
        box-shadow: 0 0 22px rgba(251, 191, 36, 0.16);
    }
    .badge-risk {
        background: rgba(248, 113, 113, 0.12);
        color: var(--danger);
        border: 1px solid rgba(248, 113, 113, 0.55);
        box-shadow: 0 0 22px rgba(248, 113, 113, 0.18);
    }

    /* ---------- Dashboard KPI text ---------- */
    .kpi-title {
        font-size: 0.74rem;
        font-weight: 600;
        color: var(--muted);
        margin-bottom: 6px;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }
    .kpi-value {
        font-size: 1.75rem;
        font-weight: 800;
        color: var(--text);
        letter-spacing: -0.02em;
    }
    .kpi-sub {
        font-size: 0.78rem;
        color: var(--muted);
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Set visualization style
# Dark chart theme so Matplotlib/Seaborn figures blend into the card surfaces
_SURFACE, _BORDER, _TEXT, _MUTED = "#111A2E", "#24304D", "#E2E8F0", "#94A3B8"
sns.set_theme(
    style="darkgrid",
    palette=["#818CF8", "#2DD4BF", "#FBBF24", "#F87171", "#A78BFA", "#34D399"],
    rc={
        "figure.facecolor": _SURFACE,
        "axes.facecolor": _SURFACE,
        "savefig.facecolor": _SURFACE,
        "axes.edgecolor": _BORDER,
        "axes.labelcolor": _MUTED,
        "axes.titlecolor": _TEXT,
        "text.color": _TEXT,
        "xtick.color": _MUTED,
        "ytick.color": _MUTED,
        "grid.color": "#1E2A45",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.facecolor": _SURFACE,
        "legend.edgecolor": _BORDER,
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "DejaVu Sans"],
        "font.size": 10,
    },
)

# =========================================================
# 2. SESSION STATE INITIALIZATION
# =========================================================

if "dataset" not in st.session_state:
    st.session_state.dataset = None
if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = "None"
if "trained_state" not in st.session_state:
    st.session_state.trained_state = None
if "active_model_name" not in st.session_state:
    st.session_state.active_model_name = None
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []
if "last_single_prediction" not in st.session_state:
    st.session_state.last_single_prediction = None

def reset_model_state():
    """Invalidates previously trained model state when dataset changes."""
    st.session_state.trained_state = None
    st.session_state.active_model_name = None
    st.session_state.last_single_prediction = None

def load_dataset_into_state(df: pd.DataFrame, name: str):
    """Loads a new dataset and safely resets model artifacts."""
    st.session_state.dataset = df.copy()
    st.session_state.dataset_name = name
    reset_model_state()

# Auto-load default dataset on startup if not already loaded so Dashboard is populated immediately
if st.session_state.dataset is None:
    sample_path = os.path.join(BASE_DIR, "data", "kaggle_student_performance_train.csv")
    if os.path.exists(sample_path):
        df_sample = pd.read_csv(sample_path)
        load_dataset_into_state(df_sample, "Kaggle Student Performance Dataset")

# =========================================================
# 3. SIDEBAR NAVIGATION & SYSTEM STATUS
# =========================================================

with st.sidebar:
    st.title("🎓 Navigation & Info")
    st.markdown("**Dynamic Student Performance Prediction**")
    st.caption("Supervised Machine Learning System for Academic Outcomes")
    
    st.divider()
    
    st.subheader("📌 System Status")
    if st.session_state.dataset is not None:
        st.success(f"📁 Dataset: **{st.session_state.dataset_name}**")
        st.write(f"📊 Records: `{len(st.session_state.dataset):,}` | Features: `{len(st.session_state.dataset.columns)}`")
    else:
        st.warning("⚠️ No dataset loaded yet.")
        
    if st.session_state.trained_state is not None:
        t_state = st.session_state.trained_state
        st.success(f"⚙️ Model: **{st.session_state.active_model_name}**")
        st.write(f"🎯 Target: `{t_state['target_col']}`")
        st.write(f"🏷️ Task: `{t_state['task_type'].title()}`")
        st.write(f"🔢 Input Features: `{len(t_state['feature_cols'])}`")
    else:
        st.info("ℹ️ Model not trained yet.")
        
    st.divider()
    
    st.subheader("⚡ Quick Load Datasets")
    if st.button("📥 Load Kaggle Exam Scores (Regression)", use_container_width=True, type="secondary"):
        sample_path = os.path.join(BASE_DIR, "data", "kaggle_student_performance_train.csv")
        if os.path.exists(sample_path):
            load_dataset_into_state(pd.read_csv(sample_path), "Kaggle Exam Scores (Regression)")
            st.rerun()
            
    if st.button("📥 Load Kaggle Grades (Classification)", use_container_width=True, type="secondary"):
        sample_path = os.path.join(BASE_DIR, "data", "kaggle_student_performance_train.csv")
        if os.path.exists(sample_path):
            load_dataset_into_state(pd.read_csv(sample_path), "Kaggle Grades & Pass-Fail (Classification)")
            st.rerun()

    if st.button("📥 Load Original Marks & Attendance", use_container_width=True, type="secondary"):
        sample_path = os.path.join(BASE_DIR, "data", "student_data.csv")
        if os.path.exists(sample_path):
            load_dataset_into_state(pd.read_csv(sample_path), "Original Student Attendance & Marks (student_data.csv)")
            st.rerun()

    st.divider()
    
    with st.expander("🔍 Historical Project Modules"):
        st.markdown("""
        **Legacy Unsupervised Module:**
        - Original K-Means Clustering & PCA artifacts are preserved in `models/` and `visual_outputs/`.
        - Historical student dataset available at `data/student_data.csv`.
        """)
        
    st.caption("MCA Final / Mini Project Submission")

# =========================================================
# 4. MAIN HEADER
# =========================================================

st.markdown('<div class="main-header"><span>🎓</span><span class="hdr-text">Dynamic Student Performance Prediction System</span></div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload any Kaggle student dataset, dynamically select academic features and target goals, train supervised ML models, and predict marks, grades, or pass/fail outcomes.</div>', unsafe_allow_html=True)

# 6 Required Application Tabs
tab_dash, tab_upload, tab_train, tab_pred, tab_batch, tab_eval = st.tabs([
    "🏠 Dashboard",
    "📂 Upload Dataset",
    "⚙️ Train Model",
    "🔮 Predict Student Marks",
    "📁 Batch Prediction",
    "📊 Model Analytics"
])

# =========================================================
# TAB 1: DASHBOARD
# =========================================================

with tab_dash:
    st.subheader("📊 Executive Academic Dashboard")
    st.caption("Dynamic summary statistics and cohort distributions computed directly from the active dataset.")
    
    if st.session_state.dataset is not None:
        df = st.session_state.dataset
        semantic = find_semantic_columns(df)
        
        # --- ROW 1: DYNAMIC KPI CARDS ---
        col_k1, col_k2, col_k3, col_k4, col_k5, col_k6 = st.columns(6)
        
        with col_k1:
            with st.container(border=True):
                st.markdown('<div class="kpi-title">Total Students</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="kpi-value">{len(df):,}</div>', unsafe_allow_html=True)
                st.markdown('<div class="kpi-sub">Enrolled Cohort</div>', unsafe_allow_html=True)
                
        with col_k2:
            with st.container(border=True):
                st.markdown('<div class="kpi-title">Features</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="kpi-value">{len(df.columns)}</div>', unsafe_allow_html=True)
                st.markdown('<div class="kpi-sub">Attributes</div>', unsafe_allow_html=True)

        with col_k3:
            with st.container(border=True):
                st.markdown('<div class="kpi-title">Avg Marks</div>', unsafe_allow_html=True)
                if semantic["marks_col"]:
                    avg_marks = df[semantic["marks_col"]].dropna().mean()
                    st.markdown(f'<div class="kpi-value">{avg_marks:.1f}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="kpi-sub">Col: {semantic["marks_col"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="kpi-value">N/A</div>', unsafe_allow_html=True)
                    st.markdown('<div class="kpi-sub">No marks col</div>', unsafe_allow_html=True)

        with col_k4:
            with st.container(border=True):
                st.markdown('<div class="kpi-title">Avg Attendance</div>', unsafe_allow_html=True)
                if semantic["attendance_col"]:
                    avg_att = df[semantic["attendance_col"]].dropna().mean()
                    st.markdown(f'<div class="kpi-value">{avg_att:.1f}%</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="kpi-sub">Col: {semantic["attendance_col"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="kpi-value">N/A</div>', unsafe_allow_html=True)
                    st.markdown('<div class="kpi-sub">No attendance col</div>', unsafe_allow_html=True)

        with col_k5:
            with st.container(border=True):
                st.markdown('<div class="kpi-title">Pass Rate</div>', unsafe_allow_html=True)
                pass_rate = None
                if semantic["pass_fail_col"]:
                    series = df[semantic["pass_fail_col"]].dropna().astype(str).str.lower()
                    passes = series.str.contains("pass|1|true").sum()
                    pass_rate = (passes / len(series)) * 100 if len(series) > 0 else None
                elif semantic["marks_col"]:
                    # Assume passing benchmark >= 50
                    m_series = pd.to_numeric(df[semantic["marks_col"]], errors="coerce").dropna()
                    passes = (m_series >= 50).sum()
                    pass_rate = (passes / len(m_series)) * 100 if len(m_series) > 0 else None
                
                if pass_rate is not None:
                    st.markdown(f'<div class="kpi-value">{pass_rate:.1f}%</div>', unsafe_allow_html=True)
                    st.markdown('<div class="kpi-sub">Calculated</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="kpi-value">N/A</div>', unsafe_allow_html=True)
                    st.markdown('<div class="kpi-sub">Not detected</div>', unsafe_allow_html=True)

        with col_k6:
            with st.container(border=True):
                st.markdown('<div class="kpi-title">At-Risk Rate</div>', unsafe_allow_html=True)
                risk_count = 0
                if semantic["marks_col"]:
                    m_series = pd.to_numeric(df[semantic["marks_col"]], errors="coerce").dropna()
                    risk_count = (m_series < 45).sum()
                    risk_pct = (risk_count / len(m_series)) * 100 if len(m_series) > 0 else 0
                    st.markdown(f'<div class="kpi-value" style="color:#F87171;">{risk_pct:.1f}%</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="kpi-sub">{risk_count} students (&lt;45)</div>', unsafe_allow_html=True)
                elif semantic["pass_fail_col"]:
                    series = df[semantic["pass_fail_col"]].dropna().astype(str).str.lower()
                    fails = series.str.contains("fail|0|false").sum()
                    fail_pct = (fails / len(series)) * 100 if len(series) > 0 else 0
                    st.markdown(f'<div class="kpi-value" style="color:#F87171;">{fail_pct:.1f}%</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="kpi-sub">{fails} failing</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="kpi-value">N/A</div>', unsafe_allow_html=True)
                    st.markdown('<div class="kpi-sub">No marks/pass rule</div>', unsafe_allow_html=True)

        # --- ROW 2: DYNAMIC CHARTS COMPUTED FROM DATASET ---
        st.write("")
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            with st.container(border=True):
                st.markdown("#### 📈 Academic Score Distribution")
                score_col = semantic["marks_col"]
                if score_col and pd.api.types.is_numeric_dtype(df[score_col]):
                    fig, ax = plt.subplots(figsize=(7, 4))
                    sns.histplot(df[score_col].dropna(), kde=True, ax=ax, color="#818CF8", bins=20)
                    ax.set_title(f"Distribution of {score_col}", fontweight="bold", pad=10)
                    ax.set_xlabel(score_col)
                    ax.set_ylabel("Student Count")
                    st.pyplot(fig)
                    plt.close()
                else:
                    st.info("No continuous numeric score column detected in this dataset.")
                    
        with col_c2:
            with st.container(border=True):
                st.markdown("#### 🎯 Attendance vs. Score Correlation")
                att_col = semantic["attendance_col"]
                score_col = semantic["marks_col"]
                if att_col and score_col and pd.api.types.is_numeric_dtype(df[att_col]) and pd.api.types.is_numeric_dtype(df[score_col]):
                    fig, ax = plt.subplots(figsize=(7, 4))
                    sns.scatterplot(data=df, x=att_col, y=score_col, ax=ax, color="#818CF8", alpha=0.75, s=40)
                    sns.regplot(data=df, x=att_col, y=score_col, ax=ax, scatter=False, color="#F87171")
                    ax.set_title(f"{att_col} vs. {score_col}", fontweight="bold", pad=10)
                    st.pyplot(fig)
                    plt.close()
                else:
                    st.info("Both attendance and score columns are required for correlation view.")

        # --- ROW 3: HEATMAP & FEATURE SUMMARY ---
        col_c3, col_c4 = st.columns(2)
        
        with col_c3:
            with st.container(border=True):
                st.markdown("#### 🗺️ Correlation Matrix (Numeric Features)")
                numeric_df = df.select_dtypes(include=[np.number])
                if len(numeric_df.columns) >= 2:
                    fig, ax = plt.subplots(figsize=(7, 4.5))
                    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="mako", ax=ax, cbar=True, square=True)
                    ax.set_title("Feature Correlation Heatmap", fontweight="bold", pad=10)
                    st.pyplot(fig)
                    plt.close()
                else:
                    st.info("Need at least 2 numeric features for a correlation heatmap.")
                    
        with col_c4:
            with st.container(border=True):
                st.markdown("#### 📊 Study Hours / Category Breakdown")
                study_col = semantic["study_hours_col"]
                if study_col and pd.api.types.is_numeric_dtype(df[study_col]):
                    fig, ax = plt.subplots(figsize=(7, 4.5))
                    sns.histplot(df[study_col].dropna(), kde=True, ax=ax, color="#2DD4BF", bins=15)
                    ax.set_title(f"Distribution of {study_col}", fontweight="bold", pad=10)
                    ax.set_xlabel("Weekly Hours")
                    st.pyplot(fig)
                    plt.close()
                else:
                    # Fallback to categorical distribution
                    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() <= 10]
                    if cat_cols:
                        fig, ax = plt.subplots(figsize=(7, 4.5))
                        sns.countplot(data=df, y=cat_cols[0], ax=ax, color="#818CF8")
                        ax.set_title(f"Breakdown by {cat_cols[0]}", fontweight="bold", pad=10)
                        st.pyplot(fig)
                        plt.close()
                    else:
                        st.info("No categorical or study hours column available.")

    else:
        st.info("No dataset is currently loaded. Please upload a CSV in Tab 2 or click one of the sample buttons in the sidebar.")

# =========================================================
# TAB 2: UPLOAD DATASET
# =========================================================

with tab_upload:
    st.subheader("📂 Dynamic Dataset Ingestion & Schema Inspector")
    st.markdown("Upload any CSV dataset containing student academic records, marks, study hours, or demographics.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], help="Upload any student academic dataset")
    
    if uploaded_file is not None:
        try:
            df_new = pd.read_csv(uploaded_file)
            if df_new.empty:
                st.error("The uploaded CSV is empty. Please upload a file with records.")
            else:
                if len(df_new.columns) != len(set(df_new.columns)):
                    df_new.columns = [f"{col}_{i}" if col in df_new.columns[:i] else col for i, col in enumerate(df_new.columns)]
                    st.warning("Duplicate column names detected and automatically renamed.")
                    
                if st.session_state.dataset_name != uploaded_file.name or st.session_state.dataset is None:
                    load_dataset_into_state(df_new, uploaded_file.name)
                    st.success(f"Successfully uploaded: **{uploaded_file.name}**")
                    st.info("Previous model artifacts have been cleared to ensure compatibility with this new dataset.")
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            
    st.divider()
    
    if st.session_state.dataset is not None:
        df = st.session_state.dataset
        st.subheader(f"📊 Dataset Explorer: {st.session_state.dataset_name}")
        
        num_cols, cat_cols = separate_feature_types(df, df.columns.tolist())
        missing_count = int(df.isna().sum().sum())
        duplicate_count = int(df.duplicated().sum())
        
        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
        col_m1.metric("Total Records", f"{len(df):,}")
        col_m2.metric("Total Columns", len(df.columns))
        col_m3.metric("Numerical Columns", len(num_cols))
        col_m4.metric("Categorical Columns", len(cat_cols))
        col_m5.metric("Missing Cells", f"{missing_count:,}")
        
        if duplicate_count > 0:
            st.warning(f"⚠️ Dataset contains **{duplicate_count}** duplicate rows.")
            if st.button("🧹 Remove Duplicate Rows", type="secondary"):
                df_dedup = df.drop_duplicates()
                load_dataset_into_state(df_dedup, st.session_state.dataset_name)
                st.success("Duplicate rows removed!")
                st.rerun()

        # Preview rows selector
        st.write("")
        col_prev1, col_prev2 = st.columns([1, 4])
        with col_prev1:
            preview_rows = st.selectbox("Rows to preview:", options=[5, 10, 20, 50, len(df)], index=1)
            
        st.markdown(f"**Dataset Preview (First {preview_rows} Rows):**")
        st.dataframe(df.head(preview_rows), use_container_width=True)
        
        # Column Schema & Missing Values Table
        st.write("")
        st.markdown("**Column Schema & Missing Values Analysis:**")
        col_info = []
        for col in df.columns:
            non_null = int(df[col].notna().sum())
            null_count = int(df[col].isna().sum())
            null_pct = round((null_count / len(df)) * 100, 2)
            dtype_str = str(df[col].dtype)
            samples = ", ".join([str(v) for v in df[col].dropna().unique()[:3]])
            col_info.append({
                "Column Name": col,
                "Data Type": dtype_str,
                "Non-Null Count": non_null,
                "Missing Count": null_count,
                "Missing (%)": f"{null_pct}%",
                "Sample Unique Values": samples
            })
        st.dataframe(pd.DataFrame(col_info), use_container_width=True)
        
        # Descriptive Statistics
        st.write("")
        st.markdown("**Statistical Summary (Numerical Features):**")
        if num_cols:
            st.dataframe(df[num_cols].describe().round(2), use_container_width=True)
        else:
            st.info("No numerical columns found in this dataset.")
    else:
        st.info("No dataset currently active. Please upload a CSV file above or pick a sample from the sidebar.")

# =========================================================
# TAB 3: TRAIN MODEL
# =========================================================

with tab_train:
    st.subheader("⚙️ Supervised Model Training & Multi-Algorithm Benchmark")
    
    if st.session_state.dataset is None:
        st.warning("⚠️ No dataset loaded. Please upload a dataset in Tab 2 or load a sample from the sidebar first.")
    else:
        df = st.session_state.dataset
        all_cols = df.columns.tolist()
        id_cols = detect_identifier_columns(df)
        
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            with st.container(border=True):
                st.markdown("#### 1. Select Target Outcome")
                target_col = st.selectbox(
                    "Choose the Target Column to Predict:",
                    options=all_cols,
                    index=len(all_cols) - 1,
                    help="Select target (e.g. Exam_Score, Final_Marks, Grade, Pass/Fail)"
                )
                
                detected_task_info = detect_task_type(df, target_col)
                detected_task = detected_task_info["task"]
                
                st.info(f"💡 **Auto-Detected Task:** `{detected_task.upper()}`\n\n{detected_task_info['reason']}")
                
                task_type_override = st.radio(
                    "Confirm or Override Task Type:",
                    options=["Regression (Numerical Marks / Continuous Scores)", "Classification (Grades / Pass-Fail / Categories)"],
                    index=0 if detected_task == "regression" else 1,
                    help="Confirm or override if you want to treat discrete numbers as categories or continuous values."
                )
                chosen_task = "regression" if "Regression" in task_type_override else "classification"
            
        with col_c2:
            with st.container(border=True):
                st.markdown("#### 2. Select Predictor Features")
                candidate_features = [c for c in all_cols if c != target_col]
                default_features = [c for c in candidate_features if c not in id_cols]
                
                if id_cols:
                    st.caption(f"🛡️ Auto-excluded ID/Identifier columns: `{', '.join(id_cols)}`")
                    
                col_sel1, col_sel2 = st.columns(2)
                with col_sel1:
                    select_all = st.button("Select All Features", key="train_sel_all")
                with col_sel2:
                    clear_all = st.button("Clear Selection", key="train_clr_all")
                    
                if select_all:
                    current_features = candidate_features
                elif clear_all:
                    current_features = []
                else:
                    current_features = default_features
                    
                selected_features = st.multiselect(
                    "Choose Input Feature Columns:",
                    options=candidate_features,
                    default=current_features,
                    key="multisel_features"
                )
                
                num_selected, cat_selected = separate_feature_types(df, selected_features)
                st.caption(f"Selected: **{len(selected_features)}** features ({len(num_selected)} numerical, {len(cat_selected)} categorical)")

        st.write("")
        with st.container(border=True):
            st.markdown("#### 3. Algorithms & Hyperparameters")
            
            avail_algos = get_available_algorithms(chosen_task)
            col_opt1, col_opt2 = st.columns(2)
            
            with col_opt1:
                training_mode = st.radio(
                    "Model Training Mode:",
                    options=["Benchmark All Available Models (Recommended)", "Train Single Specific Model"],
                    index=0
                )
                if training_mode == "Train Single Specific Model":
                    chosen_single_algo = st.selectbox("Select Algorithm to Train:", options=avail_algos)
                    models_to_run = [chosen_single_algo]
                else:
                    models_to_run = avail_algos

            with col_opt2:
                test_size_val = st.slider("Test Split Size (%):", min_value=10, max_value=40, value=20, step=5) / 100.0
                random_seed = st.number_input("Random Seed (Reproducibility):", value=42, step=1)

            st.write("")
            train_button = st.button("🚀 Train Machine Learning Models", type="primary", use_container_width=True)
        
        if train_button:
            if not selected_features:
                st.error("Please select at least one input feature column to train.")
            elif target_col in selected_features:
                st.error("Target column cannot also be an input feature. Please deselect it.")
            else:
                progress_bar = st.progress(0, text="Initializing preprocessing pipeline...")
                try:
                    progress_bar.progress(30, text="Fitting leak-free ColumnTransformer & training models...")
                    trained_state = train_supervised_models(
                        df=df,
                        feature_cols=selected_features,
                        target_col=target_col,
                        task_type=chosen_task,
                        selected_models=models_to_run,
                        test_size=test_size_val,
                        random_state=int(random_seed)
                    )
                    progress_bar.progress(100, text="Training and evaluation complete!")
                    st.session_state.trained_state = trained_state
                    st.session_state.active_model_name = trained_state["best_model_name"]
                    st.success(f"✅ Training completed! Evaluated on {trained_state['test_size']} unseen test records.")
                except Exception as e:
                    progress_bar.empty()
                    st.error(f"Training failed: {e}")
                    
        # Leaderboard display
        if st.session_state.trained_state is not None:
            ts = st.session_state.trained_state
            st.divider()
            st.subheader("🏆 Model Benchmark Leaderboard")
            st.markdown(f"**Recommended Model:** 🥇 **{ts['best_model_name']}**")
            st.dataframe(ts["metrics_df"], use_container_width=True)
            
            chosen_active_model = st.selectbox(
                "Select Active Model for Live & Batch Predictions:",
                options=list(ts["pipelines"].keys()),
                index=list(ts["pipelines"].keys()).index(st.session_state.active_model_name)
            )
            st.session_state.active_model_name = chosen_active_model
            st.info(f"Active model set to: **{chosen_active_model}**")

# =========================================================
# TAB 4: PREDICT STUDENT MARKS & WHAT-IF ANALYSIS
# =========================================================

with tab_pred:
    st.subheader("🔮 Individual Student Performance Prediction")
    
    if st.session_state.trained_state is None:
        st.warning("⚠️ No machine learning model has been trained yet. Please train a model in Tab 3 (Train Model) first.")
    else:
        ts = st.session_state.trained_state
        df = st.session_state.dataset
        active_model = st.session_state.active_model_name
        
        st.markdown(f"Enter student details below to predict **{ts['target_col']}** using **{active_model}**.")
        
        # Build dynamic inputs based on selected features
        student_inputs = {}
        feature_cols = ts["feature_cols"]
        cols_grid = st.columns(3)
        
        for idx, col in enumerate(feature_cols):
            grid_col = cols_grid[idx % 3]
            with grid_col:
                if col in ts["numeric_cols"]:
                    col_series = df[col].dropna()
                    min_val = float(col_series.min()) if not col_series.empty else 0.0
                    max_val = float(col_series.max()) if not col_series.empty else 100.0
                    mean_val = float(col_series.mean()) if not col_series.empty else 50.0
                    
                    student_inputs[col] = st.number_input(
                        f"📝 {col}:",
                        value=round(mean_val, 2),
                        min_value=round(min_val - 1.0, 2),
                        max_value=round(max_val + 10.0, 2),
                        key=f"single_num_{col}"
                    )
                else:
                    unique_vals = sorted([str(x) for x in df[col].dropna().unique()])
                    student_inputs[col] = st.selectbox(
                        f"📋 {col}:",
                        options=unique_vals if unique_vals else ["Unknown"],
                        key=f"single_cat_{col}"
                    )
                    
        st.write("")
        predict_single_btn = st.button("🔍 Predict Student Outcome", type="primary", use_container_width=True)
        
        if predict_single_btn:
            try:
                pred_result = predict_single_student(ts, student_inputs, model_name=active_model)
                st.session_state.last_single_prediction = pred_result
                st.session_state.baseline_input = student_inputs.copy()
                
                # Append to Prediction History Log
                history_record = {
                    "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Model": active_model,
                    "Target": ts["target_col"],
                    "Prediction": pred_result["prediction"],
                    **student_inputs
                }
                st.session_state.prediction_history.append(history_record)
                
            except Exception as e:
                st.error(f"Prediction failed: {e}")

        # Render Prediction Results
        if st.session_state.last_single_prediction is not None:
            pred_result = st.session_state.last_single_prediction
            st.divider()
            st.subheader("🎯 Prediction Result")
            
            col_res1, col_res2 = st.columns([1.6, 1])
            
            with col_res1:
                with st.container(border=True):
                    if pred_result["task_type"] == "regression":
                        pred_val = pred_result["prediction"]
                        badge_class = "badge-top" if pred_val >= 75 else ("badge-mid" if pred_val >= 50 else "badge-risk")
                        
                        st.markdown(f'<div class="{badge_class}">Predicted {ts["target_col"]}: {pred_val}</div>', unsafe_allow_html=True)
                        st.write("")
                        st.write(f"Algorithm: **{pred_result['model_used']}**")
                        
                        if 0 <= pred_val <= 100:
                            st.progress(float(pred_val / 100.0))
                            st.caption(f"Score Scale: {pred_val:.1f}%")
                    else:
                        pred_class = str(pred_result["prediction"])
                        is_pass = "pass" in pred_class.lower() or pred_class in ["A", "B", "1"]
                        badge_class = "badge-top" if is_pass else "badge-risk"
                        
                        st.markdown(f'<div class="{badge_class}">Predicted {ts["target_col"]}: {pred_class}</div>', unsafe_allow_html=True)
                        st.write("")
                        st.write(f"Algorithm: **{pred_result['model_used']}**")
                        
                        if pred_result["probabilities"] is not None:
                            st.write("**Class Probability Breakdown:**")
                            prob_df = pd.DataFrame(list(pred_result["probabilities"].items()), columns=["Class", "Probability"])
                            st.dataframe(prob_df, use_container_width=True)
                            
                # Academic Suggestions
                st.write("")
                st.markdown("#### 💡 Academic Insights & Guidance")
                suggestions = generate_academic_suggestions(ts, st.session_state.baseline_input, pred_result["prediction"])
                for s in suggestions:
                    st.markdown(s)
                st.caption("ℹ️ *Guideline Notice: Suggestions are rule-based advisory indicators derived from cohort distributions, not guaranteed causal conclusions.*")

            with col_res2:
                with st.container(border=True):
                    st.markdown("**Student Input Summary:**")
                    st.json(st.session_state.baseline_input)

            # --- WHAT-IF SCENARIO ANALYSIS ---
            st.write("")
            with st.expander("🔬 What-If Scenario Analysis (Interactive Simulator)", expanded=False):
                st.markdown("""
                Modify academic inputs below to simulate how the model's estimate changes.  
                *Note: Results reflect statistical model estimates and do not imply direct causation.*
                """)
                
                sim_inputs = st.session_state.baseline_input.copy()
                cols_whatif = st.columns(3)
                
                for idx, col in enumerate(ts["numeric_cols"]):
                    grid_col = cols_whatif[idx % 3]
                    with grid_col:
                        val = float(sim_inputs[col])
                        sim_inputs[col] = st.slider(
                            f"Simulate {col}:",
                            min_value=0.0,
                            max_value=max(100.0, val * 1.5),
                            value=val,
                            step=1.0,
                            key=f"whatif_{col}"
                        )
                        
                if st.button("⚡ Run What-If Simulation", type="secondary"):
                    sim_result = predict_single_student(ts, sim_inputs, model_name=active_model)
                    
                    col_b1, col_b2, col_b3 = st.columns(3)
                    with col_b1:
                        st.metric("Baseline Prediction", f"{pred_result['prediction']}")
                    with col_b2:
                        st.metric("Simulated Prediction", f"{sim_result['prediction']}")
                    with col_b3:
                        if ts["task_type"] == "regression":
                            delta = sim_result["prediction"] - pred_result["prediction"]
                            st.metric("Predicted Change (Δ)", f"{delta:+.2f}")
                        else:
                            st.write(f"Class Transition: `{pred_result['prediction']}` → `{sim_result['prediction']}`")
                            
                    st.caption("⚠️ Simulation Notice: Prediction changes represent mathematical model approximations based on training correlations.")

        # --- PREDICTION HISTORY SECTION ---
        if st.session_state.prediction_history:
            st.write("")
            with st.expander(f"📜 Prediction History Log ({len(st.session_state.prediction_history)} records in session)", expanded=False):
                history_df = pd.DataFrame(st.session_state.prediction_history)
                st.dataframe(history_df, use_container_width=True)
                
                csv_history = history_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Prediction History (CSV)",
                    data=csv_history,
                    file_name="student_prediction_history.csv",
                    mime="text/csv"
                )

# =========================================================
# TAB 5: BATCH PREDICTION
# =========================================================

with tab_batch:
    st.subheader("📁 Batch Student Performance Prediction (CSV Ingestion)")
    st.markdown("""
    Upload a CSV containing records for multiple students.  
    The file **does not need to contain the target column**. The model will apply the learned preprocessing pipeline and predict outcomes for every row.
    """)
    
    if st.session_state.trained_state is None:
        st.warning("⚠️ Please train a model in Tab 3 (Train Model) before running batch inference.")
    else:
        ts = st.session_state.trained_state
        active_model = st.session_state.active_model_name
        
        st.markdown(f"**Required Feature Columns:** `{', '.join(ts['feature_cols'])}`")
        
        col_bu1, col_bu2 = st.columns([2, 1])
        with col_bu1:
            batch_file = st.file_uploader("Upload Batch CSV", type=["csv"], key="batch_uploader")
        with col_bu2:
            st.write("")
            st.write("")
            use_test_batch = st.button("✨ Use Test Split as Sample Batch", help="Quickly test batch prediction using unseen test records")
            
        batch_df = None
        if batch_file is not None:
            try:
                batch_df = pd.read_csv(batch_file)
            except Exception as e:
                st.error(f"Error reading batch CSV: {e}")
        elif use_test_batch:
            batch_df = ts["X_test"].copy()
            st.info(f"Loaded {len(batch_df)} test split records as a sample batch.")
            
        if batch_df is not None:
            st.write("")
            with st.spinner("Processing batch predictions..."):
                output_batch_df, error_msg, batch_metrics = predict_batch_students(ts, batch_df, model_name=active_model)
                
                if error_msg:
                    st.error(error_msg)
                else:
                    st.success(f"✅ Generated predictions for all **{len(output_batch_df)}** student records using **{active_model}**!")
                    
                    pred_col_name = f"Predicted_{ts['target_col']}"
                    
                    # If batch CSV also had ground truth target, display evaluation metrics
                    if batch_metrics:
                        with st.container(border=True):
                            st.markdown("#### 🎯 Batch Ground-Truth Validation Metrics")
                            cols_bm = st.columns(len(batch_metrics))
                            for idx, (m_name, m_val) in enumerate(batch_metrics.items()):
                                cols_bm[idx].metric(m_name, m_val)
                                
                    # Summary metrics
                    if ts["task_type"] == "regression":
                        col_stat1, col_stat2, col_stat3 = st.columns(3)
                        col_stat1.metric("Average Predicted", f"{output_batch_df[pred_col_name].mean():.2f}")
                        col_stat2.metric("Highest Predicted", f"{output_batch_df[pred_col_name].max():.2f}")
                        col_stat3.metric("Lowest Predicted", f"{output_batch_df[pred_col_name].min():.2f}")
                    else:
                        st.markdown("**Predicted Class Distribution:**")
                        class_counts = output_batch_df[pred_col_name].value_counts().reset_index()
                        class_counts.columns = ["Class", "Count"]
                        st.dataframe(class_counts, use_container_width=True)
                        
                    st.write("")
                    st.dataframe(output_batch_df, use_container_width=True)
                    
                    csv_export = output_batch_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Classified Student Batch Report (CSV)",
                        data=csv_export,
                        file_name=f"batch_predictions_{ts['target_col']}.csv",
                        mime="text/csv",
                        type="primary"
                    )

# =========================================================
# TAB 6: MODEL ANALYTICS & EXPLAINABILITY
# =========================================================

with tab_eval:
    st.subheader("📊 Model Analytics, Evaluation & Explainability")
    
    if st.session_state.trained_state is None:
        st.warning("⚠️ No trained model available. Please train a model in Tab 3 to view analytics.")
    else:
        ts = st.session_state.trained_state
        
        # Model switcher within analytics to compare any trained model
        col_an1, col_an2 = st.columns([2, 1])
        with col_an1:
            st.markdown(f"Showing evaluation metrics for: **{st.session_state.active_model_name}**")
        with col_an2:
            inspect_model = st.selectbox(
                "Switch Model to Inspect:",
                options=list(ts["pipelines"].keys()),
                index=list(ts["pipelines"].keys()).index(st.session_state.active_model_name)
            )
            
        y_test = ts["y_test"]
        y_pred = ts["predictions_dict"][inspect_model]
        
        st.markdown(f"Evaluated on **{ts['test_size']} unseen test records**:")
        
        # 1. Performance Metrics Cards
        if ts["task_type"] == "regression":
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = float(np.sqrt(mse))
            r2 = r2_score(y_test, y_pred)
            
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("MAE (Mean Absolute Error)", f"{mae:.3f}")
            col_m2.metric("MSE (Mean Squared Error)", f"{mse:.3f}")
            col_m3.metric("RMSE (Root Mean Squared)", f"{rmse:.3f}")
            col_m4.metric("R² Score (Variance Explained)", f"{r2:.4f}")
            
            st.divider()
            
            # Regression Charts: Actual vs Predicted & Residuals
            col_plot1, col_plot2 = st.columns(2)
            
            with col_plot1:
                with st.container(border=True):
                    fig, ax = plt.subplots(figsize=(6, 5))
                    ax.scatter(y_test, y_pred, alpha=0.75, color="#818CF8", edgecolors="#0B1120", s=45)
                    min_val = min(float(y_test.min()), float(y_pred.min()))
                    max_val = max(float(y_test.max()), float(y_pred.max()))
                    ax.plot([min_val, max_val], [min_val, max_val], color="#F87171", linestyle="--", linewidth=2, label="Ideal Fit (y = x)")
                    ax.set_title("Actual vs. Predicted Student Marks", fontweight="bold")
                    ax.set_xlabel("Actual Marks (Test Set)")
                    ax.set_ylabel("Predicted Marks")
                    ax.legend()
                    st.pyplot(fig)
                    plt.close()
                
            with col_plot2:
                with st.container(border=True):
                    fig, ax = plt.subplots(figsize=(6, 5))
                    residuals = y_test - y_pred
                    sns.histplot(residuals, kde=True, ax=ax, color="#2DD4BF", bins=15)
                    ax.axvline(0, color="#F87171", linestyle="--")
                    ax.set_title("Residual (Prediction Error) Distribution", fontweight="bold")
                    ax.set_xlabel("Prediction Error (Actual - Predicted)")
                    st.pyplot(fig)
                    plt.close()
                
        else:
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Accuracy", f"{acc:.4f}")
            col_m2.metric("Precision (Weighted)", f"{prec:.4f}")
            col_m3.metric("Recall (Weighted)", f"{rec:.4f}")
            col_m4.metric("F1-Score (Weighted)", f"{f1:.4f}")
            
            st.divider()
            
            # Confusion Matrix & Classification Report
            col_plot1, col_plot2 = st.columns([1, 1])
            
            with col_plot1:
                with st.container(border=True):
                    classes = ts["target_classes"]
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots(figsize=(6, 5))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="mako", xticklabels=classes, yticklabels=classes, ax=ax)
                    ax.set_title("Confusion Matrix Heatmap", fontweight="bold")
                    ax.set_xlabel("Predicted Class")
                    ax.set_ylabel("True Class")
                    st.pyplot(fig)
                    plt.close()
                
            with col_plot2:
                with st.container(border=True):
                    st.markdown("**Classification Report Breakdown:**")
                    report_dict = classification_report(y_test, y_pred, target_names=[str(c) for c in classes], output_dict=True, zero_division=0)
                    report_df = pd.DataFrame(report_dict).transpose().round(3)
                    st.dataframe(report_df, use_container_width=True)
                
        st.divider()
        
        # 2. Feature Importance / Key Drivers (With Permutation Fallback)
        st.subheader("🔍 Predictive Academic Drivers (Feature Importance)")
        df_imp = get_feature_importances(ts, model_name=inspect_model)
        
        if not df_imp.empty:
            with st.container(border=True):
                fig, ax = plt.subplots(figsize=(10, max(4, len(df_imp) * 0.35)))
                top_imp = df_imp.head(15).iloc[::-1]
                ax.barh(top_imp["Feature"], top_imp["Importance"], color="#818CF8", edgecolor="#0B1120", alpha=0.85)
                ax.set_title(f"Feature Importance / Weights ({inspect_model})", fontweight="bold")
                ax.set_xlabel("Relative Importance / Permutation Score")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
        else:
            st.info(f"Feature importance not directly computable for {inspect_model}.")
            
        st.caption("ℹ️ **Academic & Explainability Disclaimer:** Feature importance highlights statistical associations learned by the model. It does not establish direct scientific causation. Evaluation metrics reflect test holdouts.")