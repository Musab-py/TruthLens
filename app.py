import streamlit as st
import math
from datetime import datetime

from src.truthlens.pipeline import TruthLensPipeline


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="TruthLens | AI Fake News Detection",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# SESSION STATE
# =========================================================

if "news_input" not in st.session_state:
    st.session_state.news_input = ""

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

if "analysis_error" not in st.session_state:
    st.session_state.analysis_error = None


def load_sample(text):
    st.session_state.news_input = text
    st.session_state.last_result = None
    st.session_state.analysis_error = None


def clear_input():
    st.session_state.news_input = ""
    st.session_state.last_result = None
    st.session_state.analysis_error = None


def clear_last_result():
    st.session_state.last_result = None
    st.session_state.analysis_error = None


def clear_history():
    st.session_state.analysis_history = []


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

/* GLOBAL THEME */

.stApp {
    background:
        radial-gradient(
            circle at 85% 5%,
            rgba(56, 189, 248, 0.10),
            transparent 30%
        ),
        linear-gradient(
            135deg,
            #f8fcff 0%,
            #edf8ff 50%,
            #ffffff 100%
        );
    color: #18344d;
    font-family: 'Segoe UI', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background: transparent;
}

[data-testid="stHeader"] {
    background: rgba(255, 255, 255, 0.75);
}

.block-container {
    max-width: 1180px;
    padding-top: 2rem;
    padding-bottom: 3rem;
    padding-left: 3rem;
    padding-right: 3rem;
}

/* TYPOGRAPHY */

h1, h2, h3, h4 {
    color: #12304a !important;
    font-weight: 750 !important;
}

p, label, li {
    color: #456179;
}

/* SIDEBAR */

[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #dceef9;
}

[data-testid="stSidebar"] > div {
    background: #ffffff;
}

[data-testid="stSidebar"] * {
    color: #24445e;
}

.sidebar-logo {
    font-size: 27px;
    font-weight: 800;
    color: #12304a;
    padding: 10px 0 5px 0;
    letter-spacing: -1px;
}

.sidebar-tagline {
    color: #71869a;
    font-size: 13px;
    margin-bottom: 20px;
}

.sidebar-section {
    padding: 15px;
    background: #f2faff;
    border: 1px solid #e0f1fc;
    border-radius: 12px;
    margin: 15px 0;
    transition: all 0.3s ease;
}

.sidebar-section:hover {
    transform: translateY(-3px);
    border-color: #7dd3fc;
    box-shadow: 0 8px 20px rgba(14, 165, 233, 0.08);
}

.sidebar-status {
    display: inline-block;
    background: #e0f7ff;
    color: #0284c7 !important;
    padding: 6px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
}

/* HERO SECTION */

.hero-card {
    position: relative;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 38px 42px;
    margin-bottom: 28px;

    background: linear-gradient(
        120deg,
        #ffffff,
        #eff9ff,
        #e0f2fe
    );

    border: 1px solid #d4ecfb;
    border-radius: 22px;
    box-shadow: 0 10px 35px rgba(30, 120, 180, 0.07);

    animation: fadeUp 0.7s ease both;
}

.hero-card::before {
    content: "";
    position: absolute;
    width: 270px;
    height: 270px;
    right: -75px;
    top: -110px;
    background: rgba(125, 211, 252, 0.20);
    border-radius: 50%;
    animation: floatGlow 7s ease-in-out infinite;
}

.hero-content {
    position: relative;
    z-index: 2;
}

.hero-label {
    display: inline-block;
    padding: 7px 13px;
    margin-bottom: 12px;
    border-radius: 20px;
    background: #dff5ff;
    color: #0284c7;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 1px;
}

.hero-heading {
    font-size: 42px;
    font-weight: 850;
    color: #12304a;
    letter-spacing: -1.5px;
    margin: 0;
}

.hero-description {
    font-size: 16px;
    color: #56738c;
    margin-top: 10px;
    max-width: 600px;
    line-height: 1.7;
}

.hero-orb {
    position: relative;
    z-index: 2;
    width: 125px;
    height: 125px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 35px;

    background: linear-gradient(
        145deg,
        #ffffff,
        #cceeff
    );

    border: 1px solid #b5e3fa;
    font-size: 67px;
    box-shadow: 0 15px 35px rgba(14, 165, 233, 0.13);
    animation: floatIcon 4s ease-in-out infinite;
}

/* CARDS */

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255, 255, 255, 0.92);
    border: 1px solid #dfedf7 !important;
    border-radius: 17px !important;
    box-shadow: 0 5px 18px rgba(30, 90, 130, 0.045);
    transition: all 0.3s ease;
}

div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: #9dddfa !important;
    box-shadow: 0 12px 28px rgba(14, 165, 233, 0.09);
    transform: translateY(-2px);
}

/* TEXT AREA */

.stTextArea textarea {
    background: #f8fcff !important;
    color: #18344d !important;
    border: 1.5px solid #cfe7f7 !important;
    border-radius: 13px !important;
    font-size: 15px !important;
    line-height: 1.7 !important;
    padding: 16px !important;
    transition: all 0.3s ease !important;
}

.stTextArea textarea:focus {
    border-color: #38bdf8 !important;
    background: #ffffff !important;
    box-shadow: 0 0 0 4px rgba(56, 189, 248, 0.13) !important;
}

.stTextArea textarea::placeholder {
    color: #94a9b9 !important;
}

/* BUTTONS */

.stButton > button {
    background: linear-gradient(
        110deg,
        #0ea5e9,
        #38bdf8
    );
    color: #ffffff !important;
    border: none !important;
    border-radius: 11px !important;
    padding: 11px 20px !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    transition: all 0.25s ease !important;
}

.stButton > button:hover {
    background: linear-gradient(
        110deg,
        #0284c7,
        #0ea5e9
    );
    color: #ffffff !important;
    transform: translateY(-3px);
    box-shadow: 0 8px 20px rgba(14, 165, 233, 0.25);
}

.stButton > button:active {
    transform: scale(0.97);
}

/* METRICS */

[data-testid="stMetric"] {
    background: #f2faff;
    border: 1px solid #dfedf7;
    border-radius: 14px;
    padding: 17px;
    transition: all 0.3s ease;
}

[data-testid="stMetric"]:hover {
    border-color: #7dd3fc;
    transform: translateY(-3px);
}

[data-testid="stMetricLabel"] {
    color: #56738c !important;
}

[data-testid="stMetricValue"] {
    color: #0284c7 !important;
    font-weight: 800 !important;
}

/* PROGRESS BAR */

[data-testid="stProgressBar"] > div {
    background: #dceef9;
    border-radius: 20px;
    height: 12px;
}

[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(
        90deg,
        #38bdf8,
        #0284c7
    );
    border-radius: 20px;
}

/* ALERTS */

[data-testid="stAlert"] {
    border-radius: 13px;
    border-width: 1px;
}

/* TABS */

[data-testid="stTabs"] button {
    color: #55758c !important;
    font-weight: 650 !important;
}

[data-testid="stTabs"] button[aria-selected="true"] {
    color: #0284c7 !important;
    border-bottom-color: #38bdf8 !important;
}

[data-testid="stTabs"] [role="tablist"] {
    gap: 12px;
}

/* ANIMATIONS */

@keyframes fadeUp {
    from {
        opacity: 0;
        transform: translateY(18px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes floatIcon {
    0%, 100% {
        transform: translateY(0);
    }
    50% {
        transform: translateY(-9px);
    }
}

@keyframes floatGlow {
    0%, 100% {
        transform: scale(1);
    }
    50% {
        transform: scale(1.15);
    }
}

@keyframes resultAppear {
    from {
        opacity: 0;
        transform: translateY(15px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.result-animation {
    animation: resultAppear 0.6s ease both;
}

/* RESPONSIVE DESIGN */

@media screen and (max-width: 900px) {
    .block-container {
        padding-left: 1.5rem;
        padding-right: 1.5rem;
    }

    .hero-card {
        padding: 28px;
    }

    .hero-heading {
        font-size: 34px;
    }

    .hero-orb {
        width: 95px;
        height: 95px;
        font-size: 50px;
    }
}

@media screen and (max-width: 600px) {
    .block-container {
        padding: 1rem 0.9rem 2rem 0.9rem;
    }

    .hero-card {
        padding: 25px 20px;
        border-radius: 16px;
    }

    .hero-heading {
        font-size: 30px;
    }

    .hero-description {
        font-size: 14px;
    }

    .hero-orb {
        display: none;
    }

    .stButton > button {
        width: 100%;
    }

    [data-testid="stMetric"] {
        padding: 12px;
    }
}

/* ACCESSIBILITY */

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MACHINE LEARNING MODEL
# =========================================================

@st.cache_resource
def load_pipeline():
    return TruthLensPipeline()


try:
    truthlens = load_pipeline()
    model_ready = True
    model_error = None

except Exception as error:
    truthlens = None
    model_ready = False
    model_error = str(error)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("""
    <div class="sidebar-logo">
        🔎 TruthLens
    </div>

    <div class="sidebar-tagline">
        AI-Powered Fake News Detection
    </div>
    """, unsafe_allow_html=True)

    if model_ready:
        st.markdown(
            '<span class="sidebar-status">● Model Ready</span>',
            unsafe_allow_html=True
        )
    else:
        st.error("Model unavailable")

        with st.expander("Model loading details"):
            st.code(model_error or "Unknown error")

    st.divider()

    st.markdown("### About the Project")

    st.write(
        "TruthLens is an AI-powered fake news detection "
        "system designed to analyze news claims related "
        "to Bahria University."
    )

    st.markdown("""
    <div class="sidebar-section">
        <b>🎯 Project Objective</b>
        <p style="font-size:13px;">
        Use Natural Language Processing and Machine
        Learning to classify university-related news claims.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Technology Stack")

    technologies = [
        "🐍 Python",
        "🧠 Scikit-learn",
        "📊 TF-IDF",
        "⚙️ Logistic Regression",
        "🌐 Streamlit"
    ]

    for technology in technologies:
        st.markdown(f"- {technology}")

    st.divider()

    st.markdown("### System Information")

    st.caption("Project: TruthLens")
    st.caption("Version: 1.0")
    st.caption("Academic Year: 2026")

    st.divider()

    st.caption("Developed as an AI & Machine Learning Project")

    st.markdown(
        "<div style='font-size:12px;color:#91a6b6;'>"
        "TruthLens © 2026"
        "</div>",
        unsafe_allow_html=True
    )


# =========================================================
# MAIN HERO
# =========================================================


st.markdown("""
<div class="hero-card">
<div class="hero-content">
<div class="hero-label">✦ INTELLIGENT NEWS ANALYSIS</div>
<h1 class="hero-heading">TruthLens</h1>
<div class="hero-description">
Detect misleading news claims with Artificial Intelligence and Machine Learning.
<br>
Designed for Bahria University-related news analysis.
</div>
</div>
<div class="hero-orb">🔎</div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# DASHBOARD TABS
# =========================================================

analyze_tab, how_tab, about_tab = st.tabs([
    "🔍 News Analysis",
    "⚙️ How It Works",
    "📘 About TruthLens"
])


# =========================================================
# TAB 1 — NEWS ANALYSIS
# =========================================================

with analyze_tab:

    st.markdown("## 📰 Analyze a News Claim")

    st.markdown(
        "Enter a Bahria University-related news claim below "
        "and let TruthLens analyze its text."
    )

    st.markdown("#### Try an example")

    # Examples based on claims already tested in the project.
    fake_sample = (
        "Bahria University announces that all students "
        "will receive free meals from the university "
        "cafeteria throughout the examination period."
    )

    real_sample = (
        "Bahria University Karachi Campus published its "
        "Fall 2026 admission plan, listing admission dates, "
        "the BUET date, class commencement date, and "
        "official admissions contacts."
    )

    sample_col1, sample_col2, sample_col3 = st.columns(3)

    with sample_col1:
        st.button(
            "📄 Load REAL Example",
            use_container_width=True,
            on_click=load_sample,
            args=(real_sample,)
        )

    with sample_col2:
        st.button(
            "📄 Load FAKE Example",
            use_container_width=True,
            on_click=load_sample,
            args=(fake_sample,)
        )

    with sample_col3:
        st.button(
            "🗑️ Clear Input",
            use_container_width=True,
            on_click=clear_input
        )

    st.markdown("#### News article")

    news = st.text_area(
        "Paste your news article below:",
        height=190,
        max_chars=5000,
        key="news_input",
        placeholder=(
            "Enter a Bahria University-related news claim..."
        ),
        on_change=clear_last_result
    )

    char_count = len(news)

    st.caption(
        f"✍️ Characters entered: {char_count:,} / 5,000"
    )

    st.progress(min(char_count / 5000, 1.0))

    st.markdown("")

    analyze_button = st.button(
        "🔎 Analyze News",
        type="primary",
        use_container_width=True
    )

    # -----------------------------------------------------
    # RUN PREDICTION
    # -----------------------------------------------------

    if analyze_button:

        if not news.strip():

            st.warning(
                "⚠️ Please enter a news article before analysis."
            )

        elif len(news.strip()) < 20:

            st.warning(
                "Please enter a longer news claim "
                "containing at least 20 characters."
            )

        elif not model_ready:

            st.error(
                "The machine learning model could not be loaded. "
                "Please check your model files and project setup."
            )

        else:

            try:

                with st.spinner(
                    "🔎 TruthLens is analyzing your news claim..."
                ):

                    prediction, raw_probability = (
                        truthlens.predict_news(news.strip())
                    )

                prediction = str(prediction).strip().upper()
                probability = float(raw_probability)

                if not math.isfinite(probability):
                    raise ValueError(
                        "The model returned an invalid probability."
                    )

                if prediction not in ["REAL", "FAKE"]:
                    raise ValueError(
                        f"Unexpected prediction: {prediction}"
                    )

                if not 0 <= probability <= 100:
                    raise ValueError(
                        "The model probability is outside the "
                        "expected percentage range."
                    )

                result_data = {
                    "prediction": prediction,
                    "probability": probability,
                    "article": news.strip(),
                    "timestamp": datetime.now().strftime(
                        "%d %b %Y | %I:%M %p"
                    )
                }

                st.session_state.last_result = result_data

                st.session_state.analysis_history.insert(
                    0, result_data
                )

                # Keep only the latest 10 analyses.
                st.session_state.analysis_history = (
                    st.session_state.analysis_history[:10]
                )

                st.session_state.analysis_error = None

            except Exception as error:

                st.session_state.last_result = None
                st.session_state.analysis_error = str(error)

    # -----------------------------------------------------
    # DISPLAY ERRORS
    # -----------------------------------------------------

    if st.session_state.analysis_error:

        st.error(
            "Something went wrong while analyzing the article."
        )

        with st.expander("Technical details"):
            st.code(st.session_state.analysis_error)

    # -----------------------------------------------------
    # DISPLAY PREDICTION
    # -----------------------------------------------------

    result = st.session_state.last_result

    if result:

        st.markdown("---")

        st.markdown(
            "<div class='result-animation'>",
            unsafe_allow_html=True
        )

        st.markdown("## 📊 Analysis Results")

        if result["prediction"] == "REAL":

            st.success(
                "Model Prediction: REAL",
                icon="✅"
            )

        else:

            st.error(
                "Model Prediction: FAKE",
                icon="⚠️"
            )

        metric_col1, metric_col2 = st.columns(2)

        with metric_col1:

            st.metric(
                label="Predicted-Class Probability",
                value=f"{result['probability']:.2f}%"
            )

        with metric_col2:

            st.metric(
                label="Article Length",
                value=f"{len(result['article']):,} characters"
            )

        st.markdown("#### Predicted-class probability")

        st.progress(result["probability"] / 100)

        st.caption(
            "This percentage is the model's estimated probability "
            "for the predicted class. It is not a verified measure "
            "of factual accuracy."
        )

        st.info(
            "Important: This prediction is generated by a "
            "machine learning model. It is not independent "
            "fact-checking or proof that a claim is true or false. "
            "Always verify important claims using reliable "
            "sources and official university announcements."
        )

        with st.expander("📰 View analyzed article"):

            st.write(result["article"])

            st.caption(
                f"Analyzed on: {result['timestamp']}"
            )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

    # -----------------------------------------------------
    # ANALYSIS HISTORY
    # -----------------------------------------------------

    st.markdown("---")

    st.markdown("## 🕘 Recent Analysis History")

    history = st.session_state.analysis_history

    if history:

        st.caption(
            f"Showing your latest {len(history)} analyses "
            "from this session."
        )

        for item in history:

            with st.container(border=True):

                history_col1, history_col2 = st.columns([3, 1])

                with history_col1:

                    st.markdown(
                        f"**{item['prediction']}** "
                        f"· {item['timestamp']}"
                    )

                    preview = item["article"]

                    if len(preview) > 130:
                        preview = preview[:130] + "..."

                    st.caption(preview)

                with history_col2:

                    st.metric(
                        "Probability",
                        f"{item['probability']:.2f}%"
                    )

        st.button(
            "🗑️ Clear Analysis History",
            on_click=clear_history
        )

    else:

        st.caption(
            "Your previous analyses will appear here."
        )


# =========================================================
# TAB 2 — HOW IT WORKS
# =========================================================

with how_tab:

    st.markdown("## ⚙️ How TruthLens Works")

    st.write(
        "TruthLens uses Natural Language Processing and "
        "Machine Learning to analyze the text of news claims."
    )

    steps = [
        (
            "01",
            "📰 News Input",
            "The user enters a news article or university-related claim."
        ),
        (
            "02",
            "🔤 Text Processing",
            "The text is transformed into numerical features "
            "using TF-IDF vectorization."
        ),
        (
            "03",
            "🧠 Machine Learning",
            "The Logistic Regression model evaluates the "
            "processed text using learned patterns."
        ),
        (
            "04",
            "📊 Prediction",
            "The system returns a REAL or FAKE prediction "
            "along with the model's estimated probability "
            "for that predicted class."
        )
    ]

    for number, title, description in steps:

        with st.container(border=True):

            col1, col2 = st.columns([1, 5])

            with col1:

                st.markdown(
                    f"""
                    <div style="
                        background:#e0f5ff;
                        border-radius:13px;
                        padding:15px 10px;
                        text-align:center;
                        color:#0284c7;
                        font-size:23px;
                        font-weight:800;">
                        {number}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col2:

                st.markdown(f"### {title}")

                st.write(description)

    st.markdown("---")

    st.markdown("### 🛠️ Technologies Used")

    tech_col1, tech_col2 = st.columns(2)

    with tech_col1:

        with st.container(border=True):

            st.markdown("### 🐍 Python")

            st.write(
                "Main programming language used to develop "
                "the TruthLens system."
            )

        with st.container(border=True):

            st.markdown("### 📊 TF-IDF")

            st.write(
                "Converts text into numerical features "
                "for machine learning."
            )

    with tech_col2:

        with st.container(border=True):

            st.markdown("### 🤖 Scikit-learn")

            st.write(
                "Provides the machine learning tools "
                "used by the prediction pipeline."
            )

        with st.container(border=True):

            st.markdown("### ⚙️ Logistic Regression")

            st.write(
                "Classification algorithm used to predict "
                "the REAL or FAKE label."
            )

    st.warning(
        "The system identifies patterns learned from its training "
        "dataset. It does not independently verify sources, dates, "
        "or factual accuracy."
    )


# =========================================================
# TAB 3 — ABOUT THE PROJECT
# =========================================================

with about_tab:

    st.markdown("## 📘 About TruthLens")

    st.write(
        "TruthLens is an academic Artificial Intelligence "
        "project focused on fake news detection."
    )

    with st.container(border=True):

        st.markdown("### 🎯 Project Mission")

        st.write(
            "To develop a machine learning-based system that "
            "analyzes Bahria University-related news claims "
            "and helps users identify potentially misleading "
            "information."
        )

    with st.container(border=True):

        st.markdown("### 💡 Core Features")

        st.markdown("""
        - Natural Language Processing
        - TF-IDF text vectorization
        - Logistic Regression classification
        - REAL and FAKE predictions
        - Model probability display
        - Interactive news analysis
        - Session-based analysis history
        """)

    with st.container(border=True):

        st.markdown("### ⚠️ Project Limitations")

        st.write(
            "The quality of predictions depends on the "
            "training dataset, text preprocessing, and "
            "machine learning model. A prediction should "
            "not be treated as a verified fact-check."
        )

    st.markdown("")

    st.markdown(
        """
        <div style="
            text-align:center;
            padding:25px;
            background:#eaf7ff;
            border:1px solid #d4ecfb;
            border-radius:15px;
        ">
            <div style="font-size:35px;">🔎</div>
            <h2 style="margin:5px;color:#12304a;">
                TruthLens
            </h2>
            <p style="color:#56738c;">
                Making university-related news analysis smarter.
            </p>
            <span style="
                background:#d6f1ff;
                color:#0284c7;
                padding:7px 13px;
                border-radius:20px;
                font-size:12px;
                font-weight:700;
            ">
                AI PROJECT 2026
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# FOOTER
# =========================================================


st.markdown("""
<div class="app-footer">
<div style="font-size:17px; margin-bottom:7px;">
🔎 <span class="footer-brand">TruthLens</span>
</div>
AI-Powered Fake News Detection System
<br>
Built with Python, Scikit-learn and Streamlit
<br><br>
© 2026 TruthLens | Bahria University AI Project
</div>
""", unsafe_allow_html=True)
