# ============================================================
# ASPECTIQ
# Aspect-Based Emotion Analysis for Product Reviews
# STREAMLIT DASHBOARD
#
# Files that must sit next to this file:
#   app.py, engine.py, text_utils.py, train_models.py
#   emotion_dataset.csv, model_comparison.csv   (created by train_models.py)
#   models/sentiment_model.pkl, models/emotion_model.pkl  (same)
# ============================================================

import os
import re
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------

st.set_page_config(
    page_title="AspectIQ | Product Emotion Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

# The saved pipelines call functions from text_utils.py, so it must be importable.
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import engine  # noqa: E402
from text_utils import ASPECT_CATEGORIES  # noqa: E402

# ------------------------------------------------------------
# MODERN UI
# ------------------------------------------------------------

st.markdown(
    """
    <style>
    .stApp {
        background:
        radial-gradient(circle at 10% 5%, rgba(99,102,241,.15), transparent 30%),
        radial-gradient(circle at 90% 5%, rgba(168,85,247,.12), transparent 30%),
        #070b16;
    }

    .main .block-container {
        max-width: 1450px;
        padding-top: 1.5rem;
        padding-bottom: 4rem;
    }

    section[data-testid="stSidebar"] {
        background: #080d1a;
        border-right: 1px solid rgba(255,255,255,.08);
    }

    h1 {
        font-size: 42px !important;
        font-weight: 800 !important;
        letter-spacing: -1.5px;
    }

    h2, h3 {
        font-weight: 750 !important;
    }

    div[data-testid="metric-container"] {
        background: rgba(255,255,255,.045);
        border: 1px solid rgba(255,255,255,.08);
        border-radius: 18px;
        padding: 18px;
    }

    .stButton > button {
        border-radius: 12px;
        min-height: 44px;
        font-weight: 700;
    }

    textarea {
        border-radius: 14px !important;
    }

    .footer {
        text-align: center;
        color: #64748b;
        padding: 30px;
        font-size: 13px;
    }

    .small-note {
        color: #94a3b8;
        font-size: 13px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# FILE DISCOVERY
# ------------------------------------------------------------

def find_file(names):
    for name in names:
        root_path = BASE_DIR / name
        if root_path.exists():
            return root_path

        model_path = MODELS_DIR / name
        if model_path.exists():
            return model_path

    return None


def find_dataset():
    names = [
        "emotion_dataset.csv",
        "emotion_analysis_dataset.csv",
        "emotion_analysis.csv",
        "dataset.csv",
        "reviews.csv",
    ]

    path = find_file(names)
    if path:
        return path

    csv_files = sorted(BASE_DIR.glob("*.csv"))
    return csv_files[0] if csv_files else None


def find_emotion_model():
    return find_file(
        [
            "emotion_model.pkl",
            "emotion_pipeline.pkl",
            "emotion_classifier.pkl",
            "emotion_model.joblib",
        ]
    )


def find_sentiment_model():
    return find_file(
        [
            "sentiment_model.pkl",
            "sentiment_pipeline.pkl",
            "sentiment_classifier.pkl",
            "sentiment_model.joblib",
        ]
    )


# ------------------------------------------------------------
# DATASET
# ------------------------------------------------------------

@st.cache_data
def load_dataset():
    path = find_dataset()

    if path is None:
        return None, None, "No CSV dataset found."

    try:
        df = pd.read_csv(path)
        return df, path.name, None
    except Exception as exc:
        return None, None, str(exc)


data, dataset_name, dataset_error = load_dataset()


def normalize_columns(df):
    if df is None:
        return None

    df = df.copy()

    rename = {}
    lookup = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    aliases = {
        "review": [
            "review",
            "sentence",
            "review text",
            "text",
            "customer review",
        ],
        "aspect": [
            "aspect",
            "aspect term",
            "aspect_term",
            "feature",
            "product aspect",
        ],
        "category": [
            "aspect category",
            "aspect_category",
            "category",
            "aspect type",
        ],
        "sentiment": [
            "sentiment",
            "polarity",
            "sentiment label",
        ],
        "emotion": [
            "emotion",
            "emotion label",
            "emotion_category",
            "emotion category",
        ],
    }

    for standard, possible_names in aliases.items():
        for possible in possible_names:
            if possible in lookup:
                rename[lookup[possible]] = standard
                break

    return df.rename(columns=rename)


data = normalize_columns(data)

# ------------------------------------------------------------
# MODEL LOADING
# ------------------------------------------------------------

@st.cache_resource
def load_models():
    emotion_path = find_emotion_model()
    sentiment_path = find_sentiment_model()

    emotion = None
    sentiment = None
    emotion_error = None
    sentiment_error = None

    if emotion_path:
        try:
            emotion = joblib.load(emotion_path)
        except Exception as exc:
            emotion_error = str(exc)

    if sentiment_path:
        try:
            sentiment = joblib.load(sentiment_path)
        except Exception as exc:
            sentiment_error = str(exc)

    return (
        emotion,
        sentiment,
        emotion_path,
        sentiment_path,
        emotion_error,
        sentiment_error,
    )


(
    emotion_model,
    sentiment_model,
    emotion_model_path,
    sentiment_model_path,
    emotion_error,
    sentiment_error,
) = load_models()

# ------------------------------------------------------------
# PREDICTION ENGINE (see engine.py and text_utils.py)
#
# Everything that decides sentiment/emotion lives in engine.py, and all
# text preprocessing lives inside the saved model pipelines
# (text_utils.py).  The app therefore cannot preprocess text differently
# from training, which was the main cause of wrong predictions before.
# ------------------------------------------------------------

EMOTION_SCORE = engine.EMOTION_SCORE
DATASET_ASPECT_TERMS = engine.dataset_terms_from(data)

MODELS_ARE_CURRENT = (
    emotion_model is not None
    and sentiment_model is not None
    and engine.is_current_model(emotion_model)
    and engine.is_current_model(sentiment_model)
)

OLD_MODEL_MESSAGE = (
    "The saved models in the models/ folder were created by the OLD notebook "
    "pipeline and must not be used with this app. Run  python train_models.py  "
    "once (it takes under a minute), then restart the app."
)


def analyze_review(review):
    return engine.analyze_review(
        review,
        sentiment_model,
        emotion_model,
        DATASET_ASPECT_TERMS,
    )


# ------------------------------------------------------------
# MODEL INFORMATION
# ------------------------------------------------------------

def estimator_name(model):
    if model is None:
        return "Not loaded"

    try:
        if hasattr(model, "named_steps"):
            last_name = list(model.named_steps.keys())[-1]
            estimator = model.named_steps[last_name]
            return type(estimator).__name__

        return type(model).__name__

    except Exception:
        return type(model).__name__


def vectorizer_summary(model):
    if model is None:
        return "Not available"

    try:
        tfidf = model.named_steps.get("tfidf")

        if hasattr(tfidf, "transformer_list"):
            parts = []

            for name, transformer in tfidf.transformer_list:
                parts.append(
                    f"{name}: {type(transformer).__name__}"
                )

            return " | ".join(parts)

        return type(tfidf).__name__

    except Exception:
        return "Not available"


# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------

with st.sidebar:
    st.title("🧠 AspectIQ")
    st.caption("PRODUCT EMOTION INTELLIGENCE")

    st.divider()

    page = st.radio(
        "WORKSPACE",
        [
            "🏠 Overview",
            "🔍 Review Intelligence",
            "😊 Emotion Analytics",
            "🎯 Aspect Analytics",
            "🤖 Model Performance",
        ],
    )

    st.divider()

    if MODELS_ARE_CURRENT:
        st.success("⚡ AI ENGINE ONLINE")
    elif emotion_model is not None and sentiment_model is not None:
        st.error("⚠️ OLD MODELS — run train_models.py")
    elif emotion_model is not None or sentiment_model is not None:
        st.warning("⚠️ PARTIAL MODEL STATUS")
    else:
        st.error("❌ MODELS NOT FOUND")

    st.divider()

    if data is not None:
        st.caption(f"📁 {dataset_name}")
        st.caption(f"📊 {len(data):,} records")
    else:
        st.caption("📁 Dataset not found")

    st.divider()

    st.caption("MSc Data Analytics")
    st.caption("Aspect-Based Emotion Analysis")


# ============================================================
# OVERVIEW
# ============================================================

if page == "🏠 Overview":

    st.title("Understand what customers really feel.")

    st.write(
        "AspectIQ identifies product aspects, sentiment and "
        "associated customer emotions from product reviews."
    )

    st.divider()

    total_records = len(data) if data is not None else 0

    if data is not None and "category" in data.columns:
        total_categories = data["category"].nunique()
    else:
        total_categories = len(ASPECT_CATEGORIES)

    if data is not None and "emotion" in data.columns:
        total_emotions = data["emotion"].nunique()
    else:
        total_emotions = len(EMOTION_SCORE)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Reviews / Records", f"{total_records:,}")
    c2.metric("Aspect Categories", total_categories)
    c3.metric("Emotion Classes", total_emotions)

    if emotion_model is not None:
        c4.metric("Emotion Model", estimator_name(emotion_model))
    else:
        c4.metric("Emotion Model", "Not loaded")

    st.divider()

    st.subheader("🔄 Project Pipeline")

    st.code(
        """
Customer Review
      ↓
Aspect Detection
      ↓
Aspect-Specific Context
      ↓
TF-IDF / Saved Text Pipeline
      ↓
Sentiment Classification
      ↓
Emotion Classification
      ↓
Emotion Spectrum
      ↓
Aspect–Emotion Insight
      ↓
Streamlit Dashboard
        """,
        language="text",
    )

    st.divider()

    if data is not None:

        col1, col2 = st.columns(2)

        if "emotion" in data.columns:
            emotion_counts = (
                data["emotion"]
                .astype(str)
                .str.title()
                .value_counts()
                .reset_index()
            )
            emotion_counts.columns = ["Emotion", "Count"]

            with col1:
                st.subheader("😊 Emotion Distribution")

                fig = px.bar(
                    emotion_counts,
                    x="Emotion",
                    y="Count",
                    text="Count",
                    template="plotly_dark",
                )

                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=420,
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

        if "category" in data.columns:
            category_counts = (
                data["category"]
                .astype(str)
                .value_counts()
                .head(12)
                .reset_index()
            )
            category_counts.columns = ["Category", "Count"]

            with col2:
                st.subheader("🎯 Aspect Categories")

                fig = px.bar(
                    category_counts,
                    x="Count",
                    y="Category",
                    orientation="h",
                    text="Count",
                    template="plotly_dark",
                )

                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=420,
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )


# ============================================================
# REVIEW INTELLIGENCE
# ============================================================

elif page == "🔍 Review Intelligence":

    st.title("🔍 Review Intelligence")

    st.write(
        "Enter a customer review to identify product aspects "
        "and the emotions associated with them."
    )

    st.divider()

    examples = {
        "✨ Display + Battery":
            "The display is amazing but the battery drains very quickly.",

        "📷 Camera + Design":
            "The camera quality is excellent and the design looks premium.",

        "⚠️ Multiple Problems":
            "The laptop is slow and the speakers are terrible. The price is too high.",
    }

    st.subheader("💡 Try a sample review")

    b1, b2, b3 = st.columns(3)

    for column, label in [
        (b1, "✨ Display + Battery"),
        (b2, "📷 Camera + Design"),
        (b3, "⚠️ Multiple Problems"),
    ]:
        with column:
            if st.button(
                label,
                use_container_width=True,
            ):
                st.session_state["selected_review"] = examples[label]

    review = st.text_area(
        "PRODUCT REVIEW",
        value=st.session_state.get(
            "selected_review",
            "",
        ),
        height=180,
        placeholder=(
            "Example: I love the display, "
            "but the battery drains quickly."
        ),
    )

    analyze = st.button(
        "🚀 Analyze Customer Emotion",
        type="primary",
        use_container_width=True,
    )

    if analyze:

        if not review.strip():
            st.warning("Please enter a review first.")
            st.stop()

        if emotion_model is None or sentiment_model is None:
            st.error(
                "Both sentiment_model.pkl and emotion_model.pkl "
                "are required for prediction. Run  python train_models.py."
            )
            st.stop()

        if not MODELS_ARE_CURRENT:
            st.error(OLD_MODEL_MESSAGE)
            st.stop()

        with st.spinner("Analyzing review..."):
            result = analyze_review(review)

        if result.empty:
            st.warning("No product aspect was detected.")
            st.info(
                "Try mentioning battery, display, camera, "
                "keyboard, price, performance, sound or design."
            )
            st.stop()

        st.success(
            f"Analysis completed — {len(result)} aspect(s) detected."
        )

        st.divider()

        # Overall summary
        valid = result[
            result["Sentiment"] != "Prediction Error"
        ].copy()

        if valid.empty:
            st.error(
                "The models could not produce valid predictions."
            )
            st.dataframe(
                result,
                use_container_width=True,
                hide_index=True,
            )
            st.stop()

        dominant_sentiment = (
            valid["Sentiment"]
            .value_counts()
            .index[0]
        )

        dominant_emotion = (
            valid["Emotion"]
            .value_counts()
            .index[0]
        )

        average_score = valid["Emotion Score"].mean()

        s1, s2, s3, s4 = st.columns(4)

        s1.metric(
            "🎯 Aspects",
            len(valid),
        )

        s2.metric(
            "💭 Dominant Sentiment",
            dominant_sentiment,
        )

        s3.metric(
            "😊 Dominant Emotion",
            dominant_emotion,
        )

        s4.metric(
            "🎚️ Spectrum Score",
            f"{average_score:.2f}",
        )

        st.divider()

        st.subheader("🎯 Aspect-Level Results")

        display_result = valid[
            [
                "Aspect",
                "Category",
                "Context",
                "Sentiment",
                "Emotion",
                "Emotion Score",
                "Confidence",
            ]
        ]

        st.dataframe(
            display_result,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Context is the part of the review about that aspect. "
            "Confidence is the sentiment model's probability for its "
            "answer - values below about 0.6 mean the model is unsure."
        )

        with st.expander("🔬 Model Debug Information"):
            debug_result = valid[
                [
                    "Aspect",
                    "Model Input",
                    "Sentiment Probabilities",
                ]
            ]

            st.dataframe(
                debug_result,
                use_container_width=True,
                hide_index=True,
            )

        st.divider()

        c1, c2 = st.columns(2)

        with c1:
            st.subheader("😊 Emotion by Aspect")

            fig = px.bar(
                valid,
                x="Aspect",
                y="Emotion Score",
                color="Emotion",
                text="Emotion Score",
                template="plotly_dark",
            )

            fig.add_hline(
                y=0,
                line_dash="dash",
            )

            fig.update_layout(
                height=430,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        with c2:
            st.subheader("💭 Sentiment Distribution")

            sentiment_counts = (
                valid["Sentiment"]
                .value_counts()
                .reset_index()
            )

            sentiment_counts.columns = [
                "Sentiment",
                "Count",
            ]

            fig = px.pie(
                sentiment_counts,
                names="Sentiment",
                values="Count",
                hole=0.55,
                template="plotly_dark",
            )

            fig.update_layout(
                height=430,
                paper_bgcolor="rgba(0,0,0,0)",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        st.subheader("🎚️ Emotion Spectrum")

        spectrum = pd.DataFrame(
            {
                "Emotion": [
                    "Anger",
                    "Frustration",
                    "Disappointment",
                    "Neutral",
                    "Satisfaction",
                    "Joy",
                ],
                "Score": [
                    -3,
                    -2,
                    -1,
                    0,
                    1,
                    2,
                ],
            }
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=spectrum["Score"],
                y=[0] * len(spectrum),
                mode="lines+markers",
                text=spectrum["Emotion"],
                hovertemplate=(
                    "<b>%{text}</b>"
                    "<br>Score: %{x}"
                    "<extra></extra>"
                ),
                marker={"size": 16},
                line={"width": 7},
            )
        )

        fig.add_vline(
            x=average_score,
            line_dash="dash",
            annotation_text=(
                f"Current Score: {average_score:.2f}"
            ),
        )

        fig.update_layout(
            template="plotly_dark",
            height=300,
            xaxis={
                "range": [-3.5, 2.5],
                "title": "Emotion Intensity",
            },
            yaxis={"visible": False},
            paper_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        st.download_button(
            "⬇️ Download Analysis CSV",
            data=valid.to_csv(index=False),
            file_name="aspect_emotion_result.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ============================================================
# EMOTION ANALYTICS
# ============================================================

elif page == "😊 Emotion Analytics":

    st.title("😊 Emotion Analytics")

    st.write(
        "Explore the emotion distribution in the complete dataset."
    )

    st.divider()

    if data is None or "emotion" not in data.columns:
        st.error(
            "The dataset does not contain an Emotion column."
        )
        st.stop()

    emotion_counts = (
        data["emotion"]
        .astype(str)
        .str.title()
        .value_counts()
        .reset_index()
    )

    emotion_counts.columns = [
        "Emotion",
        "Count",
    ]

    total = emotion_counts["Count"].sum()

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Total Records",
        f"{total:,}",
    )

    c2.metric(
        "Emotion Classes",
        len(emotion_counts),
    )

    c3.metric(
        "Dominant Emotion",
        emotion_counts.iloc[0]["Emotion"],
    )

    st.divider()

    fig = px.bar(
        emotion_counts,
        x="Emotion",
        y="Count",
        text="Count",
        template="plotly_dark",
    )

    fig.update_layout(
        height=500,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.subheader("🎚️ Emotion Spectrum Mapping")

    spectrum_table = pd.DataFrame(
        {
            "Emotion": [
                "Anger",
                "Frustration",
                "Disappointment",
                "Neutral",
                "Satisfaction",
                "Joy",
            ],
            "Score": [
                -3,
                -2,
                -1,
                0,
                1,
                2,
            ],
            "Interpretation": [
                "Very negative",
                "Negative",
                "Slightly negative",
                "Neutral",
                "Positive",
                "Very positive",
            ],
        }
    )

    st.dataframe(
        spectrum_table,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# ASPECT ANALYTICS
# ============================================================

elif page == "🎯 Aspect Analytics":

    st.title("🎯 Aspect Analytics")

    st.write(
        "Understand which product aspects occur most frequently "
        "and which emotions are associated with them."
    )

    st.divider()

    if data is None:
        st.error("Dataset could not be loaded.")
        st.stop()

    if "category" not in data.columns:
        st.error(
            "The dataset does not contain an Aspect Category column."
        )
        st.stop()

    aspect_counts = (
        data["category"]
        .astype(str)
        .value_counts()
        .reset_index()
    )

    aspect_counts.columns = [
        "Aspect",
        "Count",
    ]

    fig = px.bar(
        aspect_counts.head(15),
        x="Count",
        y="Aspect",
        orientation="h",
        text="Count",
        template="plotly_dark",
    )

    fig.update_layout(
        height=560,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    if "emotion" in data.columns:

        st.divider()

        st.subheader("🔥 Aspect × Emotion Heatmap")

        heatmap = pd.crosstab(
            data["category"],
            data["emotion"],
        )

        fig = px.imshow(
            heatmap,
            text_auto=True,
            aspect="auto",
            template="plotly_dark",
        )

        fig.update_layout(
            height=600,
            paper_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        st.subheader("🏆 Dominant Emotion by Aspect")

        dominant = (
            data.groupby("category")["emotion"]
            .agg(
                lambda x: x.value_counts().index[0]
            )
            .reset_index()
        )

        dominant.columns = [
            "Aspect",
            "Dominant Emotion",
        ]

        st.dataframe(
            dominant,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "🤖 Model Performance":

    st.title("🤖 Model Performance")

    st.write(
        "Model information and evaluation results used by AspectIQ."
    )

    st.divider()

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("😊 Emotion Model")

        if emotion_model is not None:
            st.success("Emotion model loaded")
            st.write(
                f"**Estimator:** {estimator_name(emotion_model)}"
            )
            st.write(
                f"**Features:** {vectorizer_summary(emotion_model)}"
            )
            st.write(
                f"**Trained by current pipeline:** "
                f"{engine.is_current_model(emotion_model)}"
            )
            st.code(str(emotion_model_path))
        else:
            st.error("Emotion model not found.")

    with c2:
        st.subheader("💭 Sentiment Model")

        if sentiment_model is not None:
            st.success("Sentiment model loaded")
            st.write(
                f"**Estimator:** {estimator_name(sentiment_model)}"
            )
            st.write(
                f"**Features:** {vectorizer_summary(sentiment_model)}"
            )
            st.write(
                f"**Trained by current pipeline:** "
                f"{engine.is_current_model(sentiment_model)}"
            )
            st.code(str(sentiment_model_path))
        else:
            st.error("Sentiment model not found.")

    if emotion_error:
        st.error(
            f"Emotion model loading error: {emotion_error}"
        )

    if sentiment_error:
        st.error(
            f"Sentiment model loading error: {sentiment_error}"
        )

    st.divider()

    # Existing model-comparison file, if available.
    comparison_path = find_file(
        ["model_comparison.csv"]
    )

    if comparison_path:
        st.subheader("📊 Model Comparison")

        try:
            comparison = pd.read_csv(comparison_path)

            if "Task" in comparison.columns:
                task = st.radio(
                    "Task",
                    list(comparison["Task"].unique()),
                    horizontal=True,
                )
                comparison = comparison[
                    comparison["Task"] == task
                ].drop(columns=["Task"])

            st.dataframe(
                comparison,
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                "Scores come from 5-fold cross-validation grouped by "
                "sentence, so the same sentence is never in both training "
                "and test data. Emotion labels are rule-generated, so the "
                "emotion score measures agreement with those rules."
            )

            numeric_cols = [
                c for c in [
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1-Score",
                    "F1 Score",
                ]
                if c in comparison.columns
            ]

            if numeric_cols and "Model" in comparison.columns:

                metric = (
                    "F1-Score"
                    if "F1-Score" in comparison.columns
                    else numeric_cols[-1]
                )

                chart_df = comparison.copy()

                # model_comparison.csv already stores percentages
                if chart_df[metric].max() <= 1.0:
                    chart_df[metric] = chart_df[metric] * 100

                fig = px.bar(
                    chart_df,
                    x="Model",
                    y=metric,
                    text=metric,
                    template="plotly_dark",
                    title=f"{metric} Comparison",
                )

                fig.update_layout(
                    height=450,
                    yaxis_title="Score (%)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

        except Exception as exc:
            st.warning(
                f"Could not read model_comparison.csv: {exc}"
            )

    st.divider()

    st.subheader("🧠 How the prediction pipeline works")

    st.code(
        """
Review
  ↓
Aspect Detection
  ↓
Clause around each aspect
  ↓
Negation-aware preprocessing (inside the saved model)
  ↓
TF-IDF + opinion-lexicon features
  ↓
Sentiment model  →  Emotion model
  ↓
Emotion restricted to those that fit the sentiment
  ↓
Final Aspect-Level Result
        """,
        language="text",
    )

    st.info(
        "No keyword overrides are applied any more: the saved ML models "
        "make every prediction. The only rule is that the emotion must fit "
        "the predicted sentiment (e.g. Negative can be Anger, Frustration "
        "or Disappointment, never Joy)."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="footer">
        🧠 AspectIQ &nbsp;•&nbsp;
        Aspect-Based Emotion Analysis for Product Reviews
        <br>
        MSc Data Analytics Project
    </div>
    """,
    unsafe_allow_html=True,
)
