import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from sentiment.config import LSTM_MODEL_PATH, TOKENIZER_PATH  # noqa: E402
from sentiment.predict import get_predictor, load_metadata  # noqa: E402
from sentiment.preprocessing import clean_text  # noqa: E402

st.set_page_config(
    page_title="IMDB Sentiment Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
)

SAMPLE_REVIEWS = {
    "Positive": (
        "This movie was fantastic. Brilliant acting, great soundtrack, "
        "and the story kept me hooked from start to finish."
    ),
    "Mixed": (
        "Some great moments but the pacing dragged in the second act. "
        "The lead is good, the script less so. Worth a rental."
    ),
    "Negative": (
        "Painfully boring and predictable. The dialogue is wooden and "
        "the plot makes no sense. I wanted my two hours back."
    ),
}


def models_ready():
    return TOKENIZER_PATH.exists() and LSTM_MODEL_PATH.exists()


def render_setup_help():
    st.warning(
        "Models not trained yet. Run these from the project root:\n\n"
        "```bash\n"
        "pip install -r requirements-dev.txt\n"
        "python scripts/download_data.py\n"
        "python scripts/train_model.py --models lstm\n"
        "```\n\n"
        "Then refresh this page."
    )


def render_prediction(prediction, text):
    label = prediction.label
    confidence = prediction.score
    p_pos = prediction.probability_positive
    color = "#16a34a" if label == "positive" else "#dc2626"

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        st.markdown(
            f"<div style='padding:1rem;border-radius:8px;background:{color};"
            f"color:white;text-align:center;'>"
            f"<div style='font-size:0.85rem;opacity:0.85;'>SENTIMENT</div>"
            f"<div style='font-size:1.6rem;font-weight:600;'>{label.upper()}</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.metric("Confidence", f"{confidence:.1%}")
    with col3:
        st.markdown("**Probability of positive**")
        st.progress(float(p_pos))
        st.caption(f"{p_pos:.4f}")

    with st.expander("Cleaned text fed to model"):
        st.code(clean_text(text) or "(empty after cleaning)")


def main():
    st.title("IMDB Movie Review Sentiment Analyzer")
    st.markdown(
        "Classify movie reviews as positive or negative using deep "
        "learning models trained on 50,000 IMDB reviews with GloVe word "
        "embeddings."
    )

    metadata = load_metadata()
    available_models = list(metadata.get("models", {}).keys()) or ["lstm"]

    with st.sidebar:
        st.header("Model")
        model_name = st.selectbox(
            "Architecture",
            options=available_models,
            index=available_models.index("lstm") if "lstm" in available_models else 0,
        )

        if metadata.get("models", {}).get(model_name):
            r = metadata["models"][model_name]
            st.metric("Test accuracy", f"{r['test_accuracy']:.2%}")
            st.metric("Test loss", f"{r['test_loss']:.4f}")
            st.caption(f"Vocab size: {metadata.get('vocab_size', 'n/a'):,}")

        st.divider()
        st.header("Sample reviews")
        for name, sample in SAMPLE_REVIEWS.items():
            if st.button(name, use_container_width=True, key=f"sample-{name}"):
                st.session_state["review_text"] = sample

    if not models_ready():
        render_setup_help()
        return

    text = st.text_area(
        "Paste a movie review",
        value=st.session_state.get("review_text", ""),
        height=180,
        placeholder="Type or paste a movie review here...",
        key="review_input",
    )

    col_a, col_b = st.columns([1, 5])
    predict_clicked = col_a.button("Analyze", type="primary", use_container_width=True)
    col_b.caption("Predictions run on CPU.")

    if predict_clicked:
        if not text.strip():
            st.error("Please enter a review.")
            return
        try:
            predictor = get_predictor(model_name)
            with st.spinner("Analyzing ..."):
                prediction = predictor.predict(text)
        except FileNotFoundError as e:
            st.error(str(e))
            return
        render_prediction(prediction, text)

    st.divider()
    st.subheader("Batch prediction")
    uploaded = st.file_uploader(
        "Upload a CSV with a 'Review Text' column",
        type=["csv"],
    )
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            return

        text_col = "Review Text" if "Review Text" in df.columns else (
            "review" if "review" in df.columns else None
        )
        if text_col is None:
            st.error("CSV must contain a 'Review Text' or 'review' column.")
            return

        predictor = get_predictor(model_name)
        with st.spinner(f"Predicting {len(df)} rows ..."):
            preds = predictor.predict_batch(df[text_col].astype(str).tolist())
        df_out = df.copy()
        df_out["predicted_label"] = [p.label for p in preds]
        df_out["confidence"] = [round(p.score, 4) for p in preds]
        df_out["probability_positive"] = [round(p.probability_positive, 4) for p in preds]
        st.dataframe(df_out, use_container_width=True)
        st.download_button(
            "Download predictions CSV",
            df_out.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
