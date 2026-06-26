import streamlit as st
import joblib
import numpy as np
import pandas as pd
import re
import syllables
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

st.set_page_config(page_title="Email Response Optimizer", page_icon="📧", layout="centered")

st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 760px; }
    .metric-card {
        background: #f8f9fb;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .metric-label { font-size: 0.78rem; color: #64748b; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 0.3rem; }
    .metric-value { font-size: 2.2rem; font-weight: 700; color: #1e293b; line-height: 1; }
    .metric-sub { font-size: 0.85rem; color: #94a3b8; margin-top: 0.25rem; }
    .verdict-high  { color: #16a34a; }
    .verdict-mod   { color: #d97706; }
    .verdict-low   { color: #dc2626; }
    .section-header { font-size: 0.78rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: #94a3b8; margin: 1.5rem 0 0.6rem; }
    .stTextArea textarea { border-radius: 10px; font-size: 0.95rem; }
    .stButton > button { border-radius: 10px; font-weight: 600; font-size: 1rem; padding: 0.65rem 1rem; }
    div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load("enron_model.pkl")


model = load_model()

ALL_FEATURE_COLS = ['EmailSend', 'has_question', 'char_count', 'exclamation_count',
                    'token_count', 'avg_word_length', 'stop_word_count', 'syllable_count']


def extract_text_metadata(dataframe):
    df = dataframe.copy()
    df['char_count'] = df['EmailSend'].str.len()
    df['has_question'] = df['EmailSend'].str.contains(r'\?').astype(int)
    df['exclamation_count'] = df['EmailSend'].str.count('!')
    words_series = df['EmailSend'].str.lower().str.split()
    df['token_count'] = words_series.apply(len)
    df['avg_word_length'] = words_series.apply(
        lambda w: np.mean([len(x) for x in w]) if len(w) > 0 else 0
    )
    df['stop_word_count'] = words_series.apply(
        lambda w: sum(1 for x in w if x in ENGLISH_STOP_WORDS)
    )
    def count_syllables(text):
        if pd.isna(text):
            return 0
        return sum(syllables.estimate(w) for w in re.findall(r'\b\w+\b', str(text).lower()))
    df['syllable_count'] = df['EmailSend'].apply(count_syllables)
    return df


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 📧 Email Response Optimizer")
st.markdown(
    "Predict the probability your email gets a reply **within 3 hours**, "
    "based on a logistic regression model trained on the Enron corpus."
)

st.markdown("---")

# ── Input ─────────────────────────────────────────────────────────────────────
email_text = st.text_area(
    "**Email Body**",
    placeholder="Paste your email here...",
    height=180,
    label_visibility="visible"
)

predict_btn = st.button("Predict Response Probability", type="primary", use_container_width=True)

# ── Results ───────────────────────────────────────────────────────────────────
if predict_btn:
    if not email_text.strip():
        st.warning("Please enter some email text before predicting.")
    else:
        with st.spinner("Analyzing..."):
            new_data = pd.DataFrame({'EmailSend': [email_text]})
            enriched = extract_text_metadata(new_data)
            proba = model.predict_proba(enriched[ALL_FEATURE_COLS])[0]
            prob = proba[1]

        if prob >= 0.6:
            verdict, verdict_class, tip = "High", "verdict-high", "Strong signal — clear, concise language tends to prompt fast replies."
        elif prob >= 0.35:
            verdict, verdict_class, tip = "Moderate", "verdict-mod", "Room to improve — try a direct question or shorter phrasing."
        else:
            verdict, verdict_class, tip = "Low", "verdict-low", "Low signal — consider making the ask more explicit and concise."

        st.markdown("---")
        st.markdown('<p class="section-header">Prediction</p>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Response Probability</div>
                <div class="metric-value">{prob:.0%}</div>
                <div class="metric-sub">within 3 hours</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Likelihood</div>
                <div class="metric-value {verdict_class}">{verdict}</div>
                <div class="metric-sub">&nbsp;</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"> {tip}")

        # Progress bar
        st.markdown('<p class="section-header">Confidence</p>', unsafe_allow_html=True)
        st.progress(float(prob))
        c1, c2 = st.columns(2)
        c1.caption("No response")
        c2.caption(f"Response &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; {prob:.1%}")

        # Probability bar chart
        st.markdown('<p class="section-header">Probability Distribution</p>', unsafe_allow_html=True)
        prob_df = pd.DataFrame({
            "Outcome": ["No Response", "Response within 3 hrs"],
            "Probability": [round(proba[0], 4), round(proba[1], 4)]
        }).set_index("Outcome")
        st.bar_chart(prob_df)

        # Feature breakdown
        st.markdown('<p class="section-header">Email Feature Breakdown</p>', unsafe_allow_html=True)
        row = enriched.iloc[0]
        feature_df = pd.DataFrame({
            "Feature": ["Character Count", "Word Count", "Avg Word Length",
                        "Syllable Count", "Stop Word Count", "Has Question?", "Exclamation Marks"],
            "Value": [
                int(row['char_count']),
                int(row['token_count']),
                f"{row['avg_word_length']:.2f}",
                int(row['syllable_count']),
                int(row['stop_word_count']),
                "Yes" if row['has_question'] == 1 else "No",
                int(row['exclamation_count']),
            ]
        })
        st.dataframe(feature_df, use_container_width=True, hide_index=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("About this model"):
    st.markdown("""
**Model:** Logistic Regression with CountVectorizer + StandardScaler

**Training data:** Enron email corpus (matched send/reply pairs)

**Target:** `Response_Within_3_hours` — 1 if recipient replied within 3 hours

**Features:**
- Email text (bag-of-words via CountVectorizer)
- Character count, word count, syllable count, avg word length
- Stop word count, question mark presence, exclamation count
    """)
