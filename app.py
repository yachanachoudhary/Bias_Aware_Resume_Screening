import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Bias-Aware Resume Screening", layout="wide")
st.title("🧠 Bias-Aware Resume Screening System")
st.caption("Fair candidate screening using skills-based evaluation")

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/resumes.csv")

df = load_data()

st.subheader("📂 Original Dataset")
st.dataframe(df.head())

# -------------------------------------------------
# REMOVE BIAS-PRONE COLUMNS
# -------------------------------------------------
bias_columns = [
    "Name",
    "Recruiter Decision",
    "Salary Expectation ($)"
]

df_fair = df.drop(columns=bias_columns)

# -------------------------------------------------
# CREATE FAIR LABEL
# -------------------------------------------------
df_fair["Selected"] = df_fair["AI Score (0-100)"].apply(
    lambda x: 1 if x >= 70 else 0
)

# -------------------------------------------------
# FEATURE EXTRACTION (TF-IDF)
# -------------------------------------------------
X_text = df_fair["Skills"]
y = df_fair["Selected"]

vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
X = vectorizer.fit_transform(X_text)

# -------------------------------------------------
# TRAIN MODEL
# -------------------------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# -------------------------------------------------
# RANK CANDIDATES
# -------------------------------------------------
df_fair["Selection Probability"] = model.predict_proba(X)[:, 1]

ranked_df = df_fair.sort_values(
    "Selection Probability",
    ascending=False
)

# -------------------------------------------------
# DISPLAY RESULTS
# -------------------------------------------------
st.subheader("🏆 Fair Resume Ranking")

st.dataframe(
    ranked_df[
        [
            "Resume_ID",
            "Skills",
            "Experience (Years)",
            "Projects Count",
            "AI Score (0-100)",
            "Selection Probability"
        ]
    ]
)

# -------------------------------------------------
# INTERACTIVE FILTER
# -------------------------------------------------
st.subheader("🔍 Filter Top Candidates")

top_n = st.slider("Select Top N Candidates", 1, len(ranked_df), 5)

st.dataframe(
    ranked_df.head(top_n)[
        [
            "Resume_ID",
            "Skills",
            "Experience (Years)",
            "Projects Count",
            "Selection Probability"
        ]
    ]
)

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.success("✅ Screening completed without using biased attributes.")
