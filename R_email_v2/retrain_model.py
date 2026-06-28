import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import re
import syllables
import joblib
import sklearn

print(f"Retraining with sklearn {sklearn.__version__}")

enron_ft = pd.read_csv("../data/enron_model_ft.csv", low_memory=False)
if "Unnamed: 0" in enron_ft.columns:
    enron_ft = enron_ft.drop(columns=["Unnamed: 0"])

X = enron_ft[["EmailSend"]]
y = enron_ft["Response_Within_3_hours"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def extract_text_metadata(dataframe):
    df_features = dataframe.copy()
    df_features["char_count"] = df_features["EmailSend"].str.len()
    df_features["has_question"] = df_features["EmailSend"].str.contains(r"\?").astype(int)
    df_features["exclamation_count"] = df_features["EmailSend"].str.count("!")
    words_series = df_features["EmailSend"].str.lower().str.split()
    df_features["token_count"] = words_series.apply(len)
    df_features["avg_word_length"] = words_series.apply(
        lambda words: np.mean([len(w) for w in words]) if len(words) > 0 else 0
    )
    df_features["stop_word_count"] = words_series.apply(
        lambda words: sum(1 for w in words if w in ENGLISH_STOP_WORDS)
    )
    def fast_syllables(text):
        if pd.isna(text):
            return 0
        words = re.findall(r"\b\w+\b", str(text).lower())
        return sum(syllables.estimate(w) for w in words)
    df_features["syllable_count"] = df_features["EmailSend"].apply(fast_syllables)
    return df_features

df_enriched = extract_text_metadata(X_train.copy())

num_features = ["char_count", "exclamation_count", "token_count",
                "avg_word_length", "stop_word_count", "syllable_count"]

preprocessor = ColumnTransformer(
    transformers=[
        ("text_vec", CountVectorizer(stop_words="english", lowercase=True), "EmailSend"),
        ("num_scale", StandardScaler(), num_features)
    ],
    remainder="passthrough"
)

pipeline = Pipeline([
    ("prep", preprocessor),
    ("clf", LogisticRegression(max_iter=1000))
])

all_feature_cols = ["EmailSend", "has_question"] + num_features
pipeline.fit(df_enriched[all_feature_cols], y_train)

joblib.dump(pipeline, "enron_model.pkl")
print(f"Saved enron_model.pkl with sklearn {sklearn.__version__}")
