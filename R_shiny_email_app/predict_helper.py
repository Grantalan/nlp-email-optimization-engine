# Python helper — loaded by the R Shiny app via reticulate.
# Takes raw email text, runs the same feature engineering from the notebook,
# and returns the predicted probability of a response within 3 hours.

import joblib
import pandas as pd
import numpy as np
import re
import syllables
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Load the trained model once at startup
model = joblib.load("enron_model.pkl")

ALL_FEATURE_COLS = ['EmailSend', 'has_question', 'char_count', 'exclamation_count',
                    'token_count', 'avg_word_length', 'stop_word_count', 'syllable_count']


def extract_features(email_text):
    df = pd.DataFrame({'EmailSend': [email_text]})

    df['char_count']       = df['EmailSend'].str.len()
    df['has_question']     = df['EmailSend'].str.contains(r'\?').astype(int)
    df['exclamation_count']= df['EmailSend'].str.count('!')

    words = df['EmailSend'].str.lower().str.split()
    df['token_count']     = words.apply(len)
    df['avg_word_length'] = words.apply(lambda w: np.mean([len(x) for x in w]) if w else 0)
    df['stop_word_count'] = words.apply(lambda w: sum(1 for x in w if x in ENGLISH_STOP_WORDS))
    df['syllable_count']  = df['EmailSend'].apply(
        lambda t: sum(syllables.estimate(w) for w in re.findall(r'\b\w+\b', str(t).lower()))
    )

    return df


def predict_prob(email_text):
    """Returns probability (0-100) of a reply within 3 hours."""
    df = extract_features(email_text)
    proba = model.predict_proba(df[ALL_FEATURE_COLS])[0]
    return float(proba[1]) * 100


def get_features(email_text):
    """Returns a dict of the computed email features for display."""
    df = extract_features(email_text)
    row = df.iloc[0]
    return {
        'Word Count':         int(row['token_count']),
        'Character Count':    int(row['char_count']),
        'Avg Word Length':    round(float(row['avg_word_length']), 2),
        'Syllable Count':     int(row['syllable_count']),
        'Stop Word Count':    int(row['stop_word_count']),
        'Has Question Mark':  'Yes' if int(row['has_question']) == 1 else 'No',
        'Exclamation Marks':  int(row['exclamation_count']),
    }
