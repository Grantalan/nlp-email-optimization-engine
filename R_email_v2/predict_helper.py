# Python helper — loaded by the R Shiny app via reticulate.
# Takes raw email text, runs the same feature engineering from the notebook,
# and returns the predicted probability of a response within 3 hours.

import joblib
import pandas as pd
import numpy as np
import re
import syllables
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Load the trained model from the shared data folder
model = joblib.load("../data/enron_model.pkl")

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


def get_top_coefficients(email_text=None, n=12):
    """
    Returns the top n positive and top n negative word coefficients from the
    logistic regression model — same idea as the coef_df table in the
    nlp-text_classification notebook.

    Positive coefficient → word pushes probability of response UP.
    Negative coefficient → word pushes probability of response DOWN.

    If email_text is provided, words that appear in the email are flagged.
    """
    # Pull the CountVectorizer and LogisticRegression out of the pipeline
    vectorizer = model.named_steps['prep'].named_transformers_['text_vec']
    lr         = model.named_steps['clf']

    word_names = vectorizer.get_feature_names_out()
    # LR coef_ shape is (1, n_features); first slice is the text block
    n_words    = len(word_names)
    word_coefs = lr.coef_[0][:n_words]

    # Top n most positive and top n most negative
    top_pos_idx = np.argsort(word_coefs)[::-1][:n]
    top_neg_idx = np.argsort(word_coefs)[:n]

    words = list(word_names[top_pos_idx]) + list(word_names[top_neg_idx])
    coefs = list(word_coefs[top_pos_idx]) + list(word_coefs[top_neg_idx])
    dirs  = ['Increases Response Chance'] * n + ['Decreases Response Chance'] * n

    # Flag words that appear in the user's email
    if email_text:
        email_words = set(re.findall(r'\b\w+\b', email_text.lower()))
        in_email = ['Yes' if w in email_words else 'No' for w in words]
    else:
        in_email = ['No'] * (n * 2)

    return {
        'word':     words,
        'coef':     [round(float(c), 4) for c in coefs],
        'direction': dirs,
        'in_email': in_email,
    }


def get_email_word_contributions(email_text):
    """
    Same logic as coef_df in nlp-text_classification.ipynb:

        coef_df = pd.DataFrame({
            'word': vect.get_feature_names_out(),
            'coef_response': lr.coef_[0]
        })

    Build the full coef_df, filter it to words that appear in the input
    email, add a count column, and sort by coefficient descending —
    exactly like the notebook's sort_values(...).head() pattern.
    """
    from collections import Counter

    # Pull components out of the pipeline — same as notebook's named_steps
    vectorizer = model.named_steps['prep'].named_transformers_['text_vec']
    lr         = model.named_steps['clf']

    # Build coef_df — mirrors notebook exactly
    n_words = len(vectorizer.get_feature_names_out())
    coef_df = pd.DataFrame({
        'word':          vectorizer.get_feature_names_out(),
        'coef_response': lr.coef_[0][:n_words]
    })

    # Tokenize the input email the same way CountVectorizer does
    tokens = re.findall(r'\b[a-z]{2,}\b', email_text.lower())
    counts = Counter(tokens)

    # Build a row for EVERY word in the email — words not in the vocabulary
    # get coefficient 0.0 (the model has no opinion on them)
    all_words = pd.DataFrame({
        'word':  list(counts.keys()),
        'count': list(counts.values())
    })

    # Left-join onto coef_df so vocabulary words get their coefficient,
    # unknown words get NaN → fill with 0.0
    email_coef_df = all_words.merge(coef_df, on='word', how='left')
    email_coef_df['coef_response'] = email_coef_df['coef_response'].fillna(0.0)

    # Sort by coefficient descending — same as notebook's sort_values(ascending=False)
    email_coef_df = email_coef_df.sort_values('coef_response', ascending=False)

    directions = []
    for c in email_coef_df['coef_response']:
        if c > 0:
            directions.append('Toward Response')
        elif c < 0:
            directions.append('Away From Response')
        else:
            directions.append('Not in Model')

    return {
        'Word':        list(email_coef_df['word']),
        'Count':       [int(x) for x in email_coef_df['count']],
        'Coefficient': [round(float(c), 4) for c in email_coef_df['coef_response']],
        'Direction':   directions,
    }
