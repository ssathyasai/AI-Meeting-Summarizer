"""
Task 2: Text Preprocessing
Applies lowercase, punctuation removal, tokenization, and padding.
"""

import re
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
START_TOKEN = "sostok"
END_TOKEN   = "eostok"

MAX_TEXT_LEN    = 300   # encoder input length
MAX_SUMMARY_LEN = 50    # decoder output length
VOCAB_SIZE_TEXT = 20000
VOCAB_SIZE_SUM  = 8000


# ─────────────────────────────────────────────
# Text Cleaning
# ─────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Lowercase, remove punctuation & extra whitespace."""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)       # remove URLs
    text = re.sub(r"[^a-z0-9\s]", " ", text)                 # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()                  # collapse whitespace
    return text


def clean_summary(summary: str) -> str:
    """Clean + wrap with start/end tokens."""
    summary = clean_text(summary)
    return f"{START_TOKEN} {summary} {END_TOKEN}"


# ─────────────────────────────────────────────
# Preprocessing Pipeline
# ─────────────────────────────────────────────

def preprocess(df: pd.DataFrame, fit: bool = True,
               tokenizer_path: str = None):
    """
    Clean, tokenize and pad text & summaries.

    Returns
    -------
    x_train, x_val, y_train, y_val : np.ndarray
    text_tok, sum_tok               : fitted Tokenizer objects
    """
    df = df.copy()
    df["cleaned_text"]    = df["text"].apply(clean_text)
    df["cleaned_summary"] = df["summary"].apply(clean_summary)

    # Drop rows where text or summary is too short after cleaning
    df = df[df["cleaned_text"].str.split().apply(len) >= 5]
    df = df[df["cleaned_summary"].str.split().apply(len) >= 3]
    df.reset_index(drop=True, inplace=True)

    # ── Tokenizers ──────────────────────────────────────
    if tokenizer_path is None:
        tokenizer_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models", "tokenizers.pkl"
        )

    if fit:
        text_tok = Tokenizer(num_words=VOCAB_SIZE_TEXT, oov_token="<OOV>")
        text_tok.fit_on_texts(df["cleaned_text"])

        sum_tok = Tokenizer(num_words=VOCAB_SIZE_SUM, oov_token="<OOV>")
        sum_tok.fit_on_texts(df["cleaned_summary"])

        import os
        os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)
        with open(tokenizer_path, "wb") as f:
            pickle.dump({"text_tok": text_tok, "sum_tok": sum_tok}, f)
        print(f"[Preprocessing] Tokenizers saved → {tokenizer_path}")
    else:
        with open(tokenizer_path, "rb") as f:
            toks = pickle.load(f)
        text_tok = toks["text_tok"]
        sum_tok  = toks["sum_tok"]

    # ── Sequences ────────────────────────────────────────
    x = text_tok.texts_to_sequences(df["cleaned_text"])
    y = sum_tok.texts_to_sequences(df["cleaned_summary"])

    x = pad_sequences(x, maxlen=MAX_TEXT_LEN,    padding="post", truncating="post")
    y = pad_sequences(y, maxlen=MAX_SUMMARY_LEN, padding="post", truncating="post")

    # ── Train / Val split ────────────────────────────────
    split = int(0.9 * len(x))
    x_train, x_val = x[:split], x[split:]
    y_train, y_val = y[:split], y[split:]

    print(f"[Preprocessing] Train: {len(x_train)} | Val: {len(x_val)}")
    print(f"[Preprocessing] Text vocab: {len(text_tok.word_index)} | "
          f"Summary vocab: {len(sum_tok.word_index)}")

    return x_train, x_val, y_train, y_val, text_tok, sum_tok


# ─────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────

def encode_text(text: str, text_tok: Tokenizer) -> np.ndarray:
    """Encode a single raw text string for inference."""
    cleaned = clean_text(text)
    seq = text_tok.texts_to_sequences([cleaned])
    return pad_sequences(seq, maxlen=MAX_TEXT_LEN, padding="post", truncating="post")


def load_tokenizers(tokenizer_path: str = "models/tokenizers.pkl"):
    with open(tokenizer_path, "rb") as f:
        toks = pickle.load(f)
    return toks["text_tok"], toks["sum_tok"]
