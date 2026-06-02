"""
Task 6: Generate Summary (Greedy Decoding)
Task 7: Evaluate — Training vs Validation Loss
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.text import Tokenizer

from src.preprocessing import (
    encode_text, load_tokenizers,
    START_TOKEN, END_TOKEN, MAX_TEXT_LEN, MAX_SUMMARY_LEN
)
from src.model import build_inference_models, LATENT_DIM


# ─────────────────────────────────────────────
# Task 6: Generate Summary
# ─────────────────────────────────────────────

def generate_summary(
    input_text: str,
    encoder_model: Model,
    decoder_model: Model,
    text_tok: Tokenizer,
    sum_tok: Tokenizer,
    max_summary_len: int = MAX_SUMMARY_LEN,
) -> str:
    """
    Greedy decoding:
    Input text → Encoder → Context vector → Decoder (step-by-step) → Summary
    """
    # Encode input
    x = encode_text(input_text, text_tok)

    # Get encoder context vector (state_h, state_c)
    states_value = encoder_model.predict(x, verbose=0)

    # Build initial decoder input: <sostok>
    start_idx = sum_tok.word_index.get(START_TOKEN, 1)
    end_idx   = sum_tok.word_index.get(END_TOKEN,   2)

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = start_idx

    # Reverse index (index → word)
    reverse_word_index = {v: k for k, v in sum_tok.word_index.items()}

    decoded_tokens = []
    stop = False

    while not stop:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value, verbose=0
        )

        # Greedy: pick highest probability token
        sampled_idx = np.argmax(output_tokens[0, -1, :])
        sampled_word = reverse_word_index.get(sampled_idx, "")

        if sampled_word == END_TOKEN or sampled_word == "" or len(decoded_tokens) >= max_summary_len - 1:
            stop = True
        else:
            decoded_tokens.append(sampled_word)

        # Feed sampled token back in
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_idx
        states_value = [h, c]

    return " ".join(decoded_tokens).strip()


# ─────────────────────────────────────────────
# Task 7: Evaluate — Plot Training History
# ─────────────────────────────────────────────

def plot_training_history(history, save_path: str = "plots/training_history.png"):
    """Compare training loss vs validation loss."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history.history["loss"],     label="Train Loss",     color="steelblue")
    axes[0].plot(history.history["val_loss"], label="Val Loss",       color="coral")
    axes[0].set_title("Training vs Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # Accuracy
    axes[1].plot(history.history["accuracy"],     label="Train Accuracy", color="steelblue")
    axes[1].plot(history.history["val_accuracy"], label="Val Accuracy",   color="coral")
    axes[1].set_title("Training vs Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"[Evaluation] Plot saved → {save_path}")


def compute_statistics(original: str, summary: str) -> dict:
    """Compute word statistics for display."""
    orig_words    = len(original.split())
    summ_words    = len(summary.split())
    compression   = round((1 - summ_words / orig_words) * 100, 1) if orig_words > 0 else 0
    return {
        "original_words":    orig_words,
        "summary_words":     summ_words,
        "compression_ratio": compression,
    }


# ─────────────────────────────────────────────
# Load everything for inference
# ─────────────────────────────────────────────

def load_inference_components(
    model_path: str = "models/seq2seq_model.h5",
    tokenizer_path: str = "models/tokenizers.pkl",
):
    """Load trained model + tokenizers and return inference-ready components."""
    from src.model import build_seq2seq, build_inference_models, LATENT_DIM
    from src.preprocessing import VOCAB_SIZE_TEXT, VOCAB_SIZE_SUM, MAX_TEXT_LEN, MAX_SUMMARY_LEN

    seq2seq = load_model(model_path)
    encoder_model, decoder_model = build_inference_models(seq2seq)
    text_tok, sum_tok = load_tokenizers(tokenizer_path)

    return encoder_model, decoder_model, text_tok, sum_tok
