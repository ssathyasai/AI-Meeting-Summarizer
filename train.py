"""
train.py — End-to-end training script
Run: python train.py
"""

import os
import sys
import pickle

# Make sure src/ is importable from project root
sys.path.insert(0, os.path.dirname(__file__))

from src.eda          import load_data, run_eda
from src.preprocessing import preprocess, MAX_TEXT_LEN, MAX_SUMMARY_LEN, VOCAB_SIZE_TEXT, VOCAB_SIZE_SUM
from src.model        import build_seq2seq, train_model
from src.summarizer   import plot_training_history


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots",  exist_ok=True)

    # ── Task 1: EDA ──────────────────────────────────
    print("\n[Step 1] Exploratory Data Analysis")
    df = load_data()
    stats, df = run_eda(df, save_plots=True)

    # ── Task 2: Preprocessing ─────────────────────────
    print("\n[Step 2] Text Preprocessing")
    x_train, x_val, y_train, y_val, text_tok, sum_tok = preprocess(df)

    text_vocab = min(VOCAB_SIZE_TEXT, len(text_tok.word_index) + 1)
    sum_vocab  = min(VOCAB_SIZE_SUM,  len(sum_tok.word_index)  + 1)

    # ── Tasks 3 & 4: Build Encoder-Decoder ───────────
    print("\n[Step 3/4] Building Seq2Seq Model")
    model = build_seq2seq(
        text_vocab=text_vocab,
        sum_vocab=sum_vocab,
        max_text_len=MAX_TEXT_LEN,
        max_summary_len=MAX_SUMMARY_LEN,
    )

    # ── Task 5: Train ────────────────────────────────
    print("\n[Step 5] Training Seq2Seq Model")
    history = train_model(
        model, x_train, y_train, x_val, y_val,
        batch_size=64, epochs=30,
        model_path="models/seq2seq_model.h5",
    )

    # ── Task 7: Evaluate ─────────────────────────────
    print("\n[Step 7] Plotting Training History")
    plot_training_history(history)

    # ── Task 6: Quick inference sample ───────────────
    print("\n[Step 6] Sample Summary Generation")
    from src.model     import build_inference_models
    from src.summarizer import generate_summary

    encoder_model, decoder_model = build_inference_models(model)

    sample = (
        "Yesterday the development team completed the authentication APIs. "
        "The frontend team integrated the login functionality into the main branch. "
        "Testing will begin next Monday after the QA team finishes setting up the environment."
    )
    summary = generate_summary(sample, encoder_model, decoder_model, text_tok, sum_tok)
    print(f"\n  Input   : {sample}")
    print(f"  Summary : {summary}")
    print("\n[Done] Training complete.")


if __name__ == "__main__":
    main()
