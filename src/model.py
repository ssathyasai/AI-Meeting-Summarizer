"""
Tasks 3 & 4: Seq2Seq Encoder-Decoder Model
Encoder: Embedding → LSTM → Context Vector
Decoder: Embedding → LSTM → Dense → Summary
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, LSTM, Dense, Concatenate, TimeDistributed
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np


# ─────────────────────────────────────────────
# Hyper-parameters
# ─────────────────────────────────────────────
LATENT_DIM     = 256    # LSTM hidden units
EMBED_DIM      = 128    # Embedding dimension
DROPOUT        = 0.3


# ─────────────────────────────────────────────
# Task 3: Build Encoder
# ─────────────────────────────────────────────

def build_encoder(vocab_size: int, max_text_len: int):
    """
    Encoder:
        Input → Embedding → LSTM → (context_h, context_c)
    Returns encoder_inputs, encoder_states, encoder_model
    """
    encoder_inputs = Input(shape=(max_text_len,), name="encoder_input")

    # Embedding layer
    enc_emb = Embedding(vocab_size, EMBED_DIM,
                        mask_zero=True, name="encoder_embedding")(encoder_inputs)

    # LSTM Encoder — return states for context vector
    encoder_lstm = LSTM(LATENT_DIM, return_state=True,
                        dropout=DROPOUT, name="encoder_lstm")
    _, state_h, state_c = encoder_lstm(enc_emb)

    # Context vector = (state_h, state_c)
    encoder_states = [state_h, state_c]

    return encoder_inputs, encoder_states


# ─────────────────────────────────────────────
# Task 4: Build Decoder
# ─────────────────────────────────────────────

def build_decoder(vocab_size: int, max_summary_len: int, encoder_states):
    """
    Decoder:
        Input → Embedding → LSTM (init with encoder states) → Dense → Output
    Returns decoder_inputs, decoder_outputs
    """
    decoder_inputs = Input(shape=(None,), name="decoder_input")

    # Embedding layer
    dec_emb_layer = Embedding(vocab_size, EMBED_DIM,
                               mask_zero=True, name="decoder_embedding")
    dec_emb = dec_emb_layer(decoder_inputs)

    # LSTM Decoder — initialized with encoder context vector
    decoder_lstm = LSTM(LATENT_DIM, return_sequences=True, return_state=True,
                        dropout=DROPOUT, name="decoder_lstm")
    decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)

    # Dense output layer
    decoder_dense = Dense(vocab_size, activation="softmax", name="decoder_dense")
    decoder_outputs = decoder_dense(decoder_outputs)

    return decoder_inputs, decoder_outputs


# ─────────────────────────────────────────────
# Task 5: Build full Seq2Seq training model
# ─────────────────────────────────────────────

def build_seq2seq(text_vocab: int, sum_vocab: int,
                  max_text_len: int, max_summary_len: int) -> Model:
    """
    Full Seq2Seq model for training.
    Article → Encoder → Context → Decoder → Summary token probabilities
    """
    encoder_inputs, encoder_states = build_encoder(text_vocab, max_text_len)
    decoder_inputs, decoder_outputs = build_decoder(sum_vocab, max_summary_len,
                                                     encoder_states)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs,
                  name="seq2seq_summarizer")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    return model


# ─────────────────────────────────────────────
# Inference Models (encoder + decoder separate)
# ─────────────────────────────────────────────

def build_inference_models(model: Model):
    """
    Extract separate encoder and decoder models for beam / greedy inference.
    """
    # ── Encoder inference model ──────────────────────
    encoder_inputs  = model.get_layer("encoder_input").output
    # Re-run encoder LSTM to get states
    enc_emb         = model.get_layer("encoder_embedding")(encoder_inputs)
    enc_lstm_layer  = model.get_layer("encoder_lstm")
    _, state_h, state_c = enc_lstm_layer(enc_emb)
    encoder_model   = Model(encoder_inputs, [state_h, state_c],
                            name="encoder_inference")

    # ── Decoder inference model ──────────────────────
    dec_state_input_h = Input(shape=(LATENT_DIM,), name="dec_state_h")
    dec_state_input_c = Input(shape=(LATENT_DIM,), name="dec_state_c")
    dec_states_inputs = [dec_state_input_h, dec_state_input_c]

    decoder_inputs_inf = model.get_layer("decoder_input").output
    dec_emb   = model.get_layer("decoder_embedding")(decoder_inputs_inf)
    dec_lstm  = model.get_layer("decoder_lstm")
    dec_out, state_h_out, state_c_out = dec_lstm(
        dec_emb, initial_state=dec_states_inputs
    )
    dec_dense = model.get_layer("decoder_dense")
    dec_out   = dec_dense(dec_out)

    decoder_model = Model(
        [decoder_inputs_inf] + dec_states_inputs,
        [dec_out, state_h_out, state_c_out],
        name="decoder_inference",
    )

    return encoder_model, decoder_model


# ─────────────────────────────────────────────
# Training helper
# ─────────────────────────────────────────────

def train_model(model: Model,
                x_train, y_train, x_val, y_val,
                batch_size: int = 64, epochs: int = 30,
                model_path: str = None):
    """Task 5: Train Article → Summary mapping."""
    import os
    if model_path is None:
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models", "seq2seq_model.h5"
        )
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True, verbose=1),
    ]

    # Decoder target = y shifted by 1 (teacher forcing)
    history = model.fit(
        [x_train, y_train[:, :-1]],
        y_train[:, 1:],
        validation_data=(
            [x_val, y_val[:, :-1]],
            y_val[:, 1:],
        ),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
    )
    print(f"[Training] Best model saved → {model_path}")
    return history
