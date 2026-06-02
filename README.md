# 🤖 AI Meeting Minutes Summarizer

A **Seq2Seq Encoder-Decoder** model (LSTM) that automatically generates concise summaries from long meeting notes — with a full **Streamlit** web application.

---

## 📋 Table of Contents
- [Problem Statement](#problem-statement)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Train the Model](#train-the-model)
- [Run the App](#run-the-app)
- [Features](#features)
- [Dataset](#dataset)

---

## Problem Statement

Software teams hold daily standups, sprint planning, and client meetings. Employees don't have time to read long meeting notes. This system automatically generates concise summaries using a Seq2Seq LSTM model.

---

## Architecture

```
Long Meeting Notes
       ↓
   Encoder (Embedding → LSTM)
       ↓
  Context Vector (state_h, state_c)
       ↓
   Decoder (Embedding → LSTM → Dense)
       ↓
  Short Summary
```

---

## Project Structure

```
AI-Meeting-Summarizer/
├── app.py                  # Streamlit application
├── train.py                # End-to-end training script
├── requirements.txt        # Python dependencies
├── src/
│   ├── eda.py              # Task 1: Exploratory Data Analysis
│   ├── preprocessing.py    # Task 2: Text preprocessing + tokenization
│   ├── model.py            # Tasks 3, 4, 5: Seq2Seq model
│   └── summarizer.py       # Tasks 6, 7: Inference + evaluation
├── Data/
│   ├── news_summary.csv
│   └── news_summary_more.csv
├── models/                 # Saved after training (git-ignored)
└── plots/                  # Generated plots (git-ignored)
```

---

## Setup & Installation

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/AI-Meeting-Summarizer.git
cd AI-Meeting-Summarizer

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Train the Model

```bash
python train.py
```

This will:
1. Run **EDA** and save plots to `plots/`
2. **Preprocess** text and save tokenizers to `models/tokenizers.pkl`
3. **Build** the Seq2Seq Encoder-Decoder model
4. **Train** and save the best model to `models/seq2seq_model.h5`
5. **Plot** training vs validation loss
6. Generate a **sample summary**

---

## Run the App

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Features

| Section | Feature |
|---------|---------|
| Section 1 | Paste meeting notes |
| Section 2 | Generate Summary button |
| Section 3 | AI-generated summary output |
| Section 4 | Statistics: Original Words, Summary Words, Compression Ratio |
| Bonus 1 | Upload PDF → Generate Summary |
| Bonus 2 | Upload DOCX → Generate Summary |
| Bonus 3 | Download Summary as PDF |

---

## Dataset

- `news_summary.csv` — News articles with short summaries
- `news_summary_more.csv` — Additional news articles with headlines

Both datasets are used to train the Seq2Seq summarization model.

---

## Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set **Main file**: `app.py`
5. Deploy!

> **Note:** For cloud deployment, train locally first, then commit the `models/` directory (remove it from `.gitignore`), or use cloud storage for model files.

---

## Tech Stack

- **Python 3.9+**
- **TensorFlow / Keras** — Seq2Seq model
- **Streamlit** — Web application
- **NLTK** — NLP utilities
- **fpdf2** — PDF generation
- **python-docx** — DOCX parsing
- **PyMuPDF** — PDF parsing
