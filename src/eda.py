"""
Task 1: Exploratory Data Analysis (EDA)
Analyzes the dataset for records, article lengths, and summary lengths.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ─────────────────────────────────────────────
# Load Data
# ─────────────────────────────────────────────

def load_data(data_dir: str = "Data") -> pd.DataFrame:
    """Load and merge both CSV files into a unified DataFrame."""
    df1 = pd.read_csv(os.path.join(data_dir, "news_summary.csv"), encoding="latin-1")
    df2 = pd.read_csv(os.path.join(data_dir, "news_summary_more.csv"), encoding="latin-1")

    # Normalize column names: both need 'text' and 'summary'
    df1 = df1[["text", "ctext"]].rename(columns={"text": "summary", "ctext": "text"})
    df2 = df2[["text", "headlines"]].rename(columns={"headlines": "summary"})

    df = pd.concat([df1, df2], ignore_index=True)
    df.dropna(subset=["text", "summary"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ─────────────────────────────────────────────
# EDA Functions
# ─────────────────────────────────────────────

def run_eda(df: pd.DataFrame, save_plots: bool = False, plot_dir: str = "plots") -> dict:
    """Perform EDA and return key statistics."""
    df = df.copy()
    df["text_words"]    = df["text"].apply(lambda x: len(str(x).split()))
    df["summary_words"] = df["summary"].apply(lambda x: len(str(x).split()))

    stats = {
        "num_records":          len(df),
        "avg_article_length":   round(df["text_words"].mean(), 2),
        "avg_summary_length":   round(df["summary_words"].mean(), 2),
        "max_article_length":   int(df["text_words"].max()),
        "min_article_length":   int(df["text_words"].min()),
        "max_summary_length":   int(df["summary_words"].max()),
        "min_summary_length":   int(df["summary_words"].min()),
        "compression_ratio":    round(
            1 - df["summary_words"].mean() / df["text_words"].mean(), 4
        ),
    }

    print("=" * 50)
    print("           EDA REPORT")
    print("=" * 50)
    for k, v in stats.items():
        print(f"  {k:<25}: {v}")
    print("=" * 50)

    if save_plots:
        os.makedirs(plot_dir, exist_ok=True)
        _plot_distributions(df, plot_dir)

    return stats, df


def _plot_distributions(df: pd.DataFrame, plot_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(df["text_words"],    bins=50, kde=True, ax=axes[0], color="steelblue")
    axes[0].set_title("Article Word Count Distribution")
    axes[0].set_xlabel("Word Count")

    sns.histplot(df["summary_words"], bins=50, kde=True, ax=axes[1], color="coral")
    axes[1].set_title("Summary Word Count Distribution")
    axes[1].set_xlabel("Word Count")

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "word_distributions.png"), dpi=120)
    plt.close()
    print(f"[EDA] Plot saved → {plot_dir}/word_distributions.png")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    df = load_data()
    stats, df = run_eda(df, save_plots=True)
