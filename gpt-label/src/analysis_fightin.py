import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict

# Load the DataFrame from the pickle file
df = pd.read_pickle("./final_dataset.pkl")

# Download necessary NLTK resources if not already installed
nltk.download("punkt")
nltk.download("stopwords")


def tokenize(text):
    """
    Tokenizes the text using NLTK, converts to lowercase, and filters out:
      - non-alphabetic tokens,
      - stopwords,
      - single-letter words.
    """
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [
        token
        for token in tokens
        if token.isalpha() and token not in stop_words and len(token) > 1
    ]
    return filtered_tokens


def fightin_words_analysis(df, alpha=1.0):
    """
    Runs a fightin' words analysis on a DataFrame with 'label' and 'instruction' columns.
    For each label, computes the log odds ratio and z-score for each word,
    comparing its frequency in that label to its frequency in all other labels.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'label' and 'instruction' columns.
        alpha (float): Smoothing parameter for the Dirichlet prior.

    Returns:
        results (dict): Dictionary where keys are labels and values are lists of tuples:
                        (word, {'log_odds': value, 'z_score': value}), sorted by descending z-score.
    """
    label_word_counts = {}
    total_counts = defaultdict(int)
    label_totals = {}

    # Initialize counts for each label
    labels = df["label"].unique()
    for label in labels:
        label_word_counts[label] = defaultdict(int)
        label_totals[label] = 0

    # Tokenize and count words for each row
    for _, row in df.iterrows():
        label = row["label"]
        text = row["instruction"]
        tokens = tokenize(text)
        for token in tokens:
            label_word_counts[label][token] += 1
            total_counts[token] += 1
            label_totals[label] += 1

    # Build the full vocabulary
    vocab = set(total_counts.keys())
    V = len(vocab)
    overall_total = sum(label_totals.values())
    results = {}

    # Compute log odds and z-scores for each label compared to the others
    for label in labels:
        n_label = label_totals[label]
        n_other = overall_total - n_label
        words_scores = {}
        for word in vocab:
            count_label = label_word_counts[label][word]
            count_other = total_counts[word] - count_label

            # Compute log odds ratio with additive smoothing
            log_odds = np.log(
                (count_label + alpha) / (n_label - count_label + alpha * V)
            ) - np.log((count_other + alpha) / (n_other - count_other + alpha * V))

            # Compute variance and z-score
            variance = 1.0 / (count_label + alpha) + 1.0 / (count_other + alpha)
            z_score = log_odds / np.sqrt(variance)
            words_scores[word] = {"log_odds": log_odds, "z_score": z_score}

        # Sort words by descending z-score (more significant words first)
        sorted_words = sorted(
            words_scores.items(), key=lambda x: x[1]["z_score"], reverse=True
        )
        results[label] = sorted_words

    return results


def main():
    # Run the analysis
    analysis_results = fightin_words_analysis(df, alpha=1.0)

    # Define the z-score threshold corresponding to p < 0.05 (two-tailed)
    significance_threshold = 1.96

    # For each label, print only the words with significant z-scores
    for label, word_stats in analysis_results.items():
        print(f"\nLabel: {label}")
        print("Significant words (p < 0.05):")
        for word, stats in word_stats:
            if stats["z_score"] > significance_threshold:
                print(
                    f"  {word}: log odds = {stats['log_odds']:.2f}, z-score = {stats['z_score']:.2f}"
                )


if __name__ == "__main__":
    main()
