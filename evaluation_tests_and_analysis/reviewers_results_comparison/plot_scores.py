#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["pandas", "seaborn", "matplotlib"]
# ///

import json
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Use non-GUI backend to avoid tkinter errors


def load_review_scores(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Adjusted here: scores list is at the top level, not inside "review"
    score_entries = data["scores"]

    score_counts = {}

    for entry in score_entries:
        scores = entry["scores"]
        for key, value in scores.items():
            if key != "comments" and value is True:
                score_counts[key] = score_counts.get(key, 0) + 1

    return pd.DataFrame(list(score_counts.items()), columns=["Score Type", "Count"])


def plot_score_histogram(df, output_path):
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="Count", y="Score Type", data=df, orient="h")

    # Add black count labels to each bar
    for container in ax.containers:
        ax.bar_label(container, fmt="%d", label_type="edge", color="black", padding=3)

    total_n = df["Count"].sum()
    plt.title(f"True Score Type Frequencies (N={total_n})")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved histogram to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot histogram of score types from review JSON.")
    parser.add_argument("-i", "--input", default="review_data.json", help="Input JSON file path")
    parser.add_argument("-o", "--output", default="score_histogram.png", help="Output PNG file path")
    args = parser.parse_args()

    df_scores = load_review_scores(args.input)
    plot_score_histogram(df_scores, args.output)


if __name__ == "__main__":
    main()
