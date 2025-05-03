from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import pandas as pd

def find_average_performance_by_formatpair(input, img_path):
    def plot_summary_table(summary_df, title="Similarity & Content Metrics by Type Pair"):
        fig, ax = plt.subplots(figsize=(10, len(summary_df) * 0.6 + 1))
        ax.axis('off')

        # Create table
        table = ax.table(
            cellText=summary_df.values,
            rowLabels=summary_df.index,
            colLabels=summary_df.columns,
            cellLoc='center',
            rowLoc='center',
            loc='center'
        )

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)

        plt.title(title, fontweight='bold', fontsize=12, pad=20)
        plt.tight_layout()
        plt.savefig(img_path, dpi=300)
        plt.close()
  
    # Load the CSV file
    df = pd.read_csv(input)
    # Normalize Type1 + Type2 into an unordered pair key
    df["TypePair"] = df.apply(
        lambda row: " + ".join(sorted([row["Type1"].strip(), row["Type2"].strip()])),
        axis=1
    )
    # Select metrics to average
    metrics = ['Cosine_similarity', 'Rouge1', 'Rouge2', 'RougeL', 'MissingInfoRatio']
    # Group by the TypePair and compute mean of metrics
    summary_df = df.groupby("TypePair")[metrics].mean().round(3)
    plot_summary_table(summary_df)
    print(summary_df)


if __name__ == "__main__":
    input = "group_similarity_chunks_1.csv"
    find_average_performance_by_formatpair(input, "average_summary_table_chunk1.png")
    input = "group_similarity_chunks_12.csv"
    find_average_performance_by_formatpair(input, "average_summary_table_chunk12.png")