import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

def load_evaluation_rows(file_path):
    """Load evaluation rows from JSONL file."""
    rows = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)

def plot_model_comparison():
    """Plot comparison of three models' evaluation scores."""

    # Load data for all three models
    gpt_120b_df = load_evaluation_rows('data/evaluation_results/test_vanila_rag_no_context_gpt_oss_120b.rows.jsonl')
    gpt_20b_df = load_evaluation_rows('data/evaluation_results/test_vanila_rag_no_context_gpt_oss_20b.rows.jsonl')
    qwen_df = load_evaluation_rows('data/evaluation_results/test_vanila_rag_qwen3_32b.DeepEvalEvaluator.rows.jsonl')
    claude_sonnet_4_df = load_evaluation_rows('data/evaluation_results/test_vanila_rag_claude_sonnet_4.DeepEvalEvaluator.rows.jsonl')

    # Set style
    sns.set_style("whitegrid")

    # Create subplots for each metric
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    metrics = ['faithfulness', 'answer_relevancy', 'contextual_relevancy']
    model_names = ['GPT-120B', 'GPT-20B', 'Qwen3-32B', 'Claude-Sonnet-4']

    for i, metric in enumerate(metrics):
        # Combine data for this metric
        data = [
            gpt_120b_df[metric].values,
            gpt_20b_df[metric].values,
            qwen_df[metric].values,
            claude_sonnet_4_df[metric].values
        ]

        # Create KDE plots
        for j, (model_data, name) in enumerate(zip(data, model_names)):
            sns.kdeplot(data=model_data, ax=axs[i], fill=True, alpha=0.3,
                       label=name, linewidth=2)

        axs[i].set_title(f'{metric.replace("_", " ").title()} Distribution', fontsize=14)
        axs[i].set_xlabel('Score', fontsize=12)
        axs[i].set_ylabel('Density', fontsize=12)
        axs[i].legend()
        axs[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print summary statistics
    print("Summary Statistics:")
    print("=" * 50)

    for metric in metrics:
        print(f"\n{metric.replace('_', ' ').title()}:")
        print("-" * 30)

        for name, df in [('GPT-120B', gpt_120b_df), ('GPT-20B', gpt_20b_df), ('Qwen3-32B', qwen_df), ('Claude-Sonnet-4', claude_sonnet_4_df)]:
            scores = df[metric]
            print(f"{name}: Mean={scores.mean():.3f}, Std={scores.std():.3f}, Min={scores.min():.3f}, Max={scores.max():.3f}")

if __name__ == "__main__":
    plot_model_comparison()