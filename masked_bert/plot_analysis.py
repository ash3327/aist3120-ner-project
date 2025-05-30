import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

def plot_training_curves():
    # Read the training curves data
    df = pd.read_csv('results/training_curves.csv')
    
    # Set style
    # plt.style.use('seaborn')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance vs Masking Probability', fontsize=16)
    
    # Plot F1 Score
    sns.lineplot(data=df, x='masking_prob', y='f1', marker='o', ax=axes[0,0])
    axes[0,0].set_title('F1 Score')
    axes[0,0].set_xlabel('Masking Probability')
    axes[0,0].set_ylabel('F1 Score')
    axes[0,0].grid(True)
    
    # Plot Loss
    sns.lineplot(data=df, x='masking_prob', y='loss', marker='o', ax=axes[0,1])
    axes[0,1].set_title('Loss')
    axes[0,1].set_xlabel('Masking Probability')
    axes[0,1].set_ylabel('Loss')
    axes[0,1].grid(True)
    
    # Plot Precision
    sns.lineplot(data=df, x='masking_prob', y='precision', marker='o', ax=axes[1,0])
    axes[1,0].set_title('Precision')
    axes[1,0].set_xlabel('Masking Probability')
    axes[1,0].set_ylabel('Precision')
    axes[1,0].grid(True)
    
    # Plot Recall
    sns.lineplot(data=df, x='masking_prob', y='recall', marker='o', ax=axes[1,1])
    axes[1,1].set_title('Recall')
    axes[1,1].set_xlabel('Masking Probability')
    axes[1,1].set_ylabel('Recall')
    axes[1,1].grid(True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_seen_unseen_comparison():
    # Read the seen/unseen metrics data
    df = pd.read_csv('results/seen_unseen_metrics.csv')
    
    # Set style
    # plt.style.use('seaborn')
    sns.set_palette("husl")
    
    # Create figure with subplots for each dataset
    for dataset in ['conll', 'wikiann']:
        dataset_df = df[df['dataset'] == dataset]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Seen vs Unseen Entity Performance - {dataset.upper()}', fontsize=16)
        
        # Plot F1 Score comparison
        sns.lineplot(data=dataset_df, x='masking_prob', y='seen_f1', label='Seen', marker='o', ax=axes[0,0])
        sns.lineplot(data=dataset_df, x='masking_prob', y='unseen_f1', label='Unseen', marker='o', ax=axes[0,0])
        axes[0,0].set_title('F1 Score')
        axes[0,0].set_xlabel('Masking Probability')
        axes[0,0].set_ylabel('F1 Score')
        axes[0,0].grid(True)
        axes[0,0].legend()
        
        # Plot Precision comparison
        sns.lineplot(data=dataset_df, x='masking_prob', y='seen_precision', label='Seen', marker='o', ax=axes[0,1])
        sns.lineplot(data=dataset_df, x='masking_prob', y='unseen_precision', label='Unseen', marker='o', ax=axes[0,1])
        axes[0,1].set_title('Precision')
        axes[0,1].set_xlabel('Masking Probability')
        axes[0,1].set_ylabel('Precision')
        axes[0,1].grid(True)
        axes[0,1].legend()
        
        # Plot Recall comparison
        sns.lineplot(data=dataset_df, x='masking_prob', y='seen_recall', label='Seen', marker='o', ax=axes[1,0])
        sns.lineplot(data=dataset_df, x='masking_prob', y='unseen_recall', label='Unseen', marker='o', ax=axes[1,0])
        axes[1,0].set_title('Recall')
        axes[1,0].set_xlabel('Masking Probability')
        axes[1,0].set_ylabel('Recall')
        axes[1,0].grid(True)
        axes[1,0].legend()
        
        # Create a table in the last subplot
        axes[1,1].axis('off')
        table_data = dataset_df[['masking_prob', 'seen_f1', 'unseen_f1', 'seen_precision', 'unseen_precision', 'seen_recall', 'unseen_recall']]
        table_data = table_data.round(4)
        table = axes[1,1].table(cellText=table_data.values,
                               colLabels=table_data.columns,
                               loc='center',
                               cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        output_dir = Path('results')
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / f'seen_unseen_comparison_{dataset}.png', dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == '__main__':
    plot_training_curves()
    plot_seen_unseen_comparison() 