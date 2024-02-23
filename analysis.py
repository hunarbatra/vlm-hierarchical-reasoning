import fire

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import os

from typing import List, Optional
from utils import task_mapper

def get_results_df_from_exp(exp_dirs = List[str]) -> List[pd.DataFrame]:
    res_dfs = []
    for exp_dir in exp_dirs:
        curr_response_path = f'./experiments/{exp_dir}/responses.csv'
        res_dfs.append(pd.read_csv(curr_response_path))
    return res_dfs

def print_bar_values(ax) -> None:
    for patch in ax.patches:
        height = patch.get_height()
        ax.annotate(
            f"{height:.2f}", 
            (patch.get_x() + patch.get_width() / 2., height), 
            ha='center', va='center', 
            fontsize=10, color='black',
            xytext=(0, 5),
            textcoords='offset points'
        )

def accuracy_plot(
    exp_dirs: List[str],
    ground_truth_label: Optional[str] = 'correct_answer',
    response_label: Optional[str] = 'parsed_answer',
    title: Optional[str] = '',
    ylim: Optional[tuple] = (0, 1),
    save_fig: Optional[bool] = False,
):
    response_exp_dfs = get_results_df_from_exp(exp_dirs)
    all_data = pd.DataFrame()
    
    for idx, df in enumerate(response_exp_dfs):
        df['exp'] = exp_dirs[idx]
        df['exp'] = df['exp'].apply(task_mapper)
        df['is_correct'] = df[response_label] == df[ground_truth_label]
        all_data = pd.concat([all_data, df[['exp', 'is_correct']]], ignore_index=True)

    sns.set(style="whitegrid")
    print(len(exp_dirs))
    palette = sns.color_palette("Set3", n_colors=len(exp_dirs)+2)[2:]
    plt.figure(figsize=(10, 6))
    p1 = sns.barplot(x='exp', y='is_correct', data=all_data, errorbar="se", palette=palette, edgecolor="black")
    print_bar_values(p1)
    plt.xticks(rotation=45)
    plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('')
    if ylim:
        plt.ylim(ylim)
        
    plt.tight_layout()
    if save_fig:
        plt.savefig(f'./experiments/{exp_dirs[0]}/accuracy_plot.png')
    plt.show()


if __name__ == '__main__':
    fire.Fire(
        {
            "accuracy_plot": accuracy_plot,
        }
    )