import os
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from utils.utils import save_plot


def plot_progress(df, column, exp_name, mode="training"):
    sns.set_palette("muted")
    plt.subplots(figsize=(10, 5))

    if mode == "training":
        df.sort_values(by=["epoch", "batches"], inplace=True)
        x = df.index
        xlabel = "Iterations"
    else:
        x = df.epoch
        xlabel = "Epoch"
    
    plt.plot(x, df[column], label=column)
    plt.xlabel(xlabel)
    plt.ylabel(column)
    plt.title(f"{mode.capitalize()} {column} {exp_name}")
    plt.grid()

    filename = f"{exp_name}_{mode}_{column}.png"
    save_path = f"./experiments/{exp_name}/plots/"
    save_plot(plt, save_path, filename)


if __name__=="__main__":
    i = 1
    while i <= 5:
        train_csv = f"./experiments/exp_{i}/logs/logs_train.csv"
        df = pd.read_csv(train_csv)
        plot_progress(df, 'ppl', f"exp_{i}")
        plot_progress(df, 'loss', f"exp_{i}")
        val_csv = f"./experiments/exp_{i}/logs/logs_val.csv"
        df = pd.read_csv(val_csv)
        plot_progress(df, 'loss', f"exp_{i}", mode="validation")
        i += 1


    
