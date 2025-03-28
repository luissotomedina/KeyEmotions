import os
import csv
import json
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from torch import save as torch_save

   
def load_json(filename, file_path):
    """
    Load JSON file from a specific directory.
    
    Parameters: 
        filename (str): Name of the file to load.
        file_path (str): Path to the directory where the file is located.
    
    Returns:
        data: loaded data.
    """
    file_path = os.path.join(file_path, filename)
    try: 
        with open(file_path, 'r') as f:
            data = json.load(f)
            print(f"JSON file loaded from {file_path}")
            return data
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return None
    
def save_to_json(data, output_path):
    """
    Save data to a JSON file.
    
    Parameters:
        data: data to save.
        output_path (str): Path to save the JSON file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)
        print(f"Data saved to {output_path}")

def save_plot(plt, output_path, filename):
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(os.path.join(output_path, filename))
    print(f'Plot saved to {output_path}{filename}')
    plt.close()

def load_pickle(file_path):
    """
    Load pickle file from a specific directory.
    
    Parameters: 
        file_path (str): Path to the directory where the file is located.
    
    Returns:
        data: loaded data.
    """
    try: 
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            print(f"Pickle file loaded from {file_path}")
            return data
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return None

def save_to_pickle(data, output_path):
    """
    Save data to a pickle file.
    
    Parameters:
        data: data to save.
        output_path (str): Path to save the pickle file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
        print(f"Data saved to {output_path}")

def save_metadata(metadata_list, output_path, is_midi=True):
    """
    Save combined metadata to a JSON file

    Parameters:
        metadata_list: list of pd.DataFrame, list of metadata
        output_path: str, path to save the metadata
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if is_midi:
        combined_metadata = pd.DataFrame(metadata_list)
    else:
        combined_metadata = pd.concat(metadata_list, ignore_index=True)

    metadata_dict = combined_metadata.to_dict(orient='records')

    with open(output_path, 'w') as f:
        json.dump(metadata_dict, f, indent=4)
        print(f"Metadata saved to {output_path}")

def create_exp_environment(experiments_dir):
    try:
        os.mkdir(experiments_dir)
    except:
        pass

    num_experiments = len(os.listdir(experiments_dir))

    if num_experiments == 0:
        exp_idx = 1
    else:
        exp_idx = num_experiments + 1

    exp_name = f"exp_{exp_idx}"

    exp_folder = os.path.join(experiments_dir, exp_name)

    generation_folder = os.path.join(exp_folder, "generations")
    logs_folder = os.path.join(exp_folder, "logs")
    plots_folder = os.path.join(exp_folder, "plots")

    try:
        os.mkdir(exp_folder)
        with open(os.path.join(exp_folder, "notes.txt"), "w") as f:
            pass
    except:
        pass
    try:
        os.mkdir(generation_folder)
        with open(os.path.join(generation_folder, "generations.txt"), "w") as f:
            pass
    except:
        pass
    try:
        os.mkdir(logs_folder)
    except:
        pass
    try:
        os.mkdir(plots_folder)
    except:
        pass

    return exp_folder, logs_folder, exp_name

def save_exp_environment(exp_dir, model, config):
    weights_fn = os.path.join(exp_dir, "weigths.pt")
    config_fn = os.path.join(exp_dir, "config.json")
    torch_save(model.state_dict(), weights_fn)
    save_to_json(config, config_fn)
    print(f"Experiment environment saved to {exp_dir}")

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

def plot_loss(train_epoch_loss, val_epoch_loss, exp_name):
    for mode, epoch_loss in zip(["training", "validation"], [train_epoch_loss, val_epoch_loss]):
        sns.set_palette("muted")
        plt.subplots(figsize=(10, 5))

        plt.plot(epoch_loss, label="loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{mode.capitalize()} Loss {exp_name}")
        plt.grid()

        filename = f"{exp_name}_epoch_loss_{mode}.png"
        save_path = f"./experiments/{exp_name}/plots/"
        save_plot(plt, save_path, filename)

def plot_accuracy(train_epoch_acc, val_epoch_acc, exp_name, metric_name):
    sns.set_palette("muted")
    plt.figure(figsize=(10, 5))

    plt.plot(train_epoch_acc, label=f"Training {metric_name}")
    plt.plot(val_epoch_acc, label=f"Validation {metric_name}")
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.title(f"Training and Validation {metric_name} ({exp_name})")
    plt.legend()
    plt.grid()

    filename = f"{exp_name}_epoch_{metric_name.lower().replace(' ', '_')}.png"
    save_path = f"./experiments/{exp_name}/plots/"
    save_plot(plt, save_path, filename)
    

class LogsWriter:
    def __init__(self, output_path, columns):
        self.output_path = output_path
        self.columns = columns

        with open(output_path, 'w') as f: 
            writer = csv.DictWriter(f, fieldnames=self.columns)
            writer.writeheader()

    def update(self, dict_data):
        with open(self.output_path, 'a') as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            writer.writerow(dict_data)
        