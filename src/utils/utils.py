import os
import json
import pickle
import pandas as pd

   
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