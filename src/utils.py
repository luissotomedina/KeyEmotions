import os
import json
import pickle

def load_dataset(filename, file_path):
    """
    Load dataset from a specific directory.
    
    Parameters: 
        filename (str): Name of the file to load.
        file_path (str): Path to the directory where the file is located.
    
    Returns:
        data: loaded data.
    """
    file_path = os.path.join(file_path, filename)
    try: 
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            print(f"Dataset loaded from {file_path}")
            return data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
    
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
    
def save_midi(mid, output_path):
    """
    Save MIDI file.
    
    Parameters:
        mid: MidiFile, MIDI file to save.
        output_path (str): Path to save the MIDI file.
    """
    mid.save(output_path)
    print(f"MIDI file saved to {output_path}")