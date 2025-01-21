import os
import json
import pandas as pd
from midi_preprocessing import *
from metadata_preprocessing import *

def save_to_json(metadata_list, output_path, is_midi=True):
    """
    Save combined metadata to a JSON file

    Input:
        metadata_list: list of pd.DataFrame, list of metadata
        output_path: str, path to save the metadata
    """
    if is_midi:
        combined_metadata = pd.DataFrame(metadata_list)
    else:
        combined_metadata = pd.concat(metadata_list, ignore_index=True)
    metadata_dict = combined_metadata.to_dict(orient='records')
    with open(output_path, 'w') as f:
        json.dump(metadata_dict, f, indent=4)
        print(f"Metadata saved to {output_path}")


if __name__=='__main__':
    raw_data_path = './data/raw'
    processed_data_path = './data/processed'
    csv_metadata_list = []
    midi_metadata_list = []

    files = os.listdir(raw_data_path)
    for filename in files:
        file_path = os.path.join(raw_data_path, filename)
        print(f"Processing {filename}")
        if filename.endswith('.mid'):
            midi_file = load_midi(file_path)
            if midi_file is not None:
                midi_features = get_midi_features(midi_file)
                metadata = {
                    "name": filename,
                    **midi_features
                }
                midi_metadata_list.append(metadata)
        
        if filename.endswith('.csv'):
            csv_metadata = load_metadata(file_path)
            if csv_metadata is not None:
                processed_metadata = process_metadata(csv_metadata)
                csv_metadata_list.append(processed_metadata)

    save_to_json(midi_metadata_list, os.path.join(processed_data_path, 'midi_metadata.json'))
    save_to_json(csv_metadata_list, os.path.join(processed_data_path, 'csv_metadata.json'), is_midi=False)

