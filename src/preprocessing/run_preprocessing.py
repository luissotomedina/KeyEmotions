import os
import json
import yaml
import pandas as pd
from midi_preprocessing import *
from metadata_preprocessing import *

def save_to_json(metadata_list, output_path, is_midi=True):
    """
    Save combined metadata to a JSON file

    Parameters:
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
    config_path = './src/config/default.yaml'
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

    midi_raw_data_path = config['data']['raw']['midi']
    metadata_raw_data_path = config['data']['raw']['metadata']  
    analysis_data_path = config['data']['processed']['analysis']
    cleaned_data_path = config['data']['processed']['cleaned']

    MAX_BARS = config['data']['max_bars']

    csv_metadata_list = []
    midi_metadata_list = []

    # Midi files preprocessing
    midi_paths = list(Path(midi_raw_data_path).rglob("*.mid"))
    for midi_path in midi_paths:
        print(f"Processing {midi_path}")
        midi_file = load_midi(midi_path)
        midi_features = get_midi_features(midi_file)
        metadata = {
            "name": Path(midi_path).stem,
            **midi_features
        }
        midi_metadata_list.append(metadata)
        # Merge tracks
        combined_midi = combine_midi_tracks(midi_file, output_path=cleaned_data_path)
        print(midi_path.stem)
        split_midi_by_bar(combined_midi, MAX_BARS, output_path=cleaned_data_path, filename=midi_path.stem)


    # Metadata files preprocessing
    metadata_paths = list(Path(metadata_raw_data_path).rglob("*.csv"))
    for metadata_path in metadata_paths:
        print(f"Processing {metadata_path}")
        csv_metadata = load_metadata(metadata_path)
        processed_metadata = process_metadata(csv_metadata)
        csv_metadata_list.append(processed_metadata)


    save_to_json(midi_metadata_list, os.path.join(analysis_data_path, 'midi_metadata.json'))
    save_to_json(csv_metadata_list, os.path.join(analysis_data_path, 'csv_metadata.json'), is_midi=False)



