"""
run_preprocessing.py

Execute the preprocessing pipeline for MIDI and metadata files.
"""

import os
import sys
import yaml

from pathlib import Path
from typing import List, Dict, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.utils import *
from utils.midi_utils import *
from midi_preprocessing import *
from metadata_preprocessing import *

class MidiDataPreprocessor:
    def __init__(self, config_path='./src/config/default.yaml'):

        self.config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

        self.midi_raw_data_path = self.config['data']['raw']['midi']
        self.metadata_raw_data_path = self.config['data']['raw']['metadata']
        self.analysis_data_path = self.config['data']['processed']['analysis']
        self.cleaned_data_path = self.config['data']['processed']['cleaned']
        self.max_bars = self.config['data']['max_bars']

        self.csv_metadata_list: List[Dict] = []
        self.midi_metadata_list: List[Dict] = []
        self.midi_delete_list: List[List[str]] = []

    def process_midi_files(self):
        midi_paths = list(Path(self.midi_raw_data_path).rglob("*.mid"))

        for midi_path in midi_paths:
            midi_name = Path(midi_path).stem
            print(f"Processing {midi_name}")

            try:
                midi_file = load_midi(midi_path)
                midi_features = get_midi_features(midi_file)

                metadata = {
                    "name": midi_name,
                    **midi_features
                }
                self.midi_metadata_list.append(metadata)

                if out_of_range_pitch(midi_file):
                    self.midi_delete_list.append([midi_name, "Out of range pitch"])
                    continue

                if more_than_one_time_signature(midi_file):
                    self.midi_delete_list.append([midi_name, "More than one time signature"])
                    continue

                combined_midi = combine_midi_tracks(midi_file)
                split_midi_by_bar(
                    combined_midi, 
                    self.max_bars, 
                    output_path=self.cleaned_data_path, 
                    filename=midi_name
                )

            except Exception as e:
                print(f"Error processing {midi_path}: {e}")
                self.midi_delete_list.append([midi_name, str(e)])
        
    def process_metadata_files(self):
        metadata_paths = list(Path(self.metadata_raw_data_path).rglob("*.csv"))

        for metadata_path in metadata_paths:
            print(f"Processing {metadata_path}")
            csv_metadata = load_metadata(metadata_path)
            processed_metadata = process_metadata(csv_metadata)
            self.csv_metadata_list.append(processed_metadata)

    def save_results(self):
        save_metadata(
            self.midi_metadata_list,
            os.path.join(self.analysis_data_path, 'midi_metadata.json')
        )
        save_metadata(
            self.csv_metadata_list,
            os.path.join(self.analysis_data_path, 'csv_metadata.json'),
            is_midi=False
        )
        save_to_json(
            self.midi_delete_list,
            os.path.join(self.analysis_data_path, 'midi_delete_list.json')
        )

    def run(self):
        print('Starting MIDI data preprocessing...')
        self.process_midi_files()

        print('Starting metadata data preprocessing...')
        self.process_metadata_files()

        print('Saving results...')
        self.save_results()

        print('Preprocessing completed!')

if __name__ == '__main__':
    preprocessor = MidiDataPreprocessor()
    preprocessor.run()



# if __name__=='__main__':
#     config_path = './src/config/default.yaml'
#     sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
#     config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

#     midi_raw_data_path = config['data']['raw']['midi']
#     metadata_raw_data_path = config['data']['raw']['metadata']  
#     analysis_data_path = config['data']['processed']['analysis']
#     cleaned_data_path = config['data']['processed']['cleaned']

#     MAX_BARS = config['data']['max_bars']

#     csv_metadata_list = []
#     midi_metadata_list = []

#     midi_delete_list = []

#     # Midi files preprocessing
#     midi_paths = list(Path(midi_raw_data_path).rglob("*.mid"))
#     for midi_path in midi_paths:
#         midi_name = Path(midi_path).stem
#         print(f"Processing {midi_name}")
#         try:
#             midi_file = load_midi(midi_path)
#             midi_features = get_midi_features(midi_file)
#             metadata = {
#                 "name": midi_name,
#                 **midi_features
#             }
#             midi_metadata_list.append(metadata)

#             if out_of_range_pitch(midi_file):
#                 midi_delete_list.append([midi_name, "Out of range pitch"])
#                 continue

#             if more_than_one_time_signature(midi_file):
#                 midi_delete_list.append([midi_name, "More than one time signature"])
#                 continue

#             # Merge tracks
#             combined_midi = combine_midi_tracks(midi_file)

#             # Split midi by bar
#             split_midi_by_bar(combined_midi, MAX_BARS, output_path=cleaned_data_path, filename=midi_name)

#         except Exception as e:
#             print(f"Error processing {midi_path}: {e}") # 19 files with error due to no time signature
#             midi_delete_list.append([midi_name, str(e)])

#         # Merge tracks
#         # combined_midi = combine_midi_tracks(midi_file, output_path=cleaned_data_path)
#         # print(midi_path.stem)
#         # split_midi_by_bar(combined_midi, MAX_BARS, output_path=cleaned_data_path, filename=midi_path.stem)


#     # Metadata files preprocessing
#     metadata_paths = list(Path(metadata_raw_data_path).rglob("*.csv"))
#     for metadata_path in metadata_paths:
#         print(f"Processing {metadata_path}")
#         csv_metadata = load_metadata(metadata_path)
#         processed_metadata = process_metadata(csv_metadata)
#         csv_metadata_list.append(processed_metadata)


#     save_metadata(midi_metadata_list, os.path.join(analysis_data_path, 'midi_metadata.json'))
#     save_metadata(csv_metadata_list, os.path.join(analysis_data_path, 'csv_metadata.json'), is_midi=False)
#     save_to_json(midi_delete_list, os.path.join(analysis_data_path, 'midi_delete_list.json'))




