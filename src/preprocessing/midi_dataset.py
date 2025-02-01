import os
import sys
import yaml
import pickle
import numpy as np
from pathlib import Path
from random import shuffle
from mido import MidiFile, MidiTrack

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.utils import *
from utils.midi_utils import *

def transpose_midi(mid, min_pitch, max_pitch, output_path, semitone):
    """
    Transpose MIDI file by a given semitone value.

    Parameters:
        mid: MidiFile, MIDI file to transpose.
        min_pitch: int, minimum pitch value.
        max_pitch: int, maximum pitch value.
        output_path: str, path to save the transposed MIDI file.
        semitone: int, number of semitones to transpose the MIDI file.
    """
    midi_name = Path(mid.filename).stem
    midi_path = os.path.join(output_path, midi_name + '.mid')
    transposed_midi = MidiFile(ticks_per_beat=mid.ticks_per_beat)
    for track in mid.tracks:
        transposed_track = MidiTrack()
        for msg in track:
            if msg.type in ['note_on', 'note_off']:
                msg.note += semitone
            transposed_track.append(msg)
        transposed_midi.tracks.append(transposed_track)
    
    min_transpose, max_transpose = pitch_range(transposed_midi)
    if min_pitch < min_transpose  and max_transpose < max_pitch:
        save_midi(transposed_midi, midi_path)

def extract_notes_and_chords(mid):
    events = []
    note_start_times = {}
    current_time = 0
    time_signature = []

    for track in mid.tracks:
        current_time = 0 # Ojo aquí si las pistas comienzan en tiempos distintos
        # events.append(mid.tracks.index(track))  
        for msg in track:
            if msg.type == 'time_signature':
                time_signature = [msg.numerator, msg.denominator]

            if msg.type in ['note_on', 'note_off']:
                current_time += msg.time
            
            if msg.type == 'note_on' and msg.velocity > 0:
                note_start_times[msg.note] = current_time
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                if msg.note in note_start_times:
                    duration = current_time - note_start_times[msg.note]
                    # print(f"Duration of note {msg.note}: {duration}, current time: {current_time} and note start time: {note_start_times[msg.note]}")
                    events.append({
                        'time': note_start_times[msg.note],
                        'note': msg.note,
                        'velocity': msg.velocity,
                        'duration': duration,
                        # 'track': mid.tracks.index(track)
                    })
                    del note_start_times[msg.note]

    return sorted(events, key=lambda x: x['time']), time_signature

def quantize_to_grid(ticks_time, ticks_duration, ticks_per_beat, grids_per_bar):
    ticks_per_grid = (ticks_per_beat * 4) // grids_per_bar
    ticks_per_bar = ticks_per_beat * 4

    # Quantize the start time
    quantized_time = round(ticks_time / ticks_per_grid) * ticks_per_grid

    # Quantize the duration
    quantized_duration = round(ticks_duration / ticks_per_grid) * ticks_per_grid # 1024

    # Calculate the bar and position in the bar of the quantized time
    bar = quantized_time // ticks_per_bar
    position = int((ticks_time % ticks_per_bar) / ticks_per_grid)

    # Calcute the duration in grids
    duration_position = int((quantized_duration % ticks_per_bar) / ticks_per_grid)

    return bar, position, duration_position

TIMESIGN={'[2, 2]': 0, '[2, 4]': 1, '[3, 4]': 2, '[4, 4]': 3, '[5, 4]': 4, '[6, 4]': 5, '[5, 8]': 6, '[6, 8]': 7, '[7, 8]': 8, '[9, 8]': 9}
TPB = {48: 0, 96: 1, 120: 2, 192: 3, 256: 4, 384: 5, 480: 6, 960: 7, 1024: 8}
def midi_to_REMI(mid, grids_per_bar):
    """
    Extracts the notes and chords from a MIDI file and returns a list of tokens in the REMI format.

    Format: 
        - 0: Bar
        - 1-32: Position, 32 values
        - 33-122: Pitch, 90 values [20-109]
        - 123-154: Duration, 32 values
        - 155-164: Time signature, 10 values
        - 165-173: Ticks per beat, 9 values
    """
    ticks_per_beat = mid.ticks_per_beat

    events, time_signature = extract_notes_and_chords(mid)

    tokens = []
    position_set = set()
    duration_set = set()
    current_bar = None

    # Add ticks per beat token
    ticks_per_beat_token = TPB[ticks_per_beat] + 165
    tokens.append(ticks_per_beat_token)

    # Add time signature token
    time_signature_token = TIMESIGN[str(time_signature)] + 155
    tokens.append(time_signature_token)

    for event in events:
        bar, position, duration = quantize_to_grid(event['time'], event['duration'], ticks_per_beat, grids_per_bar)
        if current_bar is None or bar > current_bar:
            tokens.append(0)
            current_bar = bar

        position_token = position + 1
        pitch_token = event['note'] - 20 + 33 # 20 is minimum pitch, 33 is the offset
        duration_token = duration + 123
        try: 
            tokens.append(position_token)
            tokens.append(pitch_token)
            tokens.append(duration_token)
            position_set.add(position)
            duration_set.add(event['duration'])
        except:
            print(f"Error with {Path(mid.filename).stem} - {event['duration']}")
            
    return tokens

def split_data(midi_paths, validation_split):
    """
    Split the MIDI files into training and validation sets.
    
    Parameters:
        midi_paths (list): List of paths to the MIDI files.
        validation_split (float): Fraction of the data to be used for validation.
    
    Returns:
        midi_paths_train: list, paths to the training MIDI files.
        midi_paths_valid: list, paths to the validation MIDI files.
    """
    total_num_files = len(midi_paths)
    num_files_valid = int(total_num_files * validation_split)
    shuffle(midi_paths)
    midi_paths_valid = midi_paths[:num_files_valid]
    midi_paths_train = midi_paths[num_files_valid:]
    return midi_paths_train, midi_paths_valid

def get_emotion(emotion_df, filename):
    """
    Get emotion of a specific midi file.

    Parameters:
        emotion_df (pd.DataFrame): DataFrame containing emotion data.
        filename (str): Name of the midi file to find.
    
    Returns:
        emotion (int): Emotion of the midi file.
    """
    midi_to_find = filename.rfind('_')
    midi_name = filename[:midi_to_find]
    emotion = emotion_df.loc[emotion_df['name'] == midi_name, 'label'].values[0]
    return emotion

def create_datasets(midi_paths, emotion_df, grids_per_bar, output_path, filename):
    """
    Create the datasets for the MIDI files and their corresponding emotions.

    Parameters:
        midi_paths (list): List of paths to the MIDI files.
        emotion_df (pd.DataFrame): DataFrame containing emotion data.
        grids_per_bar (int): Number of grids per bar.
        output_path (str): Path to save the datasets.
        filename (str): Name of the dataset.
    """
    all_tokens = []
    all_emotions = []
    for midi_path in midi_paths:
        try:
            mid = MidiFile(midi_path)
            midi_name = Path(midi_path).stem
            tokens = midi_to_REMI(mid, grids_per_bar)
            all_tokens.append(tokens)
            emotion = get_emotion(emotion_df, midi_name)
            all_emotions.append(emotion)
        except:
            print(f"Error with {midi_path}")

    save_file = os.path.join(output_path, f"{filename}.data")
    save_to_pickle(all_tokens, save_file)
    save_file = os.path.join(output_path, f"{filename}_emotion.data")
    save_to_pickle(all_emotions, save_file)     


if __name__ == '__main__':
    config_path = './src/config/default.yaml'
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

    cleaned_data_path = config['data']['processed']['cleaned']
    prepared_data_path = config['data']['prepared_dir']
    analysis_data_path = config['data']['processed']['analysis']

    GRIDS_PER_BAR = config['data']['grids_per_bar']
    train_ratio = config['data']['train_ratio']
    val_ratio = config['data']['val_ratio']

    midi_paths = list(Path(cleaned_data_path).rglob("*.mid"))
    midi_paths_train, midi_paths_valid = split_data(midi_paths, val_ratio)

    emotion = load_json('csv_metadata.json', analysis_data_path)
    emotion_df = pd.DataFrame(emotion)
    create_datasets(midi_paths_train, emotion_df, GRIDS_PER_BAR, prepared_data_path, 'train')
    create_datasets(midi_paths_valid, emotion_df, GRIDS_PER_BAR, prepared_data_path, 'valid')










