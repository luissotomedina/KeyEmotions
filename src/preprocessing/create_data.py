import os
import sys
import yaml
from pathlib import Path
from random import shuffle
from mido import MidiFile, MidiTrack

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.utils import *
from utils.midi_utils import *

def transpose_midi(mid, semitones, min_pitch=20, max_pitch=109):
    """
    Transpose MIDI file by a given semitone value.

    Parameters:
        mid: MidiFile, MIDI file to transpose.
        semitones: int, number of semitones to transpose the MIDI file.
        min_pitch: int, minimum pitch value.
        max_pitch: int, maximum pitch value.

    Returns:
        transposed_midi: MidiFile, transposed MIDI file.
    """
    transposed_midi = MidiFile(ticks_per_beat=mid.ticks_per_beat)

    for track in mid.tracks:
        tranposed_track = MidiTrack()
        for msg in track:
            if msg.type in ['note_on', 'note_off']:
                new_note = msg.note + semitones
                msg = msg.copy(note=new_note)  # Copy message to avoid modifying original
            tranposed_track.append(msg)
        transposed_midi.tracks.append(tranposed_track)
    
    min_transpose, max_transpose = pitch_range(transposed_midi)
    if min_pitch < min_transpose  and max_transpose < max_pitch:
        return transposed_midi
    return None

augmentation = {
    1: [2],
    2: [-2, 2],
    3: [-4, -2, 2],
    4: [-4, -2, 2, 4],
    5: [-6, -4, -2, 2, 4],
    6: [-6, -4, -2, 2, 4, 6]
}
def data_augmentation(midi, output_path, n_transpose=3):
    """
    Data augmentation for MIDI files.

    Parameters:
        midi: MidiFile, MIDI file to transpose.
        output_path: str, path to save the transposed MIDI files.
        n_transpose: int, number of new MIDI files to create (max 6).
    """
    if n_transpose not in augmentation:
        raise ValueError("n_transpose must be between 1 and 6")

    midi_name = Path(midi.filename).stem

    for i, semitones in enumerate(augmentation[n_transpose], start=1):
        transposed_midi = transpose_midi(midi, semitones)
        if transposed_midi:
            transposed_midi_name = f"{midi_name}{i}.mid"
            transposed_midi_path = os.path.join(output_path, transposed_midi_name)
            transposed_midi.save(transposed_midi_path)

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

def quantize_to_grid(ticks_time, ticks_duration, ticks_per_beat, grids_per_bar, time_signature):
    ticks_per_bar =  calculate_ticks_per_bar(time_signature, ticks_per_beat)
    ticks_per_grid = ticks_per_bar // grids_per_bar
    # ticks_per_grid = (ticks_per_beat * 4) // grids_per_bar
    # ticks_per_bar = ticks_per_beat * 4

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
def midi_to_REMI(mid, grids_per_bar, SOS_ind=0, bar_ind=5, pos_ind=6, pitch_ind=38, 
                 duration_ind=128, timesign_ind=160, tpb_ind=170, EOS_ind=180):
    """
    Extracts the notes and chords from a MIDI file and returns a list of tokens in the REMI format.

    Format: 
        - 0: Start of sequence
        - 1-4: emotion, 4 values
        - 5: Bar
        - 6-37: Position, 32 values
        - 38-127: Pitch, 90 values [20-109]
        - 128-159: Duration, 32 values
        - 160-169: Time signature, 10 values
        - 170-178: Ticks per beat, 9 values
        - 179: Padding
        - 180: End of sequence
    """
    ticks_per_beat = mid.ticks_per_beat

    events, time_signature = extract_notes_and_chords(mid)

    tokens = []
    position_set = set()
    duration_set = set()
    current_bar = None

    # Add start of sequence token
    tokens.append(SOS_ind)

    # Add ticks per beat token
    ticks_per_beat_token = TPB[ticks_per_beat] + tpb_ind
    tokens.append(ticks_per_beat_token)

    # Add time signature token
    time_signature_token = TIMESIGN[str(time_signature)] + timesign_ind
    tokens.append(time_signature_token)

    for event in events:
        bar, position, duration = quantize_to_grid(event['time'], event['duration'], ticks_per_beat, grids_per_bar, time_signature)
        if current_bar is None or bar > current_bar:
            tokens.append(bar_ind)
            current_bar = bar

        position_token = position + pos_ind
        pitch_token = event['note'] - 20 + pitch_ind # 20 is minimum pitch, pitch_ind is the offset
        duration_token = duration + duration_ind 
        try: 
            tokens.append(position_token)
            tokens.append(pitch_token)
            tokens.append(duration_token)
            position_set.add(position)
            duration_set.add(event['duration'])
        except:
            print(f"Error with {Path(mid.filename).stem} - {event['duration']}")

    # Add end of sequence token
    tokens.append(EOS_ind)
            
    return tokens

def split_data(midi_paths, validation_split, train_path, valid_path, save_midis=False):
    """
    Split the MIDI files into training and validation sets.
    
    Parameters:
        midi_paths (list): List of paths to the MIDI files.
        validation_split (float): Fraction of the data to be used for validation.
        output_path (str): Path to save the split datasets.
        save_midis (bool): Whether to save the MIDI files
    
    Returns:
        midi_paths_train: list, paths to the training MIDI files.
        midi_paths_valid: list, paths to the validation MIDI files.
    """
    total_num_files = len(midi_paths)
    num_files_valid = int(total_num_files * validation_split)
    shuffle(midi_paths)
    midi_paths_valid = midi_paths[:num_files_valid]
    midi_paths_train = midi_paths[num_files_valid:]

    if save_midis:
        os.makedirs(train_path, exist_ok=True)
        os.makedirs(valid_path, exist_ok=True)

        for midi_path, output_dir in zip(
            [midi_paths_train, midi_paths_valid], [train_path, valid_path]
        ):
            for path in midi_path:
                mid = MidiFile(path)
                save_midi(mid, os.path.join(output_dir, Path(path).name))

    return midi_paths_train, midi_paths_valid

def get_emotion(emotion_df, filename):
    """
    Get emotion of a specific midi file.

    Parameters:
        emotion_df (pd.DataFrame): DataFrame containing emotion data.
        filename (str): Name of the midi file to find.
    
    Returns:
        emotion (int or None): Emotion of the midi file.
    """
    try: 
        midi_to_find = filename.rfind('_')
        midi_name = filename[:midi_to_find] if midi_to_find != -1 else filename

        emotion_row = emotion_df.loc[emotion_df['name'] == midi_name, 'label']
        return int(emotion_row.iloc[0]) if not emotion_row.empty else None
    except Exception as e: 
        print(f"Error getting emotion for {filename}: {e}")
        return None

def create_datasets(midi_paths, emotion_df, grids_per_bar, output_path, filename):
    """
    Create the dataset for the MIDI files and their corresponding emotions.

    Parameters:
        midi_paths (list): List of paths to the MIDI files.
        emotion_df (pd.DataFrame): DataFrame containing emotion data.
        grids_per_bar (int): Number of grids per bar.
        output_path (str): Path to save the datasets.
        filename (str): Name of the dataset.
    """
    all_tokens = []
    for midi_path in midi_paths:
        try:
            mid = MidiFile(midi_path)
            midi_name = Path(midi_path).stem
            tokens = midi_to_REMI(mid, grids_per_bar)
            emotion = get_emotion(emotion_df, midi_name)
            if tokens and emotion is not None:
                all_tokens.append([tokens[0]] + [emotion] + tokens[1:]) # SOS + emotion + REMI tokens + EOS
        except:
            print(f"Error with {midi_path}")

    save_file = os.path.join(output_path, f"{filename}.pkl")
    save_to_pickle(all_tokens, save_file)
    # save_txt = os.path.join(output_path, f"{filename}.txt")
    # with open(save_txt, 'w') as f:
    #     for tokens in all_tokens:
    #         f.write(f"{tokens}\n")
    

if __name__ == '__main__':
    config_path = './src/config/default.yaml'
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

    enable_data_augmentation = True

    cleaned_data_path = config['data']['processed']['cleaned']
    analysis_data_path = config['data']['processed']['analysis']
    prepared_data_path = config['data']['prepared_dir']
    train_prep_dir = config['data']['prepared']['train']   
    val_prep_dir = config['data']['prepared']['val']

    GRIDS_PER_BAR = config['data']['grids_per_bar']
    train_ratio = config['data']['train_ratio']
    val_ratio = config['data']['val_ratio']

    midi_paths = list(Path(cleaned_data_path).rglob("*.mid"))
    midi_paths_train, midi_paths_valid = split_data(midi_paths, val_ratio, train_prep_dir, val_prep_dir, save_midis=True)

    if enable_data_augmentation:
        for midi_path in midi_paths_train:
            mid = MidiFile(midi_path)
            data_augmentation(mid, train_prep_dir)
        midi_paths_train = list(Path(train_prep_dir).rglob("*.mid"))

    emotion = load_json('csv_metadata.json', analysis_data_path)
    emotion_df = pd.DataFrame(emotion)
    create_datasets(midi_paths_train, emotion_df, GRIDS_PER_BAR, output_path=prepared_data_path, filename='train')
    create_datasets(midi_paths_valid, emotion_df, GRIDS_PER_BAR, output_path=prepared_data_path, filename='valid')










