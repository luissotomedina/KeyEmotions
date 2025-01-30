import os
from pathlib import Path

import mido
from mido import MidiFile, MidiTrack, merge_tracks

def load_midi(midi_path):
    """
    Load a midi file
    
    Parameters:
        midi_path: str, path to the midi file
    
    Returns:
        midi_data: mido.MidiFile, midi data
    """
    try:
        midi_data = mido.MidiFile(midi_path)    
        return midi_data
    
    except Exception as e:
        print(f"Error loading {midi_path}: {e}")
        return None
    
def get_note_density(mid):
    """
    Get note density of a midi file
    
    Parameters:
        mid: mido.MidiFile, midi data

    Returns:
        density: float, note density
    """
    duration = mid.length
    n_notes = sum([1 for track in mid.tracks for msg in track if msg.type == 'note_on'])
    density = n_notes / duration
    return density

def get_tempo(mid):
    """
    Get tempo of a midi file

    Parameters:
        mid: mido.MidiFile, midi data
    
    Returns:
        tempo: float, tempo
    """
    default_tempo_bpm = 120
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                tempo_bpm = round(mido.tempo2bpm(msg.tempo), 0)
                return tempo_bpm
    return default_tempo_bpm

def get_n_instruments(mid):
    """
    Get number of instruments of a midi file

    Parameters:
        mid: mido.MidiFile, midi data

    Returns:
        n_instruments: int, number of instruments
    """
    instrument_channels = set()
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0:
                instrument_channels.add(msg.channel)
    n_instruments = len(instrument_channels)
    return n_instruments

def get_numerator_denominator(mid):
    """
    Get time signature of a midi file
    
    Parameters:
        mid: mido.MidiFile, midi data
    
    Returns:
        numerator: int, numerator of the time signature
        denominator: int, denominator of the time signature
    """
    numerator, denominator = 4, 4
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'time_signature':
                numerator = msg.numerator
                denominator = msg.denominator
                return numerator, denominator
    return numerator, denominator

def get_bar_length(duration, tempo, numerator):
    """
    Get bar length of a midi file
    
    Parameters:
        mid: mido.MidiFile, midi data
        duration: float, duration of the midi file in seconds
        tempo: int, tempo
        numerator: int, numerator of the time signature
    
    Returns: 
        bar_length: float, bar length
    """
    beats_per_song = tempo * (duration / 60)
    bars = round(beats_per_song / numerator, 0)
    return bars
    

def get_midi_features(mid):
    """
    Get features of a midi file

    Parameters:
        mid: mido.MidiFile, midi data
    
    Returns:
        midi_features: dict, midi features
    """
    note_density = get_note_density(mid)
    tempo = get_tempo(mid)
    n_instruments = get_n_instruments(mid)
    duration = mid.length
    numerator, denominator = get_numerator_denominator(mid)
    bars = get_bar_length(duration, tempo, numerator)
    midi_features = {
        "note_density": note_density,
        "tempo": tempo,
        "n_instruments": n_instruments,
        "duration": duration,
        "numerator": numerator,
        "denominator": denominator, 
        "bars": bars
    }
    return midi_features

def combine_midi_tracks(mid, output_path):
    """
    Combine midi tracks into a single track

    Input:
        mid: mido.MidiFile, midi data
        output_path: str, path to save the merged midi file

    Returns:
        combined_midi: mido.MidiFile, combined midi file
    """
    if not isinstance(mid, MidiFile):
        raise ValueError("Expected a MidiFile object as input")
    
    combined_midi = MidiFile(ticks_per_beat=mid.ticks_per_beat)

    combined_midi.tracks = [merge_tracks(mid.tracks)]

    # Returns_file = os.path.join(Returns_path, Path(mid.filename).stem + "_merged.mid")
    # combined_midi.save(Returns_file)
    return combined_midi

def save_bar_messages(message, bar_number, ticks_per_beat, filename, output_path):
    """
    Save messages of a bar to a midi file

    Parameters:
        message: list, list of messages
        bar_number: int, bar number
        ticks_per_beat: int, ticks per beat
        filename: str, filename
        output_path: str, path to save the midi file
    """
    bar_midi = MidiFile(ticks_per_beat=ticks_per_beat)
    bar_track = MidiTrack()
    bar_midi.tracks.append(bar_track)
    for msg in message:
        msg.time = int(msg.time) # int(msg.time)
        bar_track.append(msg)
    Returns_file = os.path.join(output_path, f"{filename}_{bar_number}.mid")
    bar_midi.save(Returns_file)
    print(f"Bar {bar_number} saved to {Returns_file}")

def split_midi_by_bar(mid, bars_to_extract, output_path, filename):
    """
    Split a midi file by bar
    
    Parameters:
        mid: mido.MidiFile, midi data
        bars_to_extract: int, number of bars to extract
        output_path: str, path to save the midi files
        filename: str, filename
        save_remaining: bool, save remaining messages    
    """
    ticks_per_beat = mid.ticks_per_beat
    numerator_time_signature = next((msg.numerator for msg in mid if msg.type == 'time_signature'), 4)
    ticks_per_bar = ticks_per_beat * numerator_time_signature
    ticks_to_extract = ticks_per_bar * bars_to_extract

    current_bar = 0
    current_ticks = 0
    metadata_info = [msg for msg in mid if msg.is_meta]
    bar_messages = []
    chord_messages = []

    for track in mid.tracks:
        for msg in track:
            if msg.time == 0:
                chord_messages.append(msg)
            else: 
                chord_messages.append(msg)
                # CASE 1: the chord is of the next section
                if current_ticks + msg.time >= ticks_to_extract:
                    current_bar += 1
                    current_ticks = 0
                    bar_messages.append(metadata_info[-1])
                    save_bar_messages(bar_messages, current_bar, ticks_per_beat, filename, output_path)
                    bar_messages = metadata_info[:-1]
                    bar_messages.extend(chord_messages)
                    chord_messages = []
                # CASE 2: the chord is of the current section
                else:
                    current_ticks += msg.time
                    bar_messages.extend(chord_messages)
                    chord_messages = []

# IMPROVED VERSION
# def split_midi_by_bar(mid, bars_to_extract, output_path, filename, save_remaining=False):
#     """
#     Split a midi file by bar

#     Parameters: 
#         mid: mido.MidiFile, midi data
#         bars_to_extract: int, number of bars to extract
#         output_path: str, path to save the midi files
#         filename: str, filename
#         save_remaining: bool, save remaining messages
#     """
#     ticks_per_beat = mid.ticks_per_beat
#     numerator_time_signature = next((msg.numerator for msg in mid if msg.type == 'time_signature'), 4)
#     ticks_per_bar = ticks_per_beat * numerator_time_signature
#     ticks_to_extract = ticks_per_bar * bars_to_extract

#     current_bar = 0
#     current_ticks = 0
#     metadata_info = [msg for msg in mid if msg.is_meta]
#     bar_messages = []
#     chord_messages = []

#     for track in mid.tracks:
#         for msg in track:
#             chord_messages.append(msg)

#             if msg.time > 0: 
#                 current_ticks += msg.time

#                 if current_ticks >= ticks_to_extract:
#                     current_bar += 1
#                     current_ticks -= ticks_to_extract

#                     # Save the current bar messages
#                     bar_messages.extend(metadata_info)
#                     bar_messages.append(metadata_info[-1])
#                     save_bar_messages(bar_messages, current_bar, ticks_per_beat, filename, output_path=output_path)

#                     # Prepare for the next bar
#                     bar_messages = metadata_info[:-1]
#                     chord_messages.clear()

#     # Save remaining messages if any
#     if chord_messages and save_remaining:
#         current_bar += 1
#         bar_messages.extend(chord_messages)
#         save_bar_messages(bar_messages, current_bar, ticks_per_beat, filename)
