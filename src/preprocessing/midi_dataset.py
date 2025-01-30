import os
import pickle
import numpy as np
from pathlib import Path
from mido import MidiFile, MidiTrack

from ..utils import save_midi

def pitch_range(mid):
    """
    Get the pitch range of a MIDI file.

    Parameters:
        mid: MidiFile, MIDI file to get the pitch range.
    
    Returns:
        pitch_min: int, minimum pitch value.
        pitch_max: int, maximum pitch value.
    """
    pitches = [msg.note for track in mid.tracks for msg in track if msg.type == 'note_on' and msg.velocity > 0]

    if not pitches:
        return 0, 127
    
    min_pitch = min(pitches)
    max_pitch = max(pitches)
    return min_pitch, max_pitch


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

    for track in mid.tracks:
        for msg in track:
            current_time += msg.time
            
            if msg.type == 'note_on' and msg.velocity > 0:
                note_start_times[msg.note] = current_time
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                if msg.note in note_start_times:
                    duration = current_time - note_start_times[msg.note]
                    events.append({
                        'time': note_start_times[msg.note],
                        'note': msg.note,
                        'velocity': msg.velocity,
                        'duration': duration
                    })
                    del note_start_times[msg.note]

    return sorted(events, key=lambda x: x['time'])

def quantize_to_grid(time, ticks_per_beat, grid=16):
    ticks_per_bar = ticks_per_beat * 4
    bar = time // ticks_per_bar
    position = (time % ticks_per_bar) // (ticks_per_bar // grid)
    return bar, position


# TIMESIGN={'[6, 8]': 0, '[4, 4]': 1, '[9, 8]': 2, '[2, 4]': 3, '[3, 4]': 4, '[2, 2]': 5, '[6, 4]': 6, '[3, 2]': 7} # assuming 4/4 is the most common time signature
DURATION={2:0, 4:1, 6:2, 8:3, 10:4, 12:5, 16:6, 18:7, 20:8, 22:9, 24:10, 30:11, 32:12, 36:13, 42:14, 44:15, 48:16, 54:17, 56:18, 60:19,
          64:20, 66:21, 68:22, 72:23, 78:24, 80:25, 84:26, 90:27, 92:28, 96:29, 102:30, 108:31, 120:32, 126:33, 132:34, 138:35, 144:36}
def midi_to_REMI(mid):
    """
    Extracts the notes and chords from a MIDI file and returns a list of tokens in the REMI format.
    Format: 
        - 0: Bar
        - 1-15: Position
        - 16-75: Pitch
        - 76-112: Duration
    """
    ticks_per_beat = mid.ticks_per_beat

    events = extract_notes_and_chords(mid)

    tokens = []
    tokens_letras = []
    position_set = set()
    duration_set = set()
    current_bar = None
    for event in events:
        bar, position = quantize_to_grid(event['time'], ticks_per_beat)
        # print(bar, position)
        if current_bar is None or bar > current_bar:
            tokens.append(0)
            tokens_letras.append("Bar")
            current_bar = bar

        position_token = position + 1
        pitch_token = event['note'] - 42 + 16 # En ticks
        duration_token = DURATION[event['duration']] + 76
        
        tokens.append(position_token)
        tokens.append(pitch_token)
        tokens.append(duration_token)
        position_set.add(position)
        duration_set.add(event['duration'])

        tokens_letras.append(f"Position {position_token}")
        tokens_letras.append(f"Pitch {pitch_token}")
        tokens_letras.append(f"Duration {duration_token}")

    return tokens, tokens_letras, position_set, duration_set








