"""
midi_utils.py

Utilities for MIDI file processing
"""

import os
import time
import fluidsynth

def save_midi(mid, output_path):
    """
    Save MIDI file.
    
    Parameters:
        mid: MidiFile, MIDI file to save.
        output_path (str): Path to save the MIDI file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mid.save(output_path)

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

def out_of_range_pitch(mid, min_pitch=20, max_pitch=109):
    """
    Check if the MIDI file is out of the pitch range.

    Parameters:
        mid: MidiFile, MIDI file to check the pitch range.
        min_pitch: int, minimum pitch value.
        max_pitch: int, maximum pitch value.
    
    Returns:
        out_of_range: bool, True if the MIDI file is out of the pitch range
    """
    for track in mid.tracks:
        for msg in track:
            if msg.type in ['note_on', 'note_off'] and (msg.note < min_pitch or msg.note > max_pitch):
                return True
            
def more_than_one_time_signature(mid):
    """
    Check if the MIDI file has more than one time signature.

    Parameters:
        mid: MidiFile, MIDI file to check the time signature.
    
    Returns:
        more_than_one: bool, True if the MIDI file has more than one time signature.
    """
    time_signatures = {(msg.numerator, msg.denominator) for track in mid.tracks for msg in track if msg.type == 'time_signature'}
    return len(time_signatures) > 1

def calculate_ticks_per_bar(time_signature, ticks_per_beat):
    """
    Calculate the number of ticks per bar based on the time signature and ticks per beat.

    Parameters:
        time_signature: tuple, time signature (numerator, denominator).
        ticks_per_beat: int, number of ticks per beat.

    Returns: 
        ticks_per_bar: int, number of ticks per bar.
    """
    if time_signature[1] == 8 and time_signature[0] % 3 == 0:
        f = 3 # Compound time signature
    else:
        f = 1 # Simple time signature
    
    ticks_per_bar = ticks_per_beat * (time_signature[0] / f)
    return ticks_per_bar

def midi_to_wav(midi_file, soundfont, output_wav):
    """
    Convert MIDI file to WAV using FluidSynth.
    
    Parameters:
        midi_file (str): Path to the MIDI file.
        soundfont (str): Path to the SoundFont file.
        output_wav (str): Path to save the output WAV file.
    """
    print(f"Converting {midi_file} to {output_wav}")
    fs = fluidsynth.Synth()
    fs.start(driver="dsound") # Windows, use "alsa" for Linux
    
    sfid = fs.sfload(soundfont)
    fs.program_select(0, sfid, 0, 0)
    
    fs.midi_file_play(midi_file)
    time.sleep(2)

    fs.audio_file(output_wav)
    
    fs.delete()