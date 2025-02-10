import os
import sys
import mido

def save_midi(mid, output_path):
    """
    Save MIDI file.
    
    Parameters:
        mid: MidiFile, MIDI file to save.
        output_path (str): Path to save the MIDI file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mid.save(output_path)
    # print(f"MIDI file saved to {output_path}")

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