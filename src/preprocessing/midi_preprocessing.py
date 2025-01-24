import mido

def load_midi(midi_path):
    """
    Load a midi file
    
    Input:
        midi_path: str, path to the midi file
    
    Output:
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
    
    Input:
        mid: mido.MidiFile, midi data

    Output:
        density: float, note density
    """
    duration = mid.length
    n_notes = sum([1 for track in mid.tracks for msg in track if msg.type == 'note_on'])
    density = n_notes / duration
    return density

def get_tempo(mid):
    """
    Get tempo of a midi file

    Input:
        mid: mido.MidiFile, midi data
    
    Output:
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

    Input:
        mid: mido.MidiFile, midi data

    Output:
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
    
    Input:
        mid: mido.MidiFile, midi data
    
    Output:
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
    
    Input:
        mid: mido.MidiFile, midi data
        duration: float, duration of the midi file in seconds
        tempo: int, tempo
        numerator: int, numerator of the time signature
    
    Output: 
        bar_length: float, bar length
    """
    beats_per_song = tempo * (duration / 60)
    bars = round(beats_per_song / numerator, 0)
    return bars
    

def get_midi_features(mid):
    """
    Get features of a midi file

    Input:
        mid: mido.MidiFile, midi data
    
    Output:
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