import os
import json
import torch
from mido import MidiFile, MidiTrack, Message, MetaMessage
import torch.nn.functional as F
from models.KeyEmotions import KeyEmotions

class KeyEmotionsGenerator:
    """
    A class for generating music sequences using the KeyEmotions model.
    Handles sequence generation, MIDI conversion, and saving results.
    """
    
    STATE_TEMPO = 0
    STATE_TIME_SIGNATURE = 1
    STATE_BAR = 2
    STATE_POS = 3
    STATE_DUR = 4
    STATE_PITCH = 5
    
    SOS_IDX = 0
    BAR_IDX = 5
    POS_IDX = 6
    PITCH_IDX = 38
    DURATION_IDX = 128
    TIMESIGN_IDX = 160
    TPB_IDX = 170
    EOS_IDX = 180
    
    def __init__(self, experiments_dir='./experiments'):
        self.experiments_dir = experiments_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.config = None
        
    def load_experiment(self, exp_num):
        """Load model and configuration for a specific experiment number"""
        exp_name = f"exp_{exp_num}"
        config_path = os.path.join(self.experiments_dir, exp_name, "config.json")
        weights_path = os.path.join(self.experiments_dir, exp_name, "weigths.pt")
        
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load config: {str(e)}")
        
        # Initialize model
        try:
            self.model = KeyEmotions(
                vocab_size=self.config['model']['vocab_size'],
                d_model=self.config['model']['d_model'],
                nhead=self.config['model']['n_head'],
                num_layers=self.config['model']['num_layers'],
                d_ff=self.config['model']['d_ff'],
            ).to(self.device)
            
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device, weights_only=True))
        except Exception as e:
            raise RuntimeError(f"Model initialization failed: {str(e)}")
        
        return self
    
    def generate_sequence(self, emotion, max_len=1300, max_bar=8, temperature=1.0):
        """Generate a musical sequence for the given emotion"""
        if emotion not in range(1, 5):
            raise ValueError("Emotion must be in range [1, 4]")
        if not self.model:
            raise RuntimeError("Model not loaded. Call load_experiment() first.")
        
        self.model.eval()
        sequence = [self.SOS_IDX, emotion]
        num_bars = 0
        state = self.STATE_TEMPO
        
        with torch.no_grad():
            for _ in range(max_len):
                src = torch.tensor([sequence], dtype=torch.int64).to(self.device)
                
                tgt_mask = self.model.subsequent_mask(src.size(1)).to(self.device)
                tgt_pad_mask = self.model.create_pad_mask(src, pad_idx=179).to(self.device)
                
                output = self.model(src, tgt_mask, tgt_pad_mask)
                logits = output[:, -1, :] / temperature
                
                # Apply state-specific masks
                if state == self.STATE_TEMPO:
                    logits[:, :170] = -float('inf')
                    logits[:, 179:] = -float('inf')
                elif state == self.STATE_TIME_SIGNATURE:
                    logits[:, :160] = -float('inf')
                    logits[:, 170:] = -float('inf')
                elif state == self.STATE_BAR:
                    logits[:, :5] = -float('inf')
                    logits[:, 6:] = -float('inf')
                    num_bars += 1
                elif state == self.STATE_POS:
                    logits[:, :6] = -float('inf')
                    logits[:, 38:] = -float('inf')
                elif state == self.STATE_DUR:
                    logits[:, :128] = -float('inf')
                    logits[:, 160:] = -float('inf')
                elif state == self.STATE_PITCH:
                    logits[:, :38] = -float('inf')
                    logits[:, 128:] = -float('inf')
                
                probs = F.softmax(logits, dim=-1)
                
                if state > self.STATE_PITCH:
                    if probs[:, 5].item() > probs[:, 6:38].max().item():
                        next_token = 5
                        state = self.STATE_POS
                        num_bars += 1
                    else:
                        next_token = torch.multinomial(probs[:, 6:38], num_samples=1).item() + 6
                        state = self.STATE_DUR
                else:
                    next_token = torch.multinomial(probs, num_samples=1).item()
                    state += 1
                
                sequence.append(next_token)
                
                if next_token == self.EOS_IDX or num_bars > max_bar:
                    break
        
        return sequence
    
    @staticmethod
    def remi_to_midi(remi_sequence, output_path):
        """Convert a REMI sequence to MIDI file"""
        TIMESIGN = {'[2, 2]': 0, '[2, 4]': 1, '[3, 4]': 2, '[4, 4]': 3, '[5, 4]': 4,
                   '[6, 4]': 5, '[5, 8]': 6, '[6, 8]': 7, '[7, 8]': 8, '[9, 8]': 9}
        TPB = {48: 0, 96: 1, 120: 2, 192: 3, 256: 4, 384: 5, 480: 6, 960: 7, 1024: 8}
        
        time_signature = None
        ticks_per_beat = None
        emotion = None
        
        for token in remi_sequence:
            if 160 <= token <= 169:  # TIME_SIGNATURE
                signature = list(TIMESIGN.keys())[token - 160]
                time_signature = [int(x) for x in signature.strip('[]').split(',')]
            elif 170 <= token <= 178:  # TEMPO
                ticks_per_beat = list(TPB.keys())[token - 170]
            elif 1 <= token <= 4:  # Emotion
                emotion = token
            elif token == 5:  # BAR
                break
        
        if not time_signature or not ticks_per_beat:
            raise ValueError("Could not extract time signature or tempo from sequence")
        

        mid = MidiFile(ticks_per_beat=ticks_per_beat)
        track = MidiTrack()
        mid.tracks.append(track)
        track.append(MetaMessage('time_signature', 
                               numerator=time_signature[0], 
                               denominator=time_signature[1], 
                               time=0))
        
        # Process notes
        current_bar = 0
        events = []
        i = 0
        
        while i < len(remi_sequence):
            token = remi_sequence[i]
            
            if token == KeyEmotionsGenerator.SOS_IDX:
                i += 1
            elif KeyEmotionsGenerator.SOS_IDX < token < KeyEmotionsGenerator.BAR_IDX:
                i += 1  # Emotion already captured
            elif token == KeyEmotionsGenerator.BAR_IDX:
                current_bar += 1
                i += 1
            elif KeyEmotionsGenerator.POS_IDX <= token < KeyEmotionsGenerator.PITCH_IDX:
                if i + 2 < len(remi_sequence):
                    pos = token - KeyEmotionsGenerator.POS_IDX
                    dur_token = remi_sequence[i + 1]
                    pitch_token = remi_sequence[i + 2]
                    
                    if (KeyEmotionsGenerator.DURATION_IDX <= dur_token < KeyEmotionsGenerator.TIMESIGN_IDX and
                        KeyEmotionsGenerator.PITCH_IDX <= pitch_token < KeyEmotionsGenerator.DURATION_IDX):
                        
                        ticks_per_bar = (time_signature[0] * ticks_per_beat * 4) // time_signature[1]
                        ticks_per_grid = ticks_per_bar // 32
                        
                        start_time = int(current_bar * ticks_per_bar + pos * ticks_per_grid)
                        duration = (dur_token - KeyEmotionsGenerator.DURATION_IDX + 1) * ticks_per_grid
                        end_time = start_time + duration
                        pitch = pitch_token - KeyEmotionsGenerator.PITCH_IDX + 20  # MIDI note 20-107
                        
                        events.append({'type': 'note_on', 'time': start_time, 'note': pitch, 'velocity': 64})
                        events.append({'type': 'note_off', 'time': end_time, 'note': pitch, 'velocity': 64})
                    
                    i += 3
                else:
                    i += 1
            elif token == KeyEmotionsGenerator.EOS_IDX:
                break
            else:
                i += 1
        
        events.sort(key=lambda x: x['time'])
        current_time = 0
        
        for event in events:
            delta = event['time'] - current_time
            if event['type'] == 'note_on':
                track.append(Message('note_on', note=event['note'], velocity=event['velocity'], time=delta))
            elif event['type'] == 'note_off':
                track.append(Message('note_off', note=event['note'], velocity=event['velocity'], time=delta))
            current_time = event['time']
        
        os.makedirs(output_path, exist_ok=True)
        midi_path = os.path.join(output_path, f"{emotion}_generation.mid")
        mid.save(midi_path)
        return midi_path
    
    def generate_and_save(self, emotion, exp_num, output_path=None, max_len=1300, max_bar=8, temperature=1.0):
        """Complete generation pipeline"""
        if output_path is None:
            output_path = os.path.join(self.experiments_dir, f"exp_{exp_num}", "generations")
        
        self.load_experiment(exp_num)
        sequence = self.generate_sequence(emotion, max_len, max_bar, temperature)
        midi_path = self.remi_to_midi(sequence, output_path)
        
        txt_path = os.path.join(output_path, "generations.txt")
        with open(txt_path, 'a') as f:
            f.write(f"{sequence}\n")
        
        return sequence, midi_path


if __name__ == '__main__':
    generator = KeyEmotionsGenerator()
    
    for exp_num in range(20, 21):
        print(f"Processing experiment {exp_num}")
        
        for emotion in range(1, 5):
            try:
                print(f"Generating for emotion {emotion}...")
                seq, midi = generator.generate_and_save(emotion, exp_num)
                print(f"Success! Saved to {midi}")
            except Exception as e:
                print(f"Failed to generate for emotion {emotion}: {str(e)}")