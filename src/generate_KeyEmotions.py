import os
import yaml

from mido import MidiFile, MidiTrack, Message, MetaMessage

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import *
from utils.midi_utils import *
from models.KeyEmotions import KeyEmotions

def generate_sequence(model, emotion, sos_idx, eos_idx, pad_idx, max_len, max_bar, device, temperature=1.0):
    
    model.eval()
    sequence = [sos_idx, emotion]  # Inicia con [SOS, EMOTION]

    num_bars = 0

    STATE_TEMPO = 0
    STATE_TIME_SIGNATURE = 1
    STATE_BAR = 2
    STATE_POS = 3
    STATE_DUR = 4
    STATE_PITCH = 5

    state = STATE_TEMPO

    with torch.no_grad():
        for _ in range(max_len):
            src = torch.tensor([sequence], dtype=torch.int64).to(device)

            tgt_mask = model.subsequent_mask(src.size(1)).to(device)
            tgt_pad_mask = model.create_pad_mask(src, pad_idx=pad_idx).to(device)

            output = model(src, tgt_mask, tgt_pad_mask)

            # Aplicar temperatura a las logits
            logits = output[:, -1, :] / temperature

            if state == STATE_TEMPO:
                logits[:, :170] = -float('inf')
                logits[:, 179:] = -float('inf')
            elif state == STATE_TIME_SIGNATURE:
                logits[:, :160] = -float('inf')
                logits[:, 170:] = -float('inf')
            elif state == STATE_BAR:
                logits[:, :5] = -float('inf')
                logits[:, 6:] = -float('inf')
                num_bars += 1
            elif state == STATE_POS:
                logits[:, :6] = -float('inf')
                logits[:, 38:] = -float('inf')
            elif state == STATE_DUR:
                logits[:, :128] = -float('inf')
                logits[:, 160:] = -float('inf')
            elif state == STATE_PITCH:
                logits[:, :38] = -float('inf')
                logits[:, 128:] = -float('inf')

            probs = F.softmax(logits, dim=-1)

            if state > STATE_PITCH:
                bar_probs = probs[:, 5].item()
                # pos_probs = probs[:, 6:38].sum().item()
                pos_probs = probs[:, 6:38].max().item()

                if bar_probs > pos_probs:
                    next_token = 5
                    state = STATE_POS
                    num_bars += 1
                else: 
                    next_token = torch.multinomial(probs[:, 6:38], num_samples=1).item() + 6
                    state = STATE_DUR
            
            else: 
                next_token = torch.multinomial(probs, num_samples=1).item()
                state += 1

            sequence.append(next_token) 

            if next_token == eos_idx or num_bars > max_bar:
                break
    
    return sequence

def remi_to_midi(remi_sequence, path):
    SOS_ind = 0
    bar_ind = 5
    pos_ind = 6
    pitch_ind = 38
    duration_ind = 128
    timesign_ind = 160
    tpb_ind = 170
    EOS_ind = 180

    TIMESIGN={'[2, 2]': 0, '[2, 4]': 1, '[3, 4]': 2, '[4, 4]': 3, '[5, 4]': 4, 
              '[6, 4]': 5, '[5, 8]': 6, '[6, 8]': 7, '[7, 8]': 8, '[9, 8]': 9}
    TPB = {48: 0, 96: 1, 120: 2, 192: 3, 256: 4, 384: 5, 480: 6, 960: 7, 1024: 8}

    for token in remi_sequence:
        if 160 <= token <= 169: # TIME_SIGNATURE
            signature = list(TIMESIGN.keys())[token - 160]
            time_signature = [int(x) for x in signature.strip('[]').split(',')]
        elif 170 <= token <= 178: # TEMPO
            ticks_per_beat = list(TPB.keys())[token - 170]
        elif token == 5: # BAR
            break
        
    mid = MidiFile(ticks_per_beat=ticks_per_beat)
    track = MidiTrack()
    mid.tracks.append(track)
    track.append(MetaMessage('time_signature', numerator=time_signature[0], denominator=time_signature[1], time=0))
    
    current_bar = 0
    events = []

    i = 0
    while i < len(remi_sequence):
        token = remi_sequence[i]

        if token == SOS_ind:
            i += 1
        elif SOS_ind < token < bar_ind: # Emotion
            emotion = token
            i += 1
        elif token == bar_ind:
            current_bar += 1
            i += 1
        elif pos_ind <= token < pitch_ind: # POS (Note start)
            if i + 2 < len(remi_sequence): # Check if there are enough tokens to form a note 
                pos = token - pos_ind
                duration_token = remi_sequence[i + 1]
                pitch_token = remi_sequence[i + 2]

                if duration_ind <= duration_token < timesign_ind \
                    and pitch_ind <= pitch_token < duration_ind:
                    ticks_per_bar = calculate_ticks_per_bar(time_signature, ticks_per_beat)
                    ticks_per_grid = ticks_per_bar // 32
                    start_time = int(current_bar * ticks_per_bar + pos * ticks_per_grid)
                    end_time = int(start_time + (duration_token - duration_ind + 1) * ticks_per_grid)
                    pitch = pitch_token - pitch_ind + 20 # 20 is the lowest pitch in MIDI

                    events.append({'type': 'note_on', 'time': start_time, 'note': pitch, 'velocity': 64})
                    events.append({'type': 'note_off', 'time': end_time, 'note': pitch, 'velocity': 64})
                
                current_time = end_time 
                i += 3 # Skip to next POS token or BAR token
            else: 
                i += 1 # Skip to next token
        elif token == EOS_ind:
            break
        else: 
            i += 1  # Skip to next token for unknown tokens, TPB, Time_Signature, etc. 


    events = sorted(events, key=lambda x: x['time'])

    current_time = 0
    for event in events:
        if event['type'] == 'note_on':
            track.append(Message('note_on', note=event['note'], velocity=event['velocity'], time=event['time'] - current_time))
        elif event['type'] == 'note_off':
            track.append(Message('note_off', note=event['note'], velocity=event['velocity'], time=event['time'] - current_time))
        current_time = event['time']

    file = os.path.join(path, f"{emotion}_generation.mid")

    mid.save(file)

def save_generations(generation, path):
    txt_path = os.path.join(path, "generations.txt")

    with open(txt_path, 'a') as f:
        f.write(f"{generation}\n")

    remi_to_midi(generation, path)

if __name__ == '__main__':
    exp_num = 9
    exp_name = f"exp_{exp_num}"

    experiments_dir = './experiments'
    config_path = os.path.join(experiments_dir, exp_name, "config.json")
    weigths_path = os.path.join(experiments_dir, exp_name, "weigths.pt")
    generations_path = os.path.join(experiments_dir, exp_name, "generations")

    with open(config_path, "r", encoding="utf-8") as json_file:
        config = json.load(json_file)

    # config_path = './src/config/default.yaml'
    # config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

    # MODEL PATH
    experiments_dir = config['training']['experiments_dir']

    # MODEL PARAMETERS
    vocab_size = config['model']['vocab_size']
    d_model = config['model']['d_model']
    nhead = config['model']['n_head']
    num_layers = config['model']['num_layers']
    d_ff = config['model']['d_ff']
    max_epochs = config['training']['max_epochs']
    batch_size = config['training']['batch_size']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = KeyEmotions(
        vocab_size=vocab_size, 
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        d_ff=d_ff,
    ).to(device)

    # exp_num = 7
    # exp_name = f"exp_{exp_num}"
    # weigths_path = os.path.join(experiments_dir, exp_name, "weigths.pt")
    # generations_path = os.path.join(experiments_dir, exp_name, "generations")
    model.load_state_dict(torch.load(weigths_path, map_location=device, weights_only=True))

    max_len = 1300

    output_path = os.path.join(generations_path, f"gen_{len(os.listdir(generations_path))}")
    os.makedirs(output_path, exist_ok=True)
    for emotion_token in range(1, 5):
        generation = generate_sequence(
            model, emotion_token, sos_idx=0, eos_idx=180, pad_idx=179, max_len=max_len, max_bar=8, device=device
        )
        save_generations(generation, output_path)