import os
import yaml
import time
import numpy as np
from mido import MidiFile, MidiTrack, Message

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import *
from preprocessing.loader import Loader
from models.Transformer import TransformerNet


def subsequent_mask(size):
    # mask = torch.ones(size, size)
    attn_shape = (size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


TIMESIGN={'[2, 2]': 0, '[2, 4]': 1, '[3, 4]': 2, '[4, 4]': 3, '[5, 4]': 4, '[6, 4]': 5, '[5, 8]': 6, '[6, 8]': 7, '[7, 8]': 8, '[9, 8]': 9}
TPB = {48: 0, 96: 1, 120: 2, 192: 3, 256: 4, 384: 5, 480: 6, 960: 7, 1024: 8}
def REMI_to_midi(remi_seq, grids_per_bar=32, SOS_ind=0, bar_ind=5, pos_ind=6, pitch_ind=38, 
                 duration_ind=130, timesign_ind=162, tpb_ind=172, EOS_ind=182):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    ticks_per_beat = 480 # default value
    time_sign = [4, 4] # default value

    current_tick = 0
    current_bar = 0
    active_notes = {}

    i = 0
    while i < len(remi_seq):
        token = remi_seq[i]

        if token == EOS_ind:
            break
        elif token == SOS_ind:
            continue
        elif token in range(tpb_ind, EOS_ind):
            ticks_per_beat = list(TPB.keys())[list(TPB.values()).index(token-tpb_ind)]
        elif token in range(timesign_ind, tpb_ind):
            time_sign = list(TIMESIGN.keys())[list(TIMESIGN.values()).index(token-timesign_ind)]
        elif token == bar_ind:
            current_bar += 1
            current_tick = current_bar * grids_per_bar * (ticks_per_beat // grids_per_bar) # revisar
        elif token in range(pos_ind, duration_ind):
            position = token - pos_ind
            i += 1
            if i >= len(remi_seq):
                break
            pitch_token = remi_seq[i]

            i += 1
            if i >= len(remi_seq):
                break
            duration_token = remi_seq[i]

            if pitch_token in range(pitch_ind, duration_ind) and duration_token in range(duration_ind, timesign_ind):
                pitch = pitch_token - pitch_ind + 20 # min pitch is 20
                duration = (duration_token - duration_ind) * (ticks_per_beat // grids_per_bar)

                note_start = current_tick + (position * (ticks_per_beat // grids_per_bar))
                note_end = note_start + duration

                track.append(Message('note_on', note=pitch, velocity=64, time=note_start - (active_notes.get(pitch, note_start))))
                track.append(Message('note_off', note=pitch, velocity=64, time=note_end - note_start))

                active_notes[pitch] = note_end

        i += 1

    mid.ticks_per_beat = ticks_per_beat
    return mid    

def generate_sequence(model, d_model, token, device, max_bar = 8, max_len=1300, sos_token=0, eos_token=182, pad_idx=181, bar_token=5):
    model.eval()
    # generated = torch.tensor([[sos_token, token]], device=device)
    generated = torch.tensor([[token]], device=device)
    bars = 0
    for _ in range(max_len - 2):
        tgt_mask = subsequent_mask(generated.size(1)).to(device)

        tgt_emb = model.embedding(generated)
        tgt_emb = model.pos_enc(tgt_emb)
        tgt_emb = model.norm(tgt_emb)

        memory = torch.zeros(1, 1, d_model, device=device)

        output = model.decoder(tgt_emb, memory=memory, tgt_mask=tgt_mask)

        logits = model.fc_out(output[:, -1, :])
        next_token = torch.argmax(logits, dim=-1).unsqueeze(0)

        if next_token.item() == bar_token:
            bars += 1
            if bars > max_bar:
                break

        generated = torch.cat([generated, next_token], dim=1)

    return generated.squeeze(0).tolist()

def save_generations(generation, path):
    txt_path = os.path.join(path, "generations.txt")

    # if not os.path.exists(txt_path):
    #     with open(txt_path, 'w') as f:
    #         pass

    with open(txt_path, 'a') as f:
        f.write(f"{generation}\n")


if __name__ == "__main__":
    config_path = './src/config/default.yaml'
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

    # MODEL PATH
    experiments_dir = config['training']['experiments_dir']

    # MODEL PARAMETERS
    vocab_size = config['model']['vocab_size']
    d_model = config['model']['d_model']
    nhead = config['model']['n_head']
    num_layers = config['model']['num_layers']
    max_epochs = config['training']['max_epochs']
    batch_size = config['training']['batch_size']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    print("\nCreating new Transformer seq-to-seq model")
    model = TransformerNet(vocab_size, d_model, nhead, num_layers).to(device)

    # Load trained model
    # exp_num = len(os.listdir(experiments_dir))
    exp_num = 5
    exp_name = f"exp_{exp_num}"
    weigths_path = os.path.join(experiments_dir, exp_name, "weigths.pt")
    generations_path = os.path.join(experiments_dir, exp_name, "generations")
    print(f"\nLoading trained model state from {exp_name}")
    model.load_state_dict(torch.load(weigths_path, map_location=device, weights_only=True))

    # Generate sequence
    max_len = 1300
    # emotion_token = 4

    for emotion_token in range(1, 5):
        generation = generate_sequence(model, d_model, emotion_token, device, max_len=max_len)
        save_generations(generation, generations_path)

    # Save generated sequence
    print("\nSaving generated sequence")

