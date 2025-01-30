import os
import yaml
from tqdm import tqdm
from pathlib import Path
from miditok.pytorch_data import DataCollator, DatasetMIDI

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import *
from models.transformer_VAE import *
from preprocessing.data_loader import create_tokenizer, create_dataset

def create_data_loaders(dataset_train, dataset_valid, tokenizer):
    """
    Create data loaders for training and validation datasets.
    
    Parameters:
        dataset_train: DatasetMIDI, training dataset.
        dataset_valid: DatasetMIDI, validation dataset.
        tokenizer: REMI, tokenizer to use for processing.
        
    Returns:
        train_loader: DataLoader, training data loader.
        valid_loader: DataLoader, validation data loader.
    """
    collator = DataCollator(tokenizer.pad_token_id)
    train_loader = DataLoader(dataset=dataset_train, collate_fn=collator)
    valid_loader = DataLoader(dataset=dataset_valid, collate_fn=collator)
    return train_loader, valid_loader

def train_model(epoch, model, train_loader, val_loader, optimizer, scheduler, device='cuda'):
    """
    Train the TransformerVAE model for one epoch and evaluate on the validation set.

    Parameters:
        epoch (int): Current epoch number.
        model (nn.Module): The TransformerVAE model.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        optimizer (optim.Optimizer): Optimizer for training.
        scheduler (lr_scheduler): Learning rate scheduler.
        device (str): Device to use ('cuda' or 'cpu').

    Returns:
        None
    """
    pass



if __name__ == "__main__":
    BEAT_RES = {(0, 1): 12, (1, 2): 4, (2, 4): 2, (4, 8): 1}
    TOKENIZER_PARAMS = {
        "pitch_range": (21, 109),
        "beat_res": BEAT_RES,
        "num_velocities": 24,
        "special_tokens": ["PAD", "BOS", "EOS"],
        "use_chords": True,
        "use_rests": True,
        "use_tempos": True,
        "use_time_signatures": True,
        "use_programs": False,  # no multitrack here
        "num_tempos": 32,
        "tempo_range": (50, 200),  # (min_tempo, max_tempo)
    }

    config_path = './src/config/default.yaml'
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

    prepared_path = config['data']['prepared_dir']
    train_data_path = config['data']['prepared']['train']
    val_data_path = config['data']['prepared']['val']

    vocab_size = config['data']['vocab_size']
    max_seq_len = config['data']['max_seq_len']

    tokenizer, midi_paths = create_tokenizer(TOKENIZER_PARAMS, prepared_path, vocab_size=vocab_size, output_path=prepared_path)
    dataset_train = DatasetMIDI(
        files_paths=list(Path(train_data_path).glob("**/*.mid")),
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        bos_token_id=tokenizer["BOS_None"],
        eos_token_id=tokenizer["EOS_None"],
    )
    dataset_valid = DatasetMIDI(
        files_paths=list(Path(val_data_path).glob("**/*.mid")),
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        bos_token_id=tokenizer["BOS_None"],
        eos_token_id=tokenizer["EOS_None"],
    )
    collator = DataCollator(tokenizer.pad_token_id)
    train_loader = DataLoader(dataset=dataset_train, collate_fn=collator)
    val_loader = DataLoader(dataset=dataset_valid, collate_fn=collator)


    vocab_size = config['model']['vocab_size']
    d_model = config['model']['d_model']
    n_head = config['model']['n_head']
    num_layers = config['model']['num_layers']
    latent_dim = config['model']['lattent_dim']
    emotion_dim = config['model']['emotion_dim']
    max_seq_len = config['model']['max_seq_len']

    max_lr, min_lr = config['training']['max_lr'], config['training']['min_lr']
    lr_decay_steps = config['training']['lr_decay_steps']

    model = TransformerVAE(vocab_size, d_model, n_head, num_layers, latent_dim, emotion_dim, max_seq_len)
    model.train()
    opt_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(opt_params, lr=max_lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, lr_decay_steps, eta_min=min_lr
    )
    # for ep in range(config['training']['max_epochs']):
    #     train_model(ep+1, model, train_loader, val_loader, optimizer, scheduler)
