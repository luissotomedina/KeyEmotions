import os 
import time
import yaml
import traceback

import torch
import torch.nn as nn
import torch.optim as optim

from utils.utils import *
from models.KeyEmotions import *
from preprocessing.loader import Loader

def train_iter(model, train_ldr, lr, total_batches, epoch, criterion, 
               optimizer, logs_train, pad_idx, log_interval=10):
    
    epoch_loss = 0

    print("-" * 89)
    
    for batch_idx, (src, tgt) in enumerate(train_ldr):
        batch_time = time.time()    
        src = src.to(device)
        tgt = tgt.to(device)

        optimizer.zero_grad()

        tgt_mask = model.subsequent_mask(tgt.size(1)).to(device)
        tgt_pad_mask = model.create_pad_mask(tgt, pad_idx=pad_idx).to(device)

        output = model(src, tgt_mask, tgt_pad_mask)

        loss = criterion(output.view(-1, vocab_size), tgt.view(-1))

        epoch_loss += loss.item()

        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            elapsed = time.time() - batch_time
            print("| epoch {:3d} | {:4d}/{:4d} batches | lr {:02.4f} | ms/batch {:5.2f} |"
                 "loss {:5.4f} | ppl {:8.2f} ".format(epoch+1, batch_idx, total_batches, lr,
                  elapsed * 1000/log_interval, loss.item(), np.exp(loss.item())))
            logs_train.update({"epoch": epoch+1, "batches": batch_idx, "lr": lr, "ms/batch": elapsed * 1000/log_interval,
                                "loss": loss.item(), "ppl": np.exp(loss.item())})
            batch_time = time.time()
        
    print("-" * 89)

    return epoch_loss / total_batches

def validate(model, val_ldr, epoch, criterion, logs_val, pad_idx, device):
    model.eval()

    epoch_loss = 0
    epoch_start_time = time.time()

    with torch.no_grad():
        for _, (src, tgt) in enumerate(val_ldr):
            src = src.to(device)
            tgt = tgt.to(device)

            tgt_mask = model.subsequent_mask(tgt.size(1)).to(device)
            tgt_pad_mask = model.create_pad_mask(tgt, pad_idx=pad_idx).to(device)

            output = model(src, tgt_mask, tgt_pad_mask)

            loss = criterion(output.view(-1, vocab_size), tgt.view(-1))

            epoch_loss += loss.item()
    
    epoch_time = time.time() - epoch_start_time
    print("-" * 89)
    print(f"| end of epoch {epoch+1:3d} | time: {epoch_time:5.2f}s | loss {epoch_loss:5.4f} |")
    print("-" * 89)

    logs_val.update({"epoch": epoch+1, "time": epoch_time, "loss": epoch_loss})

    return epoch_loss / len(val_ldr)


def train(model, train_ldr, validation_ldr, num_epochs, lr, experiments_dir, pad_idx, config, device):
    # Create training environment
    exp_dir, logs_folder, exp_name = create_exp_environment(experiments_dir)

    model = model.to(device)
    model.train()

    logs_train_writer = LogsWriter(os.path.join(logs_folder, "logs_train.csv"), 
                                ["epoch", "batches", "lr", "ms/batch", "loss", "ppl"])
        
    logs_val_writer = LogsWriter(os.path.join(logs_folder, "logs_val.csv"),
                                ["epoch", "time", "loss"])


    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=0.1) # label smoothing to avoid overfitting

    # Early stopping
    best_val_loss = float('inf')
    patience = 5
    epochs_no_improve = 0

    total_batches = len(train_ldr)

    train_loss, validation_loss = [], []

    for epoch in range(num_epochs):
        try: 
            epoch_loss = train_iter(model, train_ldr, lr, total_batches, epoch, criterion,
                                    optimizer, logs_train_writer, pad_idx)
            train_loss.append(epoch_loss)

            scheduler.step()

            val_loss = validate(model, validation_ldr, epoch, criterion, logs_val_writer, pad_idx, device)
            validation_loss.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                # plot_loss(train_loss, validation_loss, exp_name)
                save_exp_environment(exp_dir, model, config)

            else:
                epochs_no_improve += 1
                if epochs_no_improve == patience:
                    print("Early stopping")
                    break

        except Exception as e: 
            print(f"Error during epoch {epoch}: {e}")
            continue

        torch.cuda.empty_cache()

    plot_loss(train_loss, validation_loss, exp_name)


if __name__ == "__main__":

    config_path = './src/config/default.yaml'
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

    # Data paths
    train_midi_path = config['data']['prepared']['train_data']
    validation_midi_path = config['data']['prepared']['val_data']

    # Model parameters
    nhead = config['model']['n_head']
    num_layers = config['model']['num_layers']
    d_ff = config['model']['d_ff']
    d_model = config['model']['d_model']
    dropout = config['model']['dropout']

    # Training parameters
    max_epochs = config['training']['max_epochs']
    # max_epochs = 1
    batch_size = config['training']['batch_size']
    lr = config['training']['lr']

    experiments_dir = config['training']['experiments_dir']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_loader_train = Loader(
        tokenized_data_path=train_midi_path,
        batch_size=batch_size
    )
    train_ldr = data_loader_train.create_dataloader(
        shuffle=True, 
        drop_last=True
    )

    data_loader_val = Loader(
        tokenized_data_path=validation_midi_path,
        batch_size=batch_size
    )
    val_ldr = data_loader_val.create_dataloader(
        shuffle=True, 
        drop_last=True
    )

    vocab_size = data_loader_train.vocab_size
    pad_idx = data_loader_train.pad_idx

    model = KeyEmotions(
        vocab_size=vocab_size, 
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout=dropout
    ).to(device)

    train(
        model = model, 
        train_ldr=train_ldr, 
        validation_ldr=val_ldr,
        num_epochs=max_epochs, 
        lr=lr, 
        experiments_dir=experiments_dir, 
        pad_idx=pad_idx, 
        config=config,
        device=device
    )

