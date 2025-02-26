import os
import time
import yaml
import traceback

import torch
import torch.nn as nn
import torch.optim as optim

from utils.utils import *
from models.Transformer import *
from preprocessing.loader import Loader

# def subsequent_mask(size):
#     # mask = torch.ones(size, size)
#     attn_shape = (size, size)
#     subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
#     return torch.from_numpy(subsequent_mask) == 0

def train_iter(model, train_ldr, device, lr, total_batches, epoch, loss_func, 
               optimizer, scaler, logs_train, pad_idx, log_interval=10):
    # model.train()
    epoch_loss = 0
    batches_loss = 0

    print("-" * 89)
    
    for batch_idx, batch in enumerate(train_ldr):
        batch_time = time.time()
        # src, tgt = batch[0].to(device), batch[0].to(device)
        src = batch[0].to(device)
        tgt = src.clone().to(device)

        tgt_input = tgt[:, :-1] # remove EOS token
        tgt_expected = tgt[:, 1:] # remove SOS token
        tgt_mask = subsequent_mask(tgt_input.size(1)).to(device)
        # Create padding masks
        src_pad_mask = (src == pad_idx).to(device)  # Shape: [batch_size, src_len], True if pad_idx
        tgt_pad_mask = (tgt_input == pad_idx).to(device) # Shape: [batch_size, tgt_len], True if pad_idx

        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda'):
            prediction = model(src, tgt_input, tgt_mask, src_pad_mask, tgt_pad_mask) # [batch_size, seq_len, vocab_size]
            # reshape predictions shape to [batch_size*seq_len, vocab_size] = tgt_expected
            # prediction = prediction.reshape(-1, prediction.shape[-1]) # [batch_size*seq_len, vocab_size]
            prediction = prediction.view(-1, vocab_size)
            tgt_expected = tgt_expected.reshape(-1) # [batch_size*seq_len]
            # tgt_expected = tgt_expected.view(-1)
            loss_val = loss_func(prediction, tgt_expected) # [batch_size*seq_len, vocab_size] vs [batch_size*seq_len]

        scaler.scale(loss_val).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # to avoid exploding gradient, 0.5 or 1.0
        scaler.step(optimizer)
        scaler.update()

        batches_loss += loss_val.item()

        if batch_idx % log_interval == 0:
            epoch_loss += batches_loss
            cur_loss = batches_loss / log_interval
            elapsed = time.time() - batch_time
            # writer.add_scalar("Loss/train", cur_loss, epoch * total_batches + batch_idx)
            print("| epoch {:3d} | {:4d}/{:4d} batches | lr {:02.4f} | ms/batch {:5.2f} |"
                 "loss {:5.4f} | ppl {:8.2f} ".format(epoch, batch_idx, total_batches, lr,
                  elapsed * 1000/log_interval, cur_loss, np.exp(cur_loss)))
            logs_train.update({"epoch": epoch, "batches": batch_idx, "lr": lr, "ms/batch": elapsed * 1000/log_interval,
                                "loss": cur_loss, "ppl": np.exp(cur_loss)})
            batches_loss = 0
            batch_time = time.time()

    print("-" * 89)

    return epoch_loss

def evaluate(model, val_ldr, device, epoch, loss_func, logs_val, pad_idx):
    model.eval()
    epoch_loss = 0
    epoch_start_time = time.time()

    for batch_idx, batch in enumerate(val_ldr):
        src, tgt = batch[0].to(device), batch[0].to(device)

        tgt_input = tgt[:, :-1]
        tgt_expected = tgt[:, 1:]
        tgt_mask = subsequent_mask(tgt_input.size(1)).to(device).bool()

        src_pad_mask = (src == pad_idx).to(device)
        tgt_pad_mask = (tgt_input == pad_idx).to(device)

        output = model(src, tgt_input, tgt_mask, src_pad_mask, tgt_pad_mask)
        output = output.permute(0, 2, 1)

        loss_val = loss_func(output, tgt_expected)
        epoch_loss += len(src) * loss_val.item()

    epoch_time = time.time() - epoch_start_time
    print("-" * 89)
    print(f"| end of epoch {epoch:3d} | time: {epoch_time:5.2f}s | loss {epoch_loss:5.4f} |")
    print("-" * 89)

    logs_val.update({"epoch": epoch, "time": epoch_time, "loss": epoch_loss})

    return epoch_loss

def train(model, train_ldr, val_ldr, max_epochs, device, lr, logs_train, logs_val, exp_name, pad_idx):
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98)) # reduce lr if no improvement
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1) # reduce lr
    scaler = torch.amp.GradScaler(device)

    ce_loss = nn.CrossEntropyLoss(ignore_index=pad_idx).to(device)
    # mse_loss = nn.MSELoss()
    # l1_loss = nn.L1Loss()

    total_batches = len(train_ldr)

    train_loss, val_loss = [], []

    for epoch in range(max_epochs):
        train_loss.append(train_iter(model, train_ldr, device, lr, total_batches, epoch, 
                                ce_loss, optimizer, scaler, logs_train, pad_idx))
        val_loss.append(evaluate(model, val_ldr, device, epoch, ce_loss, logs_val, pad_idx))
        scheduler.step()
        print(f"Updated learning rate: {scheduler.get_last_lr()}")

    plot_loss(train_loss, val_loss, exp_name)
    print("\nTraining complete")

if __name__ == "__main__":
    try:
        config_path = './src/config/default.yaml'
        config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

        # DATA PATHS
        prepared_path = config['data']['prepared_dir']
        train_midi_path = config['data']['prepared']['train_data']
        valid_midi_path = config['data']['prepared']['val_data']

        # MODEL PARAMETERS  
        n_head = config['model']['n_head']
        num_layers = config['model']['num_layers']
        d_model = config['model']['d_model']
        num_layers = config['model']['num_layers']

        # TRAINING PARAMETERS
        max_epochs = config['training']['max_epochs']
        # max_epochs = 2
        batch_size = config['training']['batch_size']
        lr = config['training']['lr']
        experiments_dir = config['training']['experiments_dir']

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print("Begin PyTorch Transformer seq-to-seq demo ")
        torch.manual_seed(1)
        np.random.seed(1)

        # 1. load data
        print("\nLoading data int-token train data")
        data_loader_train = Loader(train_midi_path, batch_size)
        train_ldr, _ = data_loader_train.create_training_dataset()

        print("\nLoading data int-token valid data")
        data_loader_val = Loader(valid_midi_path, batch_size)
        valid_ldr, _ = data_loader_val.create_training_dataset()

        # 2. Create experiment environment
        print("\nCreating experiment environment")
        exp_dir, logs_folder, exp_name = create_exp_environment(experiments_dir) # create specific experiment directory for the current run

        logs_train_writer = LogsWriter(os.path.join(logs_folder, "logs_train.csv"), 
                                ["epoch", "batches", "lr", "ms/batch", "loss", "ppl"])
        
        logs_val_writer = LogsWriter(os.path.join(logs_folder, "logs_val.csv"),
                                    ["epoch", "time", "loss"])

        # 3. create Transformer network
        print("\nCreating Transformer network")
        vocab_size = data_loader_train.vocab_size
        pad_idx = data_loader_train.pad_idx
        model = TransformerNet(vocab_size, d_model, n_head, num_layers, dropout=0.0).to(device)

        # 4. train network
        print("\nStarting training")
        train(model, train_ldr, valid_ldr, max_epochs, device, lr, logs_train_writer, logs_val_writer, exp_name, pad_idx)

        # 5. save trained model
        print("\nSaving trained model state")
        save_exp_environment(exp_dir, model, config)


    except Exception as e:
        print(f"Error: {e}")
        traceback.print_stack()
        raise e  
    






        
