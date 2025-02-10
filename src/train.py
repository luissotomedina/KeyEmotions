import os
import yaml
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.utils import *
from models.Transformer import *
from preprocessing.loader import Loader

def subsequent_mask(size):
    # mask = torch.ones(size, size)
    attn_shape = (size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def train(model, train_ldr, max_epochs, device, lr, pad_idx=181):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()
    log_interval = 5

    total_batches = len(train_ldr)
    
    for epoch in range(max_epochs):
        epoch_loss = 0
        for batch_idx, batch in enumerate(train_ldr):
            batch_time = time.time()
            src = batch[0].to(device)
            tgt = src.clone().to(device)

            tgt_input = tgt[:, :-1] # remove EOS token
            tgt_expected = tgt[:, 1:] # remove SOS token
            # tgt_mask = model.transformer.generate_square_subsequent_mask(tgt_input.size(1)).to(device)
            tgt_mask = subsequent_mask(tgt_input.size(1)).to(device)
            tgt_mask = tgt_mask.bool()

            # Create padding masks
            src_pad_mask = (src == pad_idx).to(device)  # Shape: [batch_size, src_len], True if pad_idx
            tgt_pad_mask = (tgt_input == pad_idx).to(device)  # Shape: [batch_size, tgt_len], True if pad_idx

            prediction = model(src, tgt_input, tgt_mask, src_pad_mask, tgt_pad_mask) # [batch_size, seq_len, vocab_size]

            # reshape predictions shape to [batch_size*seq_len, vocab_size] = tgt_expected
            prediction = prediction.permute(0, 2, 1)

            loss_val = loss_func(prediction, tgt_expected) # [batch_size, seq_len, vocab_size] vs [batch_size, seq_len]
            epoch_loss += loss_val.item()

            optimizer.zero_grad()
            loss_val.backward() # compute gradients
            optimizer.step()

            if batch_idx % log_interval == 0:
                cur_loss = epoch_loss / log_interval
                print(f"| epoch {epoch} | batch {batch_idx}/{total_batches} | lr {lr:.4f} | "
                      f"s/batch {time.time() - batch_time:.2f} | batch_loss {loss_val.item():.2f} | "
                      f"ppl {np.exp(cur_loss):.2f}")

        print("-----------------------------------------")
        print(f"\n | Epoch {epoch} | Total Loss: {epoch_loss:.2f}")
        print("-----------------------------------------")

    print("\Training complete")


if __name__ == "__main__":
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
    batch_size = config['training']['batch_size']
    lr = config['training']['lr']
    checkpoint_dir = config['training']['checkpoint_dir']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Transformer")

    print("Begin PyTorch Transformer seq-to-seq demo ")
    torch.manual_seed(1)
    np.random.seed(1)

    # 1. load data
    print("\nLoading data int-token train data")
    data_loader = Loader(train_midi_path, batch_size)
    train_ldr, _ = data_loader.create_training_dataset()

    # 2. create Transformer network
    print("\nCreating Transformer network")
    vocab_size = data_loader.vocab_size
    model = TransformerNet(vocab_size, d_model, n_head, num_layers, dropout=0.0).to(device)

    # 3. train network
    print("\nStarting training")
    train(model, train_ldr, max_epochs, device, lr)

    # 4. save trained model
    print("\nSaving trained model state")
    fn = os.path.join(checkpoint_dir, "transformer_seq_model.pt")
    model.eval()
    torch.save(model.state_dict(), fn)






    
