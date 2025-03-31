"""
train.py

Training script for KeyEmotions model.
"""

import os
import time
import yaml
import torch
import random
import traceback
import numpy as np
import torch.nn as nn
import torch.optim as optim

from utils.utils import *
from models.KeyEmotions import *
from preprocessing.loader import Loader

class Trainer:
    """
    Trainer class for training the KeyEmotions model.
    """
    def __init__(self, config_path, device):
        """
        Initialize the Trainer class.

        Parameters:
            config_path (str): Path to the configuration file.
            device (torch.device): Device to use for training (CPU or GPU).
        """
        self.config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
        self.device = device
        self.set_seeds()
        self.load_data()
        self.initialize_model()

    def set_seeds(self):
        """
        Set random seeds for reproducibility.
        """
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    def load_data(self):
        """
        Load training and validation data using the Loader class.
        """
        train_midi_path = self.config['data']['prepared']['train_data']
        validation_midi_path = self.config['data']['prepared']['val_data']
        batch_size = self.config['training']['batch_size']

        self.data_loader_train = Loader(tokenized_data_path=train_midi_path, batch_size=batch_size)
        self.train_ldr = self.data_loader_train.create_dataloader(shuffle=True, drop_last=True)

        self.data_loader_val = Loader(tokenized_data_path=validation_midi_path, batch_size=batch_size)
        self.val_ldr = self.data_loader_val.create_dataloader(shuffle=True, drop_last=True)

        self.vocab_size = self.data_loader_train.vocab_size
        self.pad_idx = self.data_loader_train.pad_idx

    def initialize_model(self):
        """
        Initialize the KeyEmotions model with the specified configuration from YAML file.
        """
        self.model = KeyEmotions(
            vocab_size=self.vocab_size,
            d_model=self.config['model']['d_model'],
            nhead=self.config['model']['n_head'],
            num_layers=self.config['model']['num_layers'],
            d_ff=self.config['model']['d_ff'],
            dropout=self.config['model']['dropout'],
        ).to(self.device)

    def calculate_topk_accuracy(self, output, tgt, topk=(1, 5)):
        """
        Calculate top-k accuracy for the model predictions.

        Parameters:
            output (torch.Tensor): Model output predictions.
            tgt (torch.Tensor): Target labels.
            topk (tuple): Tuple of k values for top-k accuracy.

        Returns:
            list: List of top-k accuracies.
        """
        batch_size, seq_len, vocab_size = output.size()
        maxk = max(topk)
        if maxk > vocab_size:
            raise ValueError("topk is larger than vocab_size")
        output = output.view(-1, vocab_size)
        tgt = tgt.view(-1)
        _, pred = output.topk(maxk, dim=1)
        correct = pred.eq(tgt.unsqueeze(1))
        topk_accuracy = []
        for k in topk:
            correct_k = correct[:, :k].sum().item()
            accuracy_k = correct_k / (batch_size * seq_len)
            topk_accuracy.append(accuracy_k)
        return topk_accuracy
    
    def train_iter(self, model, train_ldr, config, total_batches, epoch, criterion, optimizer, logs_train, pad_idx, log_interval=10):
        """
        Train the model for one epoch.
        
        Parameters:
            model (nn.Module): The model to train.
            train_ldr (DataLoader): DataLoader for training data.
            config (dict): Configuration dictionary.
            total_batches (int): Total number of batches in the training set.
            epoch (int): Current epoch number.
            criterion (nn.Module): Loss function.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            logs_train (LogsWriter): Logs writer for training logs.
            pad_idx (int): Padding index for the target sequences.
            log_interval (int): Interval for logging training progress.
        
        Returns:
            Average loss, top-1 accuracy, and top-5 accuracy for the epoch.
        """
        epoch_loss = 0
        epoch_top1_accuracy = 0
        epoch_top5_accuracy = 0
        max_grad_norm = 1.0
        gradient_accumulation_steps = config['training']['gradient_accumulation_steps']
        print("-" * 89)

        optimizer.zero_grad()

        for batch_idx, (src, tgt) in enumerate(train_ldr):
            batch_time = time.time()
            src = src.to(self.device)
            tgt = tgt.to(self.device)

            tgt_mask = model.subsequent_mask(tgt.size(1)).to(self.device)
            tgt_pad_mask = model.create_pad_mask(tgt, pad_idx=pad_idx).to(self.device)
            output = model(src, tgt_mask, tgt_pad_mask)

            loss = criterion(output.view(-1, self.vocab_size), tgt.view(-1))
            loss = loss / gradient_accumulation_steps
            loss.backward()

            top1_acc, top5_acc = self.calculate_topk_accuracy(output, tgt, topk=(1, 5))
            epoch_loss += loss.item() * gradient_accumulation_steps
            epoch_top1_accuracy += top1_acc
            epoch_top5_accuracy += top5_acc

            # Manual calculation of learning rate with warmup
            current_step = epoch * len(train_ldr) + batch_idx
            warmup_steps = int(config['training']['warmup_proportion'] * len(train_ldr) * config['training']['max_epochs'])
            
            if current_step < warmup_steps:
                current_lr = config['training']['lr'] * (current_step / warmup_steps)
            else: 
                current_lr = config['training']['lr']

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            if batch_idx % log_interval == 0:
                elapsed = time.time() - batch_time
                print(f"| epoch {epoch+1:3d} | {batch_idx:4d}/{total_batches:4d} batches | lr {current_lr:02.5f} | ms/batch {elapsed * 1000/log_interval:5.2f} | loss {loss.item() * gradient_accumulation_steps:5.4f} | ppl {np.exp(loss.item() * gradient_accumulation_steps):8.2f} | top1_acc {top1_acc:5.4f} | top5_acc {top5_acc:5.4f} |")
                logs_train.update({"epoch": epoch+1, "batches": batch_idx, "lr": current_lr, "ms/batch": elapsed * 1000/log_interval, "loss": loss.item() * gradient_accumulation_steps, "ppl": np.exp(loss.item() * gradient_accumulation_steps), "top1_acc": top1_acc, "top5_acc": top5_acc})
                batch_time = time.time()

        if (batch_idx + 1) % gradient_accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        print("-" * 89)
        return epoch_loss / total_batches, epoch_top1_accuracy / total_batches, epoch_top5_accuracy / total_batches
    

    def validate(self, model, val_ldr, epoch, criterion, logs_val, pad_idx):
        """
        Validate the model on the validation set.

        Parameters:
            model (nn.Module): The model to validate.
            val_ldr (DataLoader): DataLoader for validation data.
            epoch (int): Current epoch number.
            criterion (nn.Module): Loss function.
            logs_val (LogsWriter): Logs writer for validation logs.
            pad_idx (int): Padding index for the target sequences.

        Returns:
            Average loss, top-1 accuracy, and top-5 accuracy for the validation set.
        """
        model.eval()
        epoch_loss = 0
        epoch_top1_acc = 0
        epoch_top5_acc = 0
        epoch_start_time = time.time()
        with torch.no_grad():
            for _, (src, tgt) in enumerate(val_ldr):
                src = src.to(self.device)
                tgt = tgt.to(self.device)

                tgt_mask = model.subsequent_mask(tgt.size(1)).to(self.device)
                tgt_pad_mask = model.create_pad_mask(tgt, pad_idx=pad_idx).to(self.device)
                output = model(src, tgt_mask, tgt_pad_mask)

                loss = criterion(output.view(-1, self.vocab_size), tgt.view(-1))
                epoch_loss += loss.item()
                
                top1_acc, top5_acc = self.calculate_topk_accuracy(output, tgt, topk=(1, 5))
                epoch_top1_acc += top1_acc
                epoch_top5_acc += top5_acc
                
        epoch_time = time.time() - epoch_start_time
        print(f"| end of epoch {epoch+1:3d} | time: {epoch_time:5.2f}s | loss {epoch_loss:5.4f} | top1_acc {epoch_top1_acc / len(val_ldr):5.4f} | top5_acc {epoch_top5_acc / len(val_ldr):5.4f} |")
        print("-" * 89)
        logs_val.update({"epoch": epoch+1, "time": epoch_time, "loss": epoch_loss, "top1_acc": epoch_top1_acc / len(val_ldr), "top5_acc": epoch_top5_acc / len(val_ldr)})
        return epoch_loss / len(val_ldr), epoch_top1_acc / len(val_ldr), epoch_top5_acc / len(val_ldr)
    
    def train(self):
        """
        Train the KeyEmotions model.
        """
        exp_dir, logs_folder, exp_name = create_exp_environment(
            self.config['training']['experiments_dir'])
        
        logs_train_writer = LogsWriter(os.path.join(logs_folder, "logs_train.csv"), 
                                       ["epoch", "batches", "lr", "ms/batch", "loss", "ppl", "top1_acc", "top5_acc"])
        logs_val_writer = LogsWriter(os.path.join(logs_folder, "logs_val.csv"), 
                                     ["epoch", "time", "loss", "top1_acc", "top5_acc"])
        
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['lr'],
            betas=(0.9, 0.98),
            weight_decay=self.config['training']['weight_decay']
        )
        criterion = nn.CrossEntropyLoss(
            ignore_index=self.pad_idx, 
            label_smoothing=0.05
        )

        # Warmup scheduler config
        total_steps = len(self.train_ldr) * self.config['training']['max_epochs']
        warmup_steps = int(total_steps * self.config['training']['warmup_proportion'])

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 1.0

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        best_val_loss = float('inf')
        epochs_no_improve = 0
        total_batches = len(self.train_ldr)
        train_loss, validation_loss = [], []
        train_top1_acc, train_top5_acc = [], []
        val_top1_acc, val_top5_acc = [], []

        for epoch in range(self.config['training']['max_epochs']):
            try:
                epoch_loss, epoch_top1_acc, epoch_top5_acc = self.train_iter(
                    self.model, self.train_ldr, self.config, total_batches, 
                    epoch, criterion, optimizer, logs_train_writer, self.pad_idx)
                
                train_loss.append(epoch_loss)
                train_top1_acc.append(epoch_top1_acc)
                train_top5_acc.append(epoch_top5_acc)

                val_loss, val_top1_acc_epoch, val_top5_acc_epoch = self.validate(
                    self.model, self.val_ldr, epoch, criterion, logs_val_writer, self.pad_idx)
                
                validation_loss.append(val_loss)
                val_top1_acc.append(val_top1_acc_epoch)
                val_top5_acc.append(val_top5_acc_epoch)

                scheduler.step()

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    save_exp_environment(exp_dir, self.model, self.config)
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve == self.config['training']['patience']:
                        print("Early stopping at epoch {epoch+1}")
                        break

            except Exception as e:
                print(f"Error during epoch {epoch}: {e}")
                print(traceback.format_exc())
                continue

            torch.cuda.empty_cache()

        plot_loss(train_loss, validation_loss, exp_name)
        plot_accuracy(train_top1_acc, val_top1_acc, exp_name, "top1")
        plot_accuracy(train_top5_acc, val_top5_acc, exp_name, "top5")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(config_path="./src/config/default.yaml", device=device)
    trainer.train()
