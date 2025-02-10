import pickle
import os
import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset

class Loader:
    def __init__(self, tokenized_data_path, batch_size):
        self.tokenized_data_path = tokenized_data_path
        self.batch_size = batch_size
        self.max_len = 0
        self.number_of_tokens = 0
        self.vocab_size = 0

    # def number_of_tokens_with_padding(self):
    #     return self.number_of_tokens + 1  # Ensure padding is accounted for

    def create_training_dataset(self):
        dataset = self._load_dataset()
        if not dataset:
            raise ValueError("Dataset is empty! Check your tokenized data path.")
        
        self._set_max_len(dataset)
        padded_dataset = [self._pad_sequence(song) for song in dataset]
        self._set_vocab_size(padded_dataset)
        self._set_number_of_tokens(padded_dataset)  # Includes padding token

        return self._create_dataloader(padded_dataset)

    def _load_dataset(self):
        with open(self.tokenized_data_path, 'rb') as f:
            return pickle.load(f)

    def _set_max_len(self, songs):
        self.max_len = max((len(song) for song in songs), default=0)  # Handles empty list safely

    def _set_number_of_tokens(self, songs):
        self.number_of_tokens = len(set(token for song in songs for token in song))
        
    def _set_vocab_size(self, songs):
        self.vocab_size = max(max(song) for song in songs) + 1

    def _pad_sequence(self, sequence, pad_token=181):
        eos_token = sequence[-1]
        remi_token = sequence[:-1]
        
        return remi_token + [pad_token] * (self.max_len - len(sequence)) + [eos_token] # Ensure last token is not padded, EOS token

    def _create_dataloader(self, input_sequences, shuffle=True, drop_last=True):
        input_tensors = torch.tensor(input_sequences, dtype=torch.int64)
        dataset = TensorDataset(input_tensors)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, drop_last=drop_last)
        return dataloader, input_tensors
    

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    data_loader = Loader(
        tokenized_data_path='./data/prepared/train.pkl',
        batch_size=32
    )
    input_ldr, input_tensors = data_loader.create_training_dataset()

    input_tensors = input_tensors.to(device)

    print("Vocab size:", data_loader.vocab_size)
    print("Number of tokens:", data_loader.number_of_tokens)
    print("Max sequence length:", data_loader.max_len)

    for batch_idx, batch in enumerate(input_ldr):
        batch_data = batch[0].to(device)  # Move batch to GPU if available
        print("Batch shape:", batch_data.shape)  # Expected: (batch_size, max_len)
        print("First sample in batch:", batch_data[0])  # Print one sample
        break  # Stop after first batch