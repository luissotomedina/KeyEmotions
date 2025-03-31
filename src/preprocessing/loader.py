"""
loader.py

This module defines a PyTorch Dataset class for loading and preprocessing tokenized data.
"""

import torch
import pickle

from torch.utils.data import DataLoader, Dataset

class Loader(Dataset): # Inherit from PyTorch Dataset class
    def __init__(self, tokenized_data_path, batch_size):
        """
        Initialize the Loader class.
        
        Parameters:
            tokenized_data_path (str): Path to the tokenized data file.
            batch_size (int): Batch size for DataLoader.
        """
        self.tokenized_data_path = tokenized_data_path
        self.batch_size = batch_size
        self.max_len = 0
        self.number_of_tokens = 0
        self.vocab_size = 0

        # Token indices
        self.sos_idx = 0
        self.emotion_idx = 1
        self.bar_idx = 5
        self.pos_idx = 6
        self.pitch_idx = 38
        self.dur_idx = 128
        self.sign_idx = 160
        self.tpb_idx = 170
        self.pad_idx = 179
        self.eos_idx = 180

        self.data = self._load_dataset()
        if not self.data:
            raise ValueError("Dataset is empty! Check your tokenized data path.")
        
        self._set_max_len()
        self.padded_data = [self._pad_sequence(song) for song in self.data]
        self._set_vocab_size()
        self._set_number_of_tokens() 

    def _load_dataset(self):
        """
        Load the dataset from the specified path.

        Returns:
            list: Loaded dataset.
        """
        with open(self.tokenized_data_path, 'rb') as f:
            return pickle.load(f)

    def _set_max_len(self):
        """
        Set the maximum length of sequences in the dataset.
        """
        self.max_len = max((len(song) for song in self.data), default=0)  # Handles empty list safely

    def _set_number_of_tokens(self):
        """
        Set the number of unique tokens in the dataset.
        """
        self.number_of_tokens = len(set(token for song in self.data for token in song))
        
    def _set_vocab_size(self):
        """
        Set the vocabulary size based on the maximum token index in the dataset.
        """
        self.vocab_size = max(max(song) for song in self.data) + 1

    def _pad_sequence(self, sequence):
        """
        Pad a sequence to the maximum length.

        Returns:
            list: Padded sequence.
        """
        eos_token = sequence[-1]
        remi_token = sequence[:-1]
        
        return remi_token + [self.pad_idx] * (self.max_len - len(sequence)) + [eos_token] # Ensure last token is not padded, EOS token

    def __len__(self):
        """
        Return the number of sequences in the dataset.
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a sequence and its corresponding target sequence.
        """
        sequence = self.padded_data[idx]
        input_sequence = sequence[:-1]
        output_sequence = sequence[1:]
        return torch.tensor(input_sequence, dtype=torch.int64), \
                torch.tensor(output_sequence, dtype=torch.int64)

    def create_dataloader(self, shuffle=True, drop_last=True):
        """
        Create a DataLoader for the dataset.

        Returns:
            DataLoader: DataLoader for the dataset.
        """
        return DataLoader(
            dataset=self,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=drop_last
        ) 

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    data_loader = Loader(
        tokenized_data_path='./data/prepared/train.pkl',
        batch_size=32
    )
    train_ldr = data_loader.create_dataloader(
        shuffle=True,
        drop_last=True
    )

    for input_seq, output_seq in train_ldr:
        print("Input shape:", input_seq.shape)
        print("Input sequence:", input_seq[0])
        print("Output shape:", output_seq.shape)
        print("Output sequence:", output_seq[0])    
        break