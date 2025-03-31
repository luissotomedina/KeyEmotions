"""
KeyEmotions.py

Class for KeyEmotions model using Autorregresive Transformer based on Decoder
"""
import torch 
import torch.nn as nn

import numpy as np  

class KeyEmotions(nn.Module):
    """
    KeyEmotions model using an autoregressive Transformer decoder.

    The model is designed to process sequences of emotions and generate
    corresponding music features.
    """
    def __init__(self, vocab_size, d_model, nhead, num_layers, d_ff, dropout=0.1):
        """
        Initialize the KeyEmotions model.
        
        Parameters:
            vocab_size (int): Size of the vocabulary.
            d_model (int): Dimension of the model.
            nhead (int): Number of attention heads.
            num_layers (int): Number of decoder layers.
            d_ff (int): Dimension of the feedforward network.
            dropout (float): Dropout rate.
        """
        super(KeyEmotions, self).__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.decoder = Decoder(d_model, nhead, num_layers, d_ff, dropout)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def subsequent_mask(self, size):
        """
        Create a mask for the subsequent positions in the sequence.
        
        Parameters:
            size (int): Size of the sequence.

        Returns:
            torch.Tensor: A mask tensor of shape (size, size).
        """
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        return mask.bool()
    
    def create_pad_mask(self, tgt, pad_idx):
        """
        Create a padding mask for the target sequence.

        Parameters:
            tgt (torch.Tensor): Target sequence tensor.
            pad_idx (int): Padding index.
        """
        return (tgt == pad_idx)
    
    def forward(self, tgt, tgt_mask, tgt_pad_mask):
        """
        Forward pass of the KeyEmotions model.
        
        Parameters:
            tgt (torch.Tensor): Target sequence tensor.
            tgt_mask (torch.Tensor): Target mask tensor.
            tgt_pad_mask (torch.Tensor): Padding mask tensor.
        
        Returns:
            torch.Tensor: Output tensor after passing through the model.
        """
        tgt = self.embedding(tgt)
        tgt = self.pos_enc(tgt)
        tgt = self.norm(tgt)
        z = self.decoder(tgt, tgt_mask=tgt_mask, tgt_pad_mask=tgt_pad_mask)
        return self.fc_out(z)

class Decoder(nn.Module):
    """
    Transformer Decoder for the KeyEmotions model.
    """
    def __init__(self, d_model, nhead, num_layers, d_ff, dropout):
        """
        Initialize the Transformer Decoder.
        
        Parameters: 
            d_model (int): Dimension of the model.
            nhead (int): Number of attention heads.
            num_layers (int): Number of decoder layers.
            d_ff (int): Dimension of the feedforward network.
            dropout (float): Dropout rate.
        """
        super(Decoder, self).__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_ff,
            dropout=dropout, 
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, tgt, tgt_mask, tgt_pad_mask):
        """
        Forward pass of the Transformer Decoder.
        
        Parameters:
            tgt (torch.Tensor): Target sequence tensor.
            tgt_mask (torch.Tensor): Target mask tensor.
            tgt_pad_mask (torch.Tensor): Padding mask tensor.
        """
        return self.decoder(tgt, memory=tgt, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_pad_mask)

class PositionalEncoding(nn.Module):
    """
    Positional Encoding for the KeyEmotions model.
    """
    def __init__(self, d_model, dropout=0.1, max_len=1400):
        """
        Initialize the Positional Encoding.
        
        Parameters:
            d_model (int): Dimension of the model.
            dropout (float): Dropout rate.
            max_len (int): Maximum length of the sequence.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10_000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Forward pass of the Positional Encoding.
        
        Parameters:
            x (torch.Tensor): Input tensor.
        """
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return self.dropout(x)
    
if __name__ == "__main__":
    vocab_size = 180
    d_model = 512
    nhead = 8
    num_layers = 6
    dropout = 0.1
    seq_len = 10
    batch_size = 2

    model = KeyEmotions(
        vocab_size=vocab_size, 
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout
    )

    tgt = torch.randint(0, vocab_size, (batch_size, seq_len))

    tgt_mask = model.subsequent_mask(seq_len)
    print(tgt_mask)
    tgt_pad_mask = model.create_pad_mask(tgt, pad_idx=179)
    

    output = model(tgt, tgt_mask=tgt_mask, tgt_pad_mask=tgt_pad_mask)

    print("Output shape: ",output.shape)  # torch.Size([2, 10, 180])
    print("Output type: ", type(output))  # <class 'torch.Tensor'>

    
