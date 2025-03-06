import torch 
import torch.nn as nn

import os
import sys
import numpy as np

def init_weights(model):
    # print(f"Initializing weights for {model.__class__.__name__}")
    if isinstance(model, (nn.Linear, nn.Embedding)):
        nn.init.xavier_uniform_(model.weight)

    if isinstance(model, nn.Linear) and model.bias is not None:
        nn.init.zeros_(model.bias)

def subsequent_mask(size):
    return torch.triu(torch.ones(size, size), diagonal=1).bool()

class TransformerNet(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dropout=0.1):
        super(TransformerNet, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        self.encoder = Encoder(d_model, nhead, num_layers, dropout)
        self.decoder = Decoder(d_model, nhead, num_layers, dropout)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.apply(init_weights)
        
    def forward(self, src, tgt, tgt_mask, src_pad_mask=None, tgt_pad_mask=None):
        src = self.embedding(src)
        src = self.pos_enc(src)
        src = self.norm(src)
        src = self.dropout(src)
        memory = self.encoder(src, src_pad_mask=src_pad_mask)
        
        tgt = self.embedding(tgt)
        tgt = self.pos_enc(tgt)
        tgt = self.norm(tgt)
        tgt = self.dropout(tgt)
        
        z = self.decoder(tgt, memory, tgt_mask=tgt_mask, tgt_pad_mask=tgt_pad_mask, memory_pad_mask=src_pad_mask)
        return self.fc_out(z)

class Encoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dropout):
        super(Encoder, self).__init__()
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=num_layers
        )

    def forward(self, src, src_pad_mask=None):
        return self.encoder(src, src_key_padding_mask=src_pad_mask)

class Decoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dropout):
        super(Decoder, self).__init__()
        decoder_layers = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layers, num_layers=num_layers)

    def forward(self, tgt, memory, tgt_mask=None, tgt_pad_mask=None, memory_pad_mask=None):
        return self.decoder(tgt, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_pad_mask, memory_key_padding_mask=memory_pad_mask)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float=0.1, max_len: int=1400):
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
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
    
if __name__ == "__main__":
    dummy_input = torch.randint(0, 12, (2, 10))
    dummy_tgt = torch.randint(0, 12, (2, 9))
    dummy_tgt_mask = subsequent_mask(dummy_tgt.size(1))
    dummy_src_pad_mask = (dummy_input == 0)
    dummy_tgt_pad_mask = (dummy_tgt == 0)

    model = TransformerNet(12, 4, 2, 2)
    out = model(dummy_input, dummy_tgt, dummy_tgt_mask, dummy_src_pad_mask, dummy_tgt_pad_mask)

    print(out.shape)




