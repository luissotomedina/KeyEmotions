import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchviz import make_dot


def sinusoidal_position_encoding(n_position, d_model):
    """
    Compute position encoding for a given position and dimension. 

    Parameters: 
        n_position (int): number of positions.
        d_model (int): dimension of the model.

    Returns:
        torch.tensor: position encoding for the given position and dimension.
    """

    angles = _get_angles(
        np.arange(n_position)[:, np.newaxis], 
        np.arange(d_model)[np.newaxis, :],
        d_model
    )

    # Apply sin to even indices in the array; 2i
    sines = np.sin(angles[:, 0::2])

    # Apply cos to odd indices in the array; 2i+1
    cosines = np.cos(angles[:, 1::2])

    pos_encoding = np.concatenate([sines, cosines], axis=-1)
    pos_encoding = pos_encoding[np.newaxis, ...] # (1, position, d_model)

    return torch.tensor(pos_encoding, dtype=torch.float32)

def _get_angles(pos, i, d_model):
    """
    Compute the angles for a given position.

    Parameters:
        pos (np.ndarray): position.
        i (np.ndarray): dimension.
        d_model (int): dimension of the model.

    Returns:
        np.ndarray: angles for the given position.
    """

    angle_dropout_rates = 1 / np.power(
        10000, (2 * (i // 2)) / np.float32(d_model)
    )
    return pos * angle_dropout_rates

class Transformer(nn.Module):
    """
    Transformer model model architecture.
    """
    def __init__(
        self, 
        n_layers,
        d_model,
        n_heads,
        d_feedforward,
        input_vocab_size,
        target_vocab_size,
        max_num_positions_in_pe_encoder,
        max_num_positions_in_pe_decoder,
        dropout_rate=0.1,
    ):
        """
        Parameters:
            n_layers (int): number of layers in Encoder and Decoder.
            d_model (int): dimension of the model.
            n_heads (int): number of attention heads.
            d_feedforward (int): dimension of the feedforward network.
            input_vocab_size (int): size of the input vocabulary.
            target_vocab_size (int): size of the target vocabulary.
            max_num_positions_in_pe_encoder (int): maximum number of positions for input.
            max_num_positions_in_pe_decoder (int): maximum number of positions for target.
            dropout_rate (float): dropout rate.
        """
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            n_layers,
            d_model, 
            n_heads, 
            d_feedforward, 
            input_vocab_size,
            max_num_positions_in_pe_encoder,
            dropout_rate
        )
        self.decoder = Decoder(
            n_layers,
            d_model, 
            n_heads, 
            d_feedforward,
            target_vocab_size,
            max_num_positions_in_pe_decoder,
            dropout_rate
        )

        self.final_layer = nn.Linear(d_model, target_vocab_size)
    
    def forward(self, input, target, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        """
        Proccess the input through the model.

        Parameters:
            input (torch.Tensor): input tensor to the Encoder.
            target (torch.Tensor): target tensor to the Decoder.
            enc_padding_mask (torch.Tensor): padding mask for the Encoder.
            look_ahead_mask (torch.Tensor): look ahead mask for the Decoder.
            dec_padding_mask (torch.Tensor): padding mask for the Decoder.

        Returns:
            torch.Tensor: output tensor of the Transformer model. 
            dict: attention weights from the Decoder layers.
        """

        enc_output = self.encoder(
            input,
            enc_padding_mask
        ) # (batch_size, input_seq_len, d_model)

        dec_output = self.decoder(
            target,
            enc_output,
            look_ahead_mask,
            dec_padding_mask
        ) # (batch_size, target_seq_len, d_model)

        logits = self.final_layer(
            dec_output
        ) # (batch_size, target_seq_len, target_vocab_size)

        return logits

class Encoder(nn.Module):
    """
    Encoder of the Transformer model, composed by multiple Encoder layers.
    """

    def __init__(
        self,
        n_layers,
        d_model,
        n_heads,
        d_feedforward,
        input_vocab_size,
        max_position_in_pe,
        dropout_rate=0.1
    ):
        """
        Parameters:
            n_layers (int): number of layers in the Encoder.
            d_model (int): dimension of the model.
            n_heads (int): number of attention heads.
            d_feedforward (int): dimension of the feedforward network.
            input_vocab_size (int): size of the input vocabulary.
            max_position_in_pe (int): maximum sequence length that this model might ever be used with.
            dropout_rate (float): dropout rate.
        """
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_vocab_size, d_model)
        self.pos_encoding = sinusoidal_position_encoding(max_position_in_pe, d_model)
        self.enc_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_feedforward, dropout_rate)
            for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x, mask=None):
        """
        Process the input through the Encoder.
        
        Parameters:
            x (torch.Tensor): input tensor to the Encoder.
            mask (torch.Tensor) Mask to be applied on attention weights.
        
        Returns:
            torch.Tensor: output tensor of the Encoder.
        """
        x = self.embedding(x) # (batch_size, input_seq_len, d_model)
        x *= torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))

        sliced_pos_encoding  = self._get_sliced_pos_encoding(x)
        x += sliced_pos_encoding

        x = self.dropout(x)

        for i in range(self.n_layers):
            x = self.enc_layers[i](x, mask)

        return x # (batch_size, input_seq_len, d_model)
    
    def _get_sliced_pos_encoding(self, x):
        """
        Get a slice of the full positional encoding.
        
        Parameters:
            x (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: slice of the positional encoding.
        """
        number_of_tokens = x.size(1)
        return self.pos_encoding[:, :number_of_tokens, :]
    

class Decoder(nn.Module):
    """
    Decoder of the Transformer model, composed by multiple Decoder layers.
    """

    def __init__(
        self, 
        n_layers, 
        d_model, 
        n_heads, 
        d_feedforward,
        target_vocab_size,
        max_position_in_pe,
        dropout_rate=0.1
    ):
        """
        Parameters:
            n_layers (int): number of layers in the Decoder.
            d_model (int): dimension of the model.
            n_heads (int): number of attention heads.
            d_feedforward (int): dimension of the feedforward network.
            target_vocab_size (int): size of the target vocabulary.
            max_position_in_pe (int): maximum sequence length that this model might ever be used with.
            dropout_rate (float): dropout rate.
        """
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        self.embedding = nn.Embedding(target_vocab_size, d_model)
        self.pos_encoding = sinusoidal_position_encoding(max_position_in_pe, d_model)

        self.dec_layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_feedforward, dropout_rate)
            for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, enc_output, look_ahead_mask=None, padding_mask=None):
        """
        Process the input through the Decoder.

        Parameters:
            x (torch.Tensor): input tensor to the Decoder.
            enc_output (torch.Tensor): output tensor from the Encoder.
            look_ahead_mask (torch.Tensor): mask to be applied on attention weights.
            padding_mask (torch.Tensor): mask to be applied on attention weights.

        Returns:
            torch.Tensor: output tensor of the Decoder.
        """
        x = self.embedding(x) # (batch_size, target_seq_len, d_model)
        x *= torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))

        sliced_pos_encoding = self._get_sliced_pos_encoding(x)
        x += sliced_pos_encoding

        x = self.dropout(x)

        for i in range(self.n_layers):
            x = self.dec_layers[i](
                x, enc_output, look_ahead_mask, padding_mask
            )
        
        return x 
    
    def _get_sliced_pos_encoding(self, x):
        """
        Get a slice of the full positional encoding.

        Parameters:
            x (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: slice of the positional encoding.
        """
        num_of_tokens = x.size(1)
        return self.pos_encoding[:, :num_of_tokens, :]
    
class EncoderLayer(nn.Module):
    """
    Decoder layer of a Transformer model, consisting of two Multi-Head Attention layers and a Feed-Forward layer.
    """

    def __init__(
        self, 
        d_model, 
        n_heads, 
        d_feedforward, 
        dropout_rate=0.1
    ):
        """
        Parameters:
            d_model (int): dimension of the model.
            n_heads (int): number of attention heads.
            d_feedforward (int): dimension of the feedforward network.
            dropout_rate (float): dropout rate.
        """
        super(EncoderLayer, self).__init__()

        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout_rate)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_feedforward),
            nn.ReLU(),
            nn.Linear(d_feedforward, d_model)
        )
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
    
    def forward(self, x, mask=None):
        """
        Process the input through the Encoder layer.
        
        Parameters: 
            x (torch.Tensor): input tensor.
            mask (torch.Tensor): mask to be applied on attention weights.
        
        Returns: 
            torch.Tensor: output tensor of the Encoder layer.
        """
        attn_output, _ = self.mha(x, x, x, attn_mask=mask)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2
    
class DecoderLayer(nn.Module):
    """
    Decoder layer of a Transformer model, consisting of two Multi-Head Attention layers and a Feed-Forward layer.
    """

    def __init__(
        self, 
        d_model, 
        n_heads, 
        d_feedforward,
        dropout_rate=0.1
    ):
        """
        Parameters:
            d_model (int): dimension of the model.
            n_heads (int): number of attention heads.
            d_feedforward (int): dimension of the feedforward network.
            dropout_rate (float): dropout rate.
        """
        super(DecoderLayer, self).__init__()
        self.mha1 = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout_rate)
        self.mha2 = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout_rate)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_feedforward),
            nn.ReLU(),
            nn.Linear(d_feedforward, d_model)
        )
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)
    
    def forward(self, x, enc_output, look_ahead_mask=None, padding_mask=None):
        """
        Process the input through the Decoder layer.

        Parameters:
            x (torch.Tensor): input tensor.
            enc_output (torch.Tensor): output tensor from the Encoder.
            look_ahead_mask (torch.Tensor): mask to be applied on attention weights.
            padding_mask (torch.Tensor): mask to be applied on attention weights.

        Returns:
            torch.Tensor: output tensor of the Decoder layer.
        """
        attn1, _ = self.mha1(x, x, x, attn_mask=look_ahead_mask)
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(x + attn1)

        attn2, _ = self.mha2(out1, enc_output, enc_output, attn_mask=padding_mask)
        attn2 = self.dropout2(attn2)
        out2 = self.layernorm2(out1 + attn2)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output)
        out3 = self.layernorm3(out2 + ffn_output)

        return out3
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define the Transformer parameters
    n_layers = 6
    d_model = 64
    n_heads = 2
    d_feedforward = 128
    input_vocab_size = 100
    target_vocab_size = 100
    dropout_dropout_rate = 0.1
    pe_input = 10 # max number of positions in PE for input
    pe_target = 10 # max number of positions in PE for target

    transformer_model = Transformer(
        n_layers, 
        d_model, 
        n_heads, 
        d_feedforward,
        input_vocab_size,
        target_vocab_size,
        pe_input,
        pe_target,
        dropout_dropout_rate
    )

    dummy_input = torch.randint(0, input_vocab_size, (1, 10))
    dummy_target = torch.randint(0, target_vocab_size, (1, 10))

    output = transformer_model(
        dummy_input, 
        dummy_target, 
        enc_padding_mask = None, 
        look_ahead_mask = None, 
        dec_padding_mask = None
    )

    # make_dot(output, params=dict(transformer_model.named_parameters())).render("Transformer", format="png")
    
    print(transformer_model)
    print(f"Input shape: {dummy_input}")
    print(f"Output shape: {output.shape}")