import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import math

class PositionalEncoding(nn.Module):
    """
    Positional encoding for the transformer model.
    """
    def __init__(self, d_model, max_len=20480):
        """
        Initialize the PositionalEncoding class.

        Parameters:
            d_model (int): dimension of the model.
            max_len (int): maximum length of the input.
        """
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * - (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) # Apply sin to even indices in the array; 2i
        pe[:, 1::2] = torch.cos(position * div_term) # Apply cos to odd indices in the array; 2i+1
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, seq_len, batch_size):
        """
        Add positional encoding to the input.

        Parameters:
            seq_len (int): length of the input sequence.
            batch_size (int): batch size of the input.

        Returns:
            pos_encoding (torch.Tensor): positional encoding of the input.
        """
        pos_encoding = self.pe[:seq_len, :].repeat(1, batch_size, 1)
        return pos_encoding

class TransformerVAE(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, num_layers, latent_dim, emotion_dim, max_seq_len):
        super(TransformerVAE, self).__init__()
        self.latent_dim = latent_dim
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_head, batch_first=True), num_layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, n_head, batch_first=True), num_layers
        )
        self.fc_mu = nn.Linear(d_model, latent_dim)
        self.fc_logvar = nn.Linear(d_model, latent_dim)
        self.emotion_fc = nn.Linear(emotion_dim, latent_dim)
        self.latent_to_d_model = nn.Linear(latent_dim, d_model)
        self.output_layer = nn.Linear(d_model, vocab_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x, emotion_vec):
        """
        Forward pass of the model.
        
        Parameters: 
            x (torch.Tensor): input sequence.
            emotion_vec (torch.Tensor): emotion vector.
        """
        # Embedding and positional encoding
        x = self.embedding(x)  # (batch_size, seq_len, d_model)
        pos_enc = self.pos_encoding(x.size(1), x.size(0))  # (seq_len, batch_size, d_model)
        x = x + pos_enc.to(x.device)
        # x = x + pos_enc.permute(1, 0, 2).to(x.device)  # Add positional encoding
        # No need to permute x as batch_first=True is used in Transformer layers

        # Encoder: Get z_content
        enc_output = self.encoder(x)  # (batch_size, seq_len, d_model)
        enc_output = enc_output.mean(dim=1)  # Aggregate along sequence dimension

        mu = self.fc_mu(enc_output)  # (batch_size, latent_dim)
        logvar = self.fc_logvar(enc_output)  # (batch_size, latent_dim)
        z_content = self.reparameterize(mu, logvar)  # (batch_size, latent_dim)

        # Emotion conditioning: Get z_emotion
        z_emotion = self.emotion_fc(emotion_vec)  # (batch_size, latent_dim)

        # Combine z_content and z_emotion
        z_combined = z_content + z_emotion  # (batch_size, latent_dim)

        # Transform latent space to match d_model dimension
        z_combined = self.latent_to_d_model(z_combined)  # (batch_size, d_model)

        # Decoder
        z_combined = z_combined.unsqueeze(1).repeat(1, x.size(1), 1)  # (batch_size, seq_len, d_model)
        dec_output = self.decoder(z_combined, x)  # (batch_size, seq_len, d_model)

        # Output
        logits = self.output_layer(dec_output)  # (batch_size, seq_len, vocab_size)
        return logits, mu, logvar

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Example usage
    vocab_size = 512  # MIDI vocabulary size
    d_model = 128     # Transformer embedding size
    nhead = 8         # Number of attention heads
    num_layers = 6    # Number of transformer layers
    latent_dim = 64   # Latent space dimensionality
    emotion_dim = 4   # Dimensionality of emotion vector
    max_seq_len = 512 # Maximum MIDI sequence length

    # Datos ficticios
    input_seqs = torch.randint(0, vocab_size, (1000, max_seq_len)) # (num_samples, seq_len)
    emotion_vecs = torch.rand(1000, emotion_dim) # (num_samples, emotion_dim)
    targets = torch.randint(0, vocab_size, (1000, max_seq_len)) # (num_samples, seq_len)

    dataset = TensorDataset(input_seqs, emotion_vecs, targets)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Inicializar el modelo, optimizador y función de pérdida
    model = TransformerVAE(vocab_size, d_model, nhead, num_layers, latent_dim, emotion_dim, max_seq_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Entrenamiento
    num_epochs = 10
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            input_seq, emotion_vec, target = batch
            input_seq, emotion_vec, target = input_seq.to(device), emotion_vec.to(device), target.to(device)

            optimizer.zero_grad()
            logits, mu, logvar = model(input_seq, emotion_vec)

            # Aplanar logits y target para la función de pérdida
            logits = logits.view(-1, vocab_size)
            targets = target.view(-1)

            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")

    print("Training finished!")
