import torch

from models.transformer_VAE import *
from preprocessing.data_loader import *

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    midi_path = "data/raw"
    tokenizer = REMITokenizer(midi_path)
    dataset = tokenizer.create_dataset()
    vocab_size = tokenizer.get_vocab_size()

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    d_model = 128     # Transformer embedding size
    nhead = 8         # Number of attention heads
    num_layers = 6    # Number of transformer layers
    latent_dim = 64   # Latent space dimensionality
    emotion_dim = 4   # Dimensionality of emotion vector
    max_seq_len = 512 # Maximum MIDI sequence length

    model = TransformerVAE(vocab_size, d_model, nhead, num_layers, latent_dim, emotion_dim, max_seq_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 10
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            input_seq, emotion_vec, target = batch
            input_seq, emotion_vec, target = input_seq.to(device), emotion_vec.to(device), target.to(device)

            optimizer.zero_grad()
            logits, mu, logvar = model(input_seq, emotion_vec)

            # Flatten logits and target for the loss function
            logits = logits.view(-1, vocab_size)
            target = target.view(-1)

            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")

    print("Training finished!")