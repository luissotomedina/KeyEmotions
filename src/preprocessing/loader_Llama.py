from torch.utils.data import Dataset, DataLoader
import pickle
import torch

class Loader(Dataset): # Inherit from PyTorch Dataset class
    def __init__(self, tokenized_data_path, batch_size):
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

        # Load and preprocess dataset
        self.data = self._load_dataset()
        if not self.data:
            raise ValueError("Dataset is empty! Check your tokenized data path.")
        
        self._set_max_len()
        self.padded_data = [self._pad_sequence(song) for song in self.data]
        self._set_vocab_size()
        self._set_number_of_tokens() 

    def _load_dataset(self):
        with open(self.tokenized_data_path, 'rb') as f:
            return pickle.load(f)

    def _set_max_len(self):
        self.max_len = max((len(song) for song in self.data), default=0)  # Handles empty list safely

    def _set_number_of_tokens(self):
        self.number_of_tokens = len(set(token for song in self.data for token in song))
        
    def _set_vocab_size(self):
        self.vocab_size = max(max(song) for song in self.data) + 1

    def _pad_sequence(self, sequence):
        eos_token = sequence[-1]
        remi_token = sequence[:-1]
        
        return remi_token + [self.pad_idx] * (self.max_len - len(sequence)) + [eos_token] # Ensure last token is not padded, EOS token

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence = self.padded_data[idx]
        input_sequence = sequence[:-1]
        output_sequence = sequence[1:]

        attention_mask = [1 if token != self.pad_idx else 0 for token in input_sequence]

        return {
            'input_ids': torch.tensor(input_sequence, dtype=torch.int64),  # Tokens de entrada
            'attention_mask': torch.tensor(attention_mask, dtype=torch.int64),  # Máscara de atención
            'labels': torch.tensor(output_sequence, dtype=torch.int64)  # Tokens de salida (labels)
        }   

    def create_dataloader(self, shuffle=True, drop_last=True):
        return DataLoader(
            dataset=self,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=drop_last
        ) 
    
    def visualize_sample(self, idx=0):
        if idx >= len(self):
            raise IndexError("Index out of range.")
        
        sequence = self.data[idx]
        padded_sequence = self.padded_data[idx]

        print(f"Muestra {idx}")
        print(f"Original: {sequence}")
        print(f"Padding: {padded_sequence}")
    

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    dataset = Loader(
        tokenized_data_path='./data/prepared/train.pkl',
        batch_size=32
    )
    train_ldr = dataset.create_dataloader(
        shuffle=True,
        drop_last=True
    )

    for batch in train_ldr:
        print(batch['input_ids'].shape)
        print(batch['labels'].shape)
        break