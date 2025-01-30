import os
import yaml
import pickle
from pathlib import Path
from random import shuffle

from miditok import REMI, TokenizerConfig
from miditok.pytorch_data import DatasetMIDI, DataCollator
from miditok.utils import split_files_for_training
from miditok.data_augmentation import augment_dataset

def create_tokenizer(config_params, data_dir, output_path, vocab_size):
    """
    Create and return a REMI tokenizer with the given configuration parameters.
    
    Parameters:
        config_params (dict): Configuration parameters for the tokenizer.
        vocab_size (int): Size of the vocabulary.
        raw_data_dir (str): Path to the directory containing the MIDI files.
    
    Returns:
        tokenizer: REMI, configured tokenizer.
        midi_paths: list, list of paths to the MIDI files.
    """
    config = TokenizerConfig(**config_params)
    tokenizer = REMI(config)
    midi_paths = list(Path(os.getcwd(), data_dir).rglob("**/*.mid"))
    tokenizer.train(vocab_size=vocab_size, files_paths=midi_paths)
    tokenizer.save(os.path.join(output_path, "tokenizer_params.json"))
    # tokenizer.save_params(os.path.join(output_path, "tokenizer_params.json"))
    return tokenizer, midi_paths

def split_data(midi_paths, validation_split=0.1):
    """
    Split the MIDI files into training and validation sets.
    
    Parameters:
        midi_paths (list): List of paths to the MIDI files.
        validation_split (float): Fraction of the data to be used for validation.
    
    Returns:
        midi_paths_train: list, paths to the training MIDI files.
        midi_paths_valid: list, paths to the validation MIDI files.
    """
    total_num_files = len(midi_paths)
    num_files_valid = int(total_num_files * validation_split)
    shuffle(midi_paths)
    midi_paths_valid = midi_paths[:num_files_valid]
    midi_paths_train = midi_paths[num_files_valid:]
    return midi_paths_train, midi_paths_valid

def process_data(files_paths, tokenizer, output_path, split, max_seq_len, num_overlap_bars=2):
    """
    Process the MIDI files into chunks for training or validation.
    
    Parameters: 
        files_path (list): List of paths to the MIDI files.
        tokenizer: REMI, tokenizer to use for processing.
        output_path (str): Path to the directory to save the processed chunks.
        split (str): Split of the data (train or valid).
        max_seq_len (int): Maximum sequence length.
        num_overlap_bars (int): Number of overlapping bars.
    
    Returns:
        list, paths to the processed MIDI files.
    """
    print(f"Processing {split} files")
    save_dir = Path(output_path)
    # subset_chunks_dir = Path(os.path.join(output_path, f"Maestro_{split}"))
    os.makedirs(output_path, exist_ok=True)
    split_files_for_training(
        files_paths=files_paths,
        tokenizer=tokenizer,
        save_dir=save_dir,
        max_seq_len=max_seq_len,
        num_overlap_bars=num_overlap_bars
    )
    return list(save_dir.rglob("**/*.mid"))

def create_dataset(train_paths, valid_paths, tokenizer, max_seq_len):
    """
    Create datasets for training and validation.
    
    Parameters:
        train_paths (list): List of paths to the training MIDI files.
        valid_paths (list): List of paths to the validation MIDI files.
        tokenizer: REMI, tokenizer to use for processing.
        max_seq_len (int): Maximum sequence length.
        
    Returns:
        train_loader: DatasetMIDI, training dataset.
        valid_loader: DatasetMIDI, validation dataset.
    """
    kwargs_dataset = {
        "max_seq_len": max_seq_len, 
        "tokenizer": tokenizer, 
        "bos_token_id": tokenizer["BOS_None"], 
        "eos_token_id": tokenizer["EOS_None"]
    }
    dataset_train = DatasetMIDI(train_paths, **kwargs_dataset)
    dataset_valid = DatasetMIDI(valid_paths, **kwargs_dataset)
    return dataset_train, dataset_valid

def save_dataset(dataset, output_path):
    """Save dataset to a file."""
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)

if __name__ == "__main__":
    BEAT_RES = {(0, 1): 12, (1, 2): 4, (2, 4): 2, (4, 8): 1}
    TOKENIZER_PARAMS = {
        "pitch_range": (21, 109),
        "beat_res": BEAT_RES,
        "num_velocities": 24,
        "special_tokens": ["PAD", "BOS", "EOS"],
        "use_chords": True,
        "use_rests": True,
        "use_tempos": True,
        "use_time_signatures": True,
        "use_programs": False,  # no multitrack here
        "num_tempos": 32,
        "tempo_range": (50, 200),  # (min_tempo, max_tempo)
    }
    config_path = './src/config/default.yaml'
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

    processed_data_path = config['data']['processed']['cleaned']
    processed_path = config['data']['processed_dir']
    prepared_path = config['data']['prepared_dir']
    train_data_path = config['data']['prepared']['train']
    valid_data_path = config['data']['prepared']['val']

    vocab_size = config['data']['vocab_size']
    max_seq_len = config['data']['max_seq_len']


    tokenizer, midi_paths = create_tokenizer(TOKENIZER_PARAMS, processed_data_path, output_path=prepared_path, vocab_size=vocab_size)
    midi_paths_train, midi_paths_valid = split_data(midi_paths)
    train_paths = process_data(midi_paths_train, tokenizer, train_data_path, "train", max_seq_len)
    valid_paths = process_data(midi_paths_valid, tokenizer, valid_data_path, "validation", max_seq_len)
    dataset_train, dataset_valid = create_dataset(train_paths, valid_paths, tokenizer, max_seq_len)

    save_dataset(dataset_train, os.path.join(prepared_path, "train_dataset.pkl"))
    save_dataset(dataset_valid, os.path.join(prepared_path, "valid_dataset.pkl"))


# config = TokenizerConfig(**TOKENIZER_PARAMS)

# tokenizer = REMI(config)

# midi_paths = list(Path(os.getcwd(), "data", "raw").rglob("**/*.mid"))
# tokenizer.train(
#     vocab_size=30000,
#     files_paths=midi_paths,
# )
# tokenizer.save("tokenizer.json")

# total_num_files = len(midi_paths)
# num_files_valid = int(total_num_files * 0.2)
# shuffle(midi_paths)
# midi_paths_valid = midi_paths[:num_files_valid]
# midi_paths_train = midi_paths[num_files_valid:]

# for files_paths, split in ((midi_paths_train, "train"), (midi_paths_valid, "valid")):
#     print(f"Processing {split} files")
#     subset_chunks_dir = Path(os.getcwd(), "data", "processed", f"chunks_{split}")
#     os.makedirs(subset_chunks_dir, exist_ok=True)
#     split_files_for_training(
#         files_paths=files_paths,
#         tokenizer=tokenizer,
#         save_dir=subset_chunks_dir,
#         max_seq_len=1024,
#         num_overlap_bars=2
#     )

#     # Data augmentation
#     # augment_dataset(
#     #     subset_chunks_dir,
#     #     pitch_offsets=[-12, 12],
#     #     velocity_offsets=[-4, 4],
#     #     duration_offsets=[-0.5, 0.5]
#     # )

# midi_paths_train = list(Path(os.getcwd(), "data", "processed", "chunks_train").rglob("**/*.mid"))
# midi_paths_valid = list(Path(os.getcwd(), "data", "processed", "chunks_valid").rglob("**/*.mid"))
# kwargs_dataset = {
#     "max_seq_len": 1024, "tokenizer": tokenizer, "bos_token_id": tokenizer["BOS_None"], "eos_token_id": tokenizer["EOS_None"]
# }
# dataset_train = DatasetMIDI(midi_paths_train, **kwargs_dataset)
# dataset_valid = DatasetMIDI(midi_paths_valid, **kwargs_dataset)

# collator = DataCollator(tokenizer.pad_token_id)

# train_loader = DataLoader(dataset=dataset_train, collate_fn=collator)
# valid_loader = DataLoader(dataset=dataset_valid, collate_fn=collator)   

# vocab_size = len(tokenizer)
# print(f"Vocab size: {vocab_size}")

# print(f"Train dataset: {len(dataset_train)}")
# print(f"Lenght of train loader: {len(train_loader)}")

# i = 0
# for batch in train_loader:
#     i += 1
#     print(f"Batch {i}")
#     print(f"training model on batch of size {len(batch)}")
#     # print(batch)