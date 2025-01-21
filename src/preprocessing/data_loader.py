import os
from pathlib import Path
from random import shuffle

from miditok import REMI, TokenizerConfig
from miditok.pytorch_data import DatasetMIDI, DataCollator
from miditok.utils import split_files_for_training
from miditok.data_augmentation import augment_dataset

from torch.utils.data import DataLoader

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
config = TokenizerConfig(**TOKENIZER_PARAMS)

tokenizer = REMI(config)

midi_paths = list(Path(os.getcwd(), "data", "raw").rglob("**/*.mid"))
# midi_paths = list(Path("data", "raw").rglob("**/*.mid"))
tokenizer.train(
    vocab_size=30000,
    files_paths=midi_paths,
)
tokenizer.save("tokenizer.json")

total_num_files = len(midi_paths)
num_files_valid = int(total_num_files * 0.2)
shuffle(midi_paths)
midi_paths_valid = midi_paths[:num_files_valid]
midi_paths_train = midi_paths[num_files_valid:]

for files_paths, split in ((midi_paths_train, "train"), (midi_paths_valid, "valid")):
    print(f"Processing {split} files")
    subset_chunks_dir = Path(os.getcwd(), "data", "processed", f"chunks_{split}")
    os.makedirs(subset_chunks_dir, exist_ok=True)
    split_files_for_training(
        files_paths=files_paths,
        tokenizer=tokenizer,
        save_dir=subset_chunks_dir,
        max_seq_len=1024,
        num_overlap_bars=2
    )

    # Data augmentation
    # augment_dataset(
    #     subset_chunks_dir,
    #     pitch_offsets=[-12, 12],
    #     velocity_offsets=[-4, 4],
    #     duration_offsets=[-0.5, 0.5]
    # )

midi_paths_train = list(Path(os.getcwd(), "data", "processed", "chunks_train").rglob("**/*.mid"))
midi_paths_valid = list(Path(os.getcwd(), "data", "processed", "chunks_valid").rglob("**/*.mid"))
kwargs_dataset = {
    "max_seq_len": 1024, "tokenizer": tokenizer, "bos_token_id": tokenizer["BOS_None"], "eos_token_id": tokenizer["EOS_None"]
}
dataset_train = DatasetMIDI(midi_paths_train, **kwargs_dataset)
dataset_valid = DatasetMIDI(midi_paths_valid, **kwargs_dataset)

collator = DataCollator(tokenizer.pad_token_id)

train_loader = DataLoader(dataset=dataset_train, collate_fn=collator)
valid_loader = DataLoader(dataset=dataset_valid, collate_fn=collator)   

vocab_size = len(tokenizer)
print(f"Vocab size: {vocab_size}")

print(f"Train dataset: {len(dataset_train)}")
print(f"Lenght of train loader: {len(train_loader)}")

i = 0
for batch in train_loader:
    i += 1
    print(f"Batch {i}")
    print(f"training model on batch of size {len(batch)}")
    # print(batch)