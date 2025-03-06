import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
import torch
import sys

from huggingface_hub import login

with open("huggingface_token.txt", "r") as f:
    token = f.read()

login(token=token)

model_name = "meta-llama/Llama-3.2-1B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

# LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]
)

model = get_peft_model(model, lora_config)

# ENTRENAMIENTO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.loader_Llama import Loader

train_dataset = Loader(
        tokenized_data_path='./data/prepared/train.pkl',
        batch_size=32
    )
valid_dataset = Loader(
        tokenized_data_path='./data/prepared/valid.pkl',
        batch_size=32
    )

training_args = TrainingArguments(
    output_dir="./experiments/Llama",
    per_device_train_batch_size=2,  # Reduced batch size
    gradient_accumulation_steps=4,  # Gradient accumulation
    num_train_epochs=1,
    logging_dir="./experiments/Llama/logs",
    logging_steps=10,
    save_steps=100,
    evaluation_strategy="steps",
    eval_steps=50,
    save_total_limit=2,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer
)

trainer.train()

model.save_pretrained("./experiments/Llama/fine-tuned")
tokenizer.save_pretrained("./experiments/Llama/fine-tuned")