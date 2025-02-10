import os
import sys
import yaml
import torch 
from huggingface_hub import login 
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from preprocessing.loader import Loader

if __name__=="__main__":
    login(token='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')

    MODEL_NAME = "mistralai/Mistral-7B-v0.1"
    config_path = './src/config/default.yaml'
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

    # DATA PATHS
    prepared_path = config['data']['prepared_dir']
    train_midi_path = config['data']['prepared']['train_data']
    valid_midi_path = config['data']['prepared']['val_data']
    
    # MODEL PARAMETERS
    max_len = config['model']['max_seq_len']
    batch_size = config['training']['batch_size']
    batch_size = 1
    grad_accum = 8
    lr = config['training']['lr']
    max_epochs = config['training']['max_epochs']

    # Load model with QLoRA
    lora_config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias='none', task_type="CAUSAL_LM"
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, load_in_8bit=True, device_map="auto"
    )
    model = get_peft_model(model, lora_config)

    # Load data
    print("\nLoading data int-token train data")
    data_loader = Loader(train_midi_path, batch_size)
    train_ldr, _ = data_loader.create_training_dataset()

    # Train config
    training_args = TrainingArguments(
        output_dir='./checkpoints',
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        num_train_epochs=max_epochs,
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
        push_to_hub=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ldr,
        tokenizer=tokenizer
    )

    # Train model
    print("\nStarting training")
    trainer.train()

    # Save model
    model.save_pretrained('./checkpoints/mistral')
    tokenizer.save_pretrained('./checkpoints/mistral')