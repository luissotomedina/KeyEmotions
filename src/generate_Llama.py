from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = './experiments/Llama/fine-tuned'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

input_sequence = [0, 1, 178, 163, 5, 6, 60, 132]
input_ids = torch.tensor([input_sequence], dtype=torch.long).to(device)

attention_mask = torch.ones_like(input_ids)

output = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,  # Pasar la máscara de atención
    max_length=50,
    num_return_sequences=1,
    pad_token_id=tokenizer.pad_token_id  # Usar el token de padding correcto
)

# Decodificar la secuencia generada
generated_sequence = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_sequence)
