from LoRA import * 

from torchtune.models.llama2 import llama2_7b, lora_llama2_7b

# Build Llama2 without any LoRA layers
base_model = llama2_7b()

lora_model = lora_llama2_7b(lora_attn_modules=["q_proj", "v_proj"])

print(base_model.layers[0].attn)
print(lora_model.layers[0].attn)
