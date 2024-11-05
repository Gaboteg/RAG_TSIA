from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Nombre del modelo en Hugging Face (asegúrate de que el nombre sea correcto)
model_name = "meta-llama/LLaMA-3.2-1B-Instruct"  # Cambia esto si el nombre es diferente

# Cargar el tokenizador y el modelo
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token="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")
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token="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")

# Ejemplo de entrada
prompt = "Que tipos de tejido muscular conoces?"
inputs = tokenizer(prompt, return_tensors="pt")

# Generar una respuesta
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=200)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Respuesta:", response)
