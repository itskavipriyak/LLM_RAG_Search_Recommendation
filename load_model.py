import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set your Hugging Face token as an environment variable
os.environ["HF_TOKEN"] = "HF_APIKEY" # Replace YOUR_HF_TOKEN with your actual token

model_name = "google/gemma-3-4b-it" # You can choose other variants like "google/gemma-7b" or "google/gemma-3-2b"

# Load the tokenizer and model (this step downloads the model files)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# Example usage (inference)
input_text = "What is the capital of France?"
input_ids = tokenizer(input_text, return_tensors="pt").to(model.device)

outputs = model.generate(**input_ids, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
