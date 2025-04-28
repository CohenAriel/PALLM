from transformers import AutoModelForMaskedLM
from peft import get_peft_model, LoraConfig

# Regular Bert:
model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")

# Bert with frozen encoder:
for name, param in model.named_parameters():
    if name.startswith("bert.encoder"):
        print(f"{name} frozen")
        param.requires_grad = False
