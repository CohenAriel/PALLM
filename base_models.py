from transformers import AutoModelForMaskedLM

# Regular Bert:
model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")

# Bert with frozen encoder:
for name, param in model.named_parameters():
    print(name)
    if name.startswith("bert.encoder"):
        print(f"{name} frozen")
        param.requires_grad = False
