from transformers import AutoModelForMaskedLM
from peft import get_peft_model, LoraConfig

# Regular Bert:
model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")

# Bert with frozen encoder:
# for name, param in model.named_parameters():
#     if name.startswith("bert.encoder"):
#         print(f"{name} frozen")
#         param.requires_grad = False

# Bert with LoRA:
# lora_config = LoraConfig(
#     r=16,
#     lora_alpha=32,
#     target_modules=["query", "value"],
#     lora_dropout=0.1,
#     bias="none",
#     task_type="CAUSAL_LM",
# )
# model = get_peft_model(model, lora_config)
