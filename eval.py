from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
import torch

model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
current_model_dict = model.state_dict()
loaded_state_dict = torch.load("./models/bert-echr-normal-loss.pth", map_location="cpu")
new_state_dict = {
    k: v if v.size() == current_model_dict[k].size() else current_model_dict[k]
    for k, v in zip(current_model_dict.keys(), loaded_state_dict.values())
}
model.load_state_dict(new_state_dict, strict=False)

mask_filler = pipeline(
    task="fill-mask",
    model=model,
    tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased"),
)

text = "Harry Potter lives in [MASK]"
print(mask_filler(text, top_k=5))
