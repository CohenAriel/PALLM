from torch.utils.data import Dataset
from torch import tensor


class MaskedDataset(Dataset):
    def __init__(self, text, tokenizer, max_length=256):
        super().__init__()

        self.text = text

        self.encoded = tokenizer.batch_encode_plus(
            self.text,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_attention_mask=True,
        )

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        input_ids = tensor(self.encoded["input_ids"][idx])
        attention_mask = tensor(self.encoded["attention_mask"][idx])
        target = input_ids.clone()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target": target,
        }
