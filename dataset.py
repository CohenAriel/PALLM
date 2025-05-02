from torch.utils.data import Dataset
from torch import tensor

mask_nums = {
    'DATE_TIME': 1,
    'ID': 2,
    'LOCATION': 3,
    'MEDICAL_LICENSE': 4,
    'ORGANIZATION': 5,
    'PERSON': 6,
    'PHONE_NUMBER': 7,
    'URL': 8,
}

class MaskedDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        super().__init__()

        self.text = data[:]["text"]

        encoded_pii = [[(l["label"], tokenizer(l["value"], add_special_tokens=False)["input_ids"]) for l in line] for line in data[:]["pii_mask"]]

        self.encoded = tokenizer.batch_encode_plus(
            data[:]["text"],
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_attention_mask=True,
        )

        self.mask_map = []
        for input_ids, pii_line in zip(self.encoded["input_ids"], encoded_pii):
            masked = []
            i = 0
            while i < len(input_ids):
                curr_mask = ""
                is_masked = False
                for pii_tokens in pii_line:
                    if input_ids[i:i+len(pii_tokens[1])] == pii_tokens[1]:
                        is_masked = True
                        curr_mask = pii_tokens[0]
                        break
                if is_masked:
                    for _ in range(len(pii_tokens[1])):
                        masked.append(mask_nums[curr_mask[:-2]])
                    i += len(pii_tokens[1])
                else:
                    masked.append(0)
                    i += 1
            self.mask_map.append(masked)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        input_ids = tensor(self.encoded["input_ids"][idx])
        mask_map = tensor(self.mask_map[idx])
        attention_mask = tensor(self.encoded["attention_mask"][idx])
        target = input_ids.clone()
        return {
            "input_ids": input_ids,
            "mask_map": mask_map,
            "attention_mask": attention_mask,
            "target": target,
        }

if __name__ == "__main__":
    from datasets import load_dataset
    from transformers import AutoTokenizer

    bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    echr = load_dataset("json", data_files="data/echr.jsonl", split="train")

    echr_train_dataset = MaskedDataset(data=echr, tokenizer=bert_tokenizer)
