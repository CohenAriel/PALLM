import json
from transformers import AutoTokenizer
import torch
from numpy import random

with open('data/synonyms.json', 'r') as f:
    synonyms = json.load(f)

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

tokenized_synonyms = {}

for k in synonyms:
    for v in synonyms[k]:
        tokenized = tokenizer(v, add_special_tokens=False)["input_ids"]
        if len(tokenized) == 1:
            tokenized_synonyms[k] = tokenized_synonyms.get(k, []) + tokenized

for k in tokenized_synonyms:
    tokenized_synonyms[k] += (tokenized_synonyms["SENSITIVE"])
    tokenized_synonyms[k] = torch.tensor(tokenized_synonyms[k], dtype=torch.float)


num_masks = {
    1: 'DATE_TIME',
    2: 'ID',
    3: 'LOCATION',
    4: 'MEDICAL_LICENSE',
    5: 'ORGANIZATION',
    6: 'PERSON',
    7: 'PHONE_NUMBER',
    8: 'URL',
}


def get_random_synonyms(nums):
    randtensor = torch.zeros((nums.size(0), 9, nums.size(1)), dtype=torch.long)
    for i in range(1, 9):
        idx = random.randint(0, len(tokenized_synonyms[num_masks[i]])-1, (nums.size(0), nums.size(1)))
        randtensor[:,i] = tokenized_synonyms[num_masks[i]][idx]
    return torch.gather(randtensor, 1, nums.unsqueeze(1)).squeeze(1)


if __name__ == "__main__":
    print(get_random_synonyms(torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8],
                                            [0, 1, 2, 3, 4, 5, 6, 7, 8]])))
