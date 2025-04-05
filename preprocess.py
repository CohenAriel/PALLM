from transformers import AutoTokenizer
from data import echr_data
from itertools import chain

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def group_texts(examples):
    tokenized_inputs = tokenizer(
       echr_data[:]["text"], return_special_tokens_mask=True, truncation=True, max_length=tokenizer.model_max_length
    )
    return tokenized_inputs

tokenized_data = echr_data.map(group_texts, batched=True, remove_columns=["text"])

def con_texts(examples):
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    if total_length >= tokenizer.model_max_length:
        total_length = (total_length // tokenizer.model_max_length) * tokenizer.model_max_length
    result = {
        k: [t[i : i + tokenizer.model_max_length] for i in range(0, total_length, tokenizer.model_max_length)]
        for k, t in concatenated_examples.items()
    }
    return result

con_data = tokenized_data.map(con_texts, batched=True)

con_data.shuffle(seed=42)

con_data.save_to_disk("data/processed_echr")
