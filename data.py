from datasets import load_dataset

echr_data = load_dataset('json', data_files='./data/echr.jsonl', split="train")
enron_data = load_dataset('json', data_files='./data/enron.jsonl', split="train")
