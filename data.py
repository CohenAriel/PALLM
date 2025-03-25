import pandas as pd

echr_data = pd.read_json('./data/echr.jsonl', lines=True)
enron_data = pd.read_json('./data/enron.jsonl', lines=True)
