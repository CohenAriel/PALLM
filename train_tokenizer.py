from tqdm import tqdm
from transformers import BertTokenizerFast
from data import echr_data


def batch_iterator(batch_size=10000):
    for i in tqdm(range(0, len(echr_data), batch_size)):
        yield echr_data[i : i + batch_size]["text"]


tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

bert_tokenizer = tokenizer.train_new_from_iterator(
    text_iterator=batch_iterator(), vocab_size=32_000
)
bert_tokenizer.save_pretrained("./tokenizer")
