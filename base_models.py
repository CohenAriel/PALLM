from transformers import BertForSequenceClassification
from transformers import AutoTokenizer

bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
bert_model = BertForSequenceClassification.from_pretrained('bert-base-cased')

