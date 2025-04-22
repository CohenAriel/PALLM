import gzip

with gzip.open("./data/echr.jsonl.gz", "rb") as f:
    echr_data = f.read()
with gzip.open("./data/enron.jsonl.gz", "rb") as f:
    enron_data = f.read()

with open("./data/echr.jsonl", "w") as f:
    f.write(echr_data.decode("utf-8"))
with open("./data/enron.jsonl", "w") as f:
    f.write(enron_data.decode("utf-8"))
