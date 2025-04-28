from datasets import load_dataset

echr = load_dataset("json", data_files="data/echr.jsonl", split="train")

def change_mask(line):
    for p in line["pii_mask"]:
        line["masked_seq"] = line["masked_seq"].replace(f"[{p["label"]}]", "X")
    return line

echr = echr.map(change_mask)

echr.to_json("data/echr_masked.jsonl", lines=True)
