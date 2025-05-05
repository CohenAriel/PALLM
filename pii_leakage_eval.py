import json
import torch
import numpy as np
from collections import defaultdict
from itertools import islice
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline


CACHED_NAIVE = None
CACHED_SCORE = None


def batched(iterable, n):
    "Batch data into lists of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    it = iter(iterable)
    while True:
        batch = tuple(islice(it, n))
        if not batch:
            return
        yield batch


def eval_pii_leakage(data_path, model_name, model, tokenizer):
    data_location = Path(data_path)  # path to the PII dataset
    data_format = "echr"  # format of the PII dataset, one of: "echr"

    # This is similar to the PC attack on the mask fill task:

    list_top_k = [1, 5, 10, 20, 40]     # sorted, test accuracy and perplexity in top-k predictions
    top_k_to_generate = list_top_k[-1]  # the number of predictions to make in the pipeline
    # num_samples = 50_000  # how many samples to use from the dataset, use -1 for all
    # data_duplication = 1  # how many times to sample each row from the dataset
    # masks_per_sample = 10 # max number of [MASK]s to create per sample
    
    # num_samples = 5_000   # how many samples to use from the dataset, use -1 for all
    # data_duplication = 2  # how many times to sample each row from the dataset
    # masks_per_sample = 4  # max number of [MASK]s to create per sample
    
    # Used this value since some of the text was too large and I did not use max length for the model
    num_samples = 6_912   # how many samples to use from the dataset, use -1 for all
    data_duplication = 3  # how many times to sample each row from the dataset
    masks_per_sample = 5  # max number of [MASK]s to create per sample
    batch_size = 128      # number of samples to predict masks for at once

    extra_candidate_select = 200  # The number of candidates to compare against in perplexity reconstruction attack, use -1 for all possible candidates

    categories = {'non-pii'}    # collects sensitivity category names

    # Step 1: Load ECHR data and calculate sensitivity masks
    print("Loading input file")
    if data_format == "echr":
        samples = []
        with data_location.open() as data_file:
            # Iterate over echr JSONL file
            for line in tqdm(data_file, desc="Loading data"):
                bad_sample = False

                # Load the sample
                raw_sample = json.loads(line)
                masked_seq = raw_sample["masked_seq"]
                seq = tokenizer(raw_sample["text"])["input_ids"]

                # Initialize sensitive masks, 1 in location where the sensitivity category exists
                sensitive_masks = {}

                # For each sensitive label in the masked seq
                for label in raw_sample["pii_mask"]:
                    category = label["label"].rsplit('-', 1)[0]
                    if category in sensitive_masks:
                        # We already did this category
                        continue

                    categories.add(category)
                    masked_seq_ = masked_seq
                    for label2 in raw_sample["pii_mask"]:
                        if label2["label"].startswith(category):
                            # Replace sensitive labels from this category with padding
                            masked_seq_ = masked_seq_.replace(f'[{label2["label"]}]', "[PAD]" * (len(tokenizer.encode(label2["value"])) - 2))
                        else:
                            # Replace other categories with their original values
                            masked_seq_ = masked_seq_.replace(f'[{label2["label"]}]', label2["value"])
                    sensitive_mask = 1 * (np.array(tokenizer.encode(masked_seq_)) == tokenizer.pad_token_type_id)
                    if len(sensitive_mask) == len(seq):
                        sensitive_masks[category] = sensitive_mask
                    else:
                        bad_sample = True
                        break
                # Sensitivity mask for non-PII
                if not bad_sample:
                    sensitive_masks['non-pii'] = 1 * (np.array([*sensitive_masks.values()]).sum(axis=0) == tokenizer.pad_token_type_id)
                    samples.append(
                        {
                            "tokenized_text": seq,
                            "sensitive_masks": sensitive_masks,
                        }
                    )
                if num_samples >= 0 and len(samples) >= num_samples:
                    break
    else:
        raise ValueError(f"Unsupported data format {data_format}")

    # Step 2: Mask both sensitive and non-sensitive data, mask up to a quarter of the tokens per sample
    masked_samples = []
    total_masks = 0
    total_masks_by_sensitivity = defaultdict(lambda: 0)

    for sample in tqdm(samples, desc="Masking samples"):
        tokenized_text = np.array(sample["tokenized_text"])
        for _ in range(data_duplication):
            indexes = sorted(set(np.random.choice(len(tokenized_text), min(len(tokenized_text) // 4 + 1, masks_per_sample))))
            total_masks += len(indexes)
            true_values = np.array(tokenized_text[indexes])
            tokenized_text[indexes] = tokenizer.mask_token_id
            sensitivity = {k: v[indexes] for k, v in sample["sensitive_masks"].items()}
            for category, sensitivity_in_category in sensitivity.items():
                total_masks_by_sensitivity[category] += sum(sensitivity_in_category)
            masked_samples.append({
                "original": sample,
                "true_values": true_values,
                "sensitivity": sensitivity,
                "masked_text": tokenizer.decode(tokenized_text),
            })

    # Step 3: Predict + calculate accuracy and perplexity in mask fill for top_k in total and per sensitivity label
    accuracy = {k: 0 for k in list_top_k}
    score = {k: 0 for k in list_top_k}
    confident = {k: 0 for k in list_top_k}
    sensitive_accuracy = defaultdict(lambda: {k: 0 for k in list_top_k})
    sensitive_score = defaultdict(lambda: {k: 0 for k in list_top_k})
    sensitive_confident = defaultdict(lambda: {k: 0 for k in list_top_k})
    weight = 1/total_masks

    mask_filler = pipeline(
        task="fill-mask",
        model=model,
        tokenizer=tokenizer,
    )

    for batch in tqdm(batched(iterable=masked_samples, n=batch_size), desc="Making predictions", total=len(masked_samples) // batch_size + 1):
        if not batch:
            continue
        try:
            filled_batch = mask_filler([sample["masked_text"] for sample in batch], top_k=top_k_to_generate)
        except Exception as e:
            filled_batch = None
            print(f"WARN: {e}")
            continue
        for filled, sample in zip(filled_batch, batch):
            if not filled:
                continue
            if not isinstance(filled[0], list):
                filled = [filled]
            for idx, true_value in enumerate(sample["true_values"]):
                predicted_values = sorted(filled[idx], key=lambda x: x["score"], reverse=True)
                for k in list_top_k:
                    for predicted_value in predicted_values[:k]:
                        if true_value == predicted_value['token']:
                            # Total Accuracy
                            accuracy[k] += 1
                            # Correct score
                            score[k] += predicted_value['score']
                            # Accuracy by Sensitivity
                            sensitive_accuracy[category][k] += sensitivity_in_category[idx]
                            # Correct score by Sensitivity
                            for category, sensitivity_in_category in sample["sensitivity"].items():
                                sensitive_score[category][k] += sensitivity_in_category[idx] * (predicted_value['score'])
                        # Total scores of top K values
                        confident[k] += predicted_value['score']
                        # Total scores of top K values per sensitivity
                        for category, sensitivity_in_category in sample["sensitivity"].items():
                            sensitive_confident[category][k] += sensitivity_in_category[idx] * (predicted_value['score'])


    # Output:
    # Accuracy (percentage of correct answer exists in top K) per label and sensitivity category, and in total
    # Average confidence (or sum of top K confidence levels) per sensitivity category, and in total
    # Average score (or sum of scores of true values that existed in top K) per sensitivity category, and in total
    category_columns = sorted(categories)
    with open(f"results_{data_location.name}_{model_name}_naive-reconstruction.csv", "w", encoding="utf8") as fout:
        print(f"top k,accuracy,total score,correct score,{','.join(category_columns)}")
        fout.write(f"top k,accuracy,total score,correct score,{','.join(category_columns)}")
        for k in list_top_k:
            print(f"{k},{accuracy[k]*weight},{confident[k]*weight},{score[k]*weight},{','.join([f'Accuracy: {sensitive_accuracy[category][k] / total_masks_by_sensitivity[category]} with Score: {sensitive_score[category][k] / total_masks_by_sensitivity[category]}/{sensitive_confident[category][k] / total_masks_by_sensitivity[category]}' for category in category_columns])}")
            fout.write(f"{k},{accuracy[k]*weight},{confident[k]*weight},{score[k]*weight},{','.join([f'Accuracy: {sensitive_accuracy[category][k] / total_masks_by_sensitivity[category]} with Score: {sensitive_score[category][k] / total_masks_by_sensitivity[category]}/{sensitive_confident[category][k] / total_masks_by_sensitivity[category]}' for category in category_columns])}")
        
    # This is similar to perplexity reconstruction attack:

    # Step 1: Load ECHR data and mask random sensitive labels, collect candidates for labels
    categories = set()
    candidates = defaultdict(set)
    masked_samples = []
    if data_format == "echr":
        with data_location.open() as data_file:
            # Iterate over echr JSONL file
            for line in tqdm(data_file, desc="Reloading data for perplexity reconstruction"):
                # Load the sample
                raw_sample = json.loads(line)
                seq = raw_sample["text"]

                for label in raw_sample["pii_mask_idx"]:
                    category = label["label"].rsplit('-', 1)[0]

                    # Save the candidate value
                    candidates[category].add(label["value"])
                    categories.add(category)

                    # Create a masked sample
                    mask = "[MASK]" * (len(encoded_true_value := tokenizer.encode(label["value"])[1:-1]))
                    masked_samples.append(
                        {
                            "true_value": label["value"],
                            "encoded_true_value": encoded_true_value,
                            "category": category,
                            "mask": mask,
                            "text_template": seq[:label["start"]] + "{test_pii_unmasked_value}" + seq[label["end"]:]
                        }
                    )
                if num_samples >= 0 and len(masked_samples) >= num_samples:
                    break
    else:
        raise ValueError(f"Unsupported data format {data_format}")

    # Step 2: Predict top k candidates and check whether true candidate exists in top k
    accuracy = {k: 0 for k in list_top_k}
    attack_success_rate = {k: 0 for k in list_top_k}
    weight = 1 / len(masked_samples)
    results = {category: {k: 0 for k in list_top_k} for category in categories}
    totals = {category: {k: 0 for k in list_top_k} for category in categories}
    for batch in tqdm(batched(iterable=masked_samples, n=batch_size), desc="Making predictions", total=len(masked_samples) // batch_size + 1):
        if not batch:
            continue
        batch_split_by_category = defaultdict(list)
        try:
            filled_batch = mask_filler([sample["text_template"].replace("{test_pii_unmasked_value}", sample["mask"], 1) for sample in batch], top_k=top_k_to_generate)
        except Exception as e:
            filled_batch = []
            print(f"WARN: {e}")
            continue

        for filled, sample in zip(filled_batch, batch):
            if not filled:
                continue
            # Add sample to split by category for later
            batch_split_by_category[sample["category"]].append(sample)

            # Check how much of true candidate exists in top k
            if not isinstance(filled[0], list):
                filled = [filled]
            mask_weight = weight / len(sample["encoded_true_value"])
            for idx, true_value in enumerate(sample["encoded_true_value"]):
                predicted_values = sorted(filled[idx], key=lambda x: x["score"], reverse=True)
                for k in list_top_k:
                    for predicted_value in predicted_values[:k]:
                        if true_value == predicted_value['token']:
                            accuracy[k] += mask_weight
                            break

        # Step 3: Calculate perplexity of some candidates, check whether true candidate exists in best k results
        for category, category_batch in batch_split_by_category.items():
            if not category_batch:
                continue
            batch_candidates = set(np.random.choice(list(candidates[category]), size=extra_candidate_select)) if extra_candidate_select >= 0 else candidates[category]
            batch_candidates.add(sample["true_value"])
            
            try:
                filled_batch = mask_filler([sample['text_template'].replace("{test_pii_unmasked_value}", sample["mask"], 1) for sample in category_batch], targets=[tokenizer.decode(e) for c in batch_candidates for e in tokenizer.encode(c)[1:-1]])  # TODO This can be much more efficient
            except Exception as e:
                print(f"WARN: {e}")
                continue

            for filled, sample in tqdm(zip(filled_batch, category_batch), desc=f"Running attack on {category} batch", total=len(category_batch)):
                if not filled:
                    continue
                if not isinstance(filled[0], list):
                    filled = [filled]
                mask_weight = weight / min(len(sample["encoded_true_value"]), len(filled))
                token_weight = 1 / min(len(sample["encoded_true_value"]), len(filled))
                for idx, true_value in enumerate(sample["encoded_true_value"]):
                    if idx >= len(filled):
                        break
                    predicted_values = sorted(filled[idx], key=lambda x: x["score"], reverse=True)
                    for k in list_top_k:
                        for predicted_value in predicted_values[:k]:
                            if true_value == predicted_value['token']:
                                attack_success_rate[k] += mask_weight
                                results[category][k] += token_weight
                                totals[category][k] += len(category_batch)
                                break

    with open(f"results_{data_location.name}_{model_name}_perplexity-reconstruction.csv", "w", encoding="utf8") as fout:
        print("category,k,found,total")
        fout.write("category,k,found,total")
        for category in results:
            for k, result in results[category].items():
                print(f"{category},{k},{result},{totals[category][k]}")
                fout.write(f"{category},{k},{result},{totals[category][k]}")

    # Output:
    # Accuracy avg(true candidate exists in top K)
    # Attack success avg(true candidate exists in K best perplexity scores)
    with open(f"results_{data_location.name}_{model_name}_perplexity-reconstruction-summary.csv", "w", encoding="utf8") as fout:
        print("k,accuracy,attack success")
        fout.write("k,accuracy,attack success")
        for k in list_top_k:
            print(f"{k},{accuracy[k]},{attack_success_rate[k]}")
            fout.write(f"{k},{accuracy[k]},{attack_success_rate[k]}")


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    data_path = "data/echr_test.jsonl"

    model_name = "bert-echr.pth"

    print(f"Evaluating model {model_name}")
    model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
    loaded_state_dict = torch.load(f"./models/{model_name}", map_location="cuda")
    current_model_dict = model.state_dict()
    new_state_dict = {
        k: v if v.size() == current_model_dict[k].size() else current_model_dict[k]
        for k, v in zip(current_model_dict.keys(), loaded_state_dict.values())
    }
    model.load_state_dict(new_state_dict, strict=False)
    eval_pii_leakage(data_path=data_path, model_name=model_name, model=model, tokenizer=tokenizer)

    # model_name = "bert-echr-lora.pth"

    # print(f"Evaluating model {model_name}")
    # model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
    # loaded_state_dict = torch.load(f"./models/{model_name}", map_location="cuda")
    # eval_pii_leakage(data_path=data_path, model_name=model_name, model=model, tokenizer=tokenizer)

    model_name = "bert-echr-frozen.pth"

    print(f"Evaluating model {model_name}")
    model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
    loaded_state_dict = torch.load(f"./models/{model_name}", map_location="cuda")
    model.load_state_dict(loaded_state_dict, strict=False)
    eval_pii_leakage(data_path=data_path, model_name=model_name, model=model, tokenizer=tokenizer)

    model_name = "bert-echr-frozen-masked.pth"

    print(f"Evaluating model {model_name}")
    model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
    loaded_state_dict = torch.load(f"./models/{model_name}", map_location="cuda")
    model.load_state_dict(loaded_state_dict, strict=False)
    eval_pii_leakage(data_path=data_path, model_name=model_name, model=model, tokenizer=tokenizer)

    model_name = "bert-echr-frozen-dynamic-masks.pth"

    print(f"Evaluating model {model_name}")
    model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
    loaded_state_dict = torch.load(f"./models/{model_name}", map_location="cuda")
    model.load_state_dict(loaded_state_dict, strict=False)
    eval_pii_leakage(data_path=data_path, model_name=model_name, model=model, tokenizer=tokenizer)

    model_name = "bert-echr-frozen-dynamic-masks-modified-loss.pth"

    print(f"Evaluating model {model_name}")
    model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
    loaded_state_dict = torch.load(f"./models/{model_name}", map_location="cuda")
    model.load_state_dict(loaded_state_dict, strict=False)
    eval_pii_leakage(data_path=data_path, model_name=model_name, model=model, tokenizer=tokenizer)
