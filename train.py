import torch

_ = torch.set_grad_enabled(False)
from datasets import load_dataset
from torch.utils.data import DataLoader
from dataset import MaskedDataset
from transformers import (
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    AutoModelForMaskedLM,
)
import time
from torch.optim import AdamW
from get_synonyms import get_random_synonyms
from torch.nn import CrossEntropyLoss

bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")

# Bert with frozen encoder:
for name, param in model.named_parameters():
    if name.startswith("bert.encoder.layer.1.") or\
        name.startswith("bert.encoder.layer.2.") or\
        name.startswith("bert.encoder.layer.2.") or\
        name.startswith("bert.encoder.layer.4.") or\
        name.startswith("bert.encoder.layer.5.") or\
        name.startswith("bert.encoder.layer.6."):
        print(f"{name} frozen")
        param.requires_grad = False
    else:
        print(f"{name} trainable")
        param.requires_grad = True

echr = load_dataset("json", data_files="data/echr.jsonl", split="train[:40%]")
echr_train, echr_test = echr.train_test_split(test_size=0.1, shuffle=False).values()
echr_train.to_json("data/echr_train.jsonl", lines=True)
echr_test.to_json("data/echr_test.jsonl", lines=True)

echr_train_dataset = MaskedDataset(data=echr_train, tokenizer=bert_tokenizer)
echr_test_dataset = MaskedDataset(data=echr_test, tokenizer=bert_tokenizer)

echr_train_loader = DataLoader(echr_train_dataset, batch_size=16, shuffle=True)
echr_test_loader = DataLoader(echr_test_dataset, batch_size=16, shuffle=True)

steps = len(echr_train_loader) * 2

optimizer = AdamW(model.parameters(), lr=0.1)

scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer, num_warmup_steps=0, num_training_steps=steps, num_cycles=0.49
)

use_mask_map = True
modify_loss = True

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
model.to(device)

best_valid_loss = 1e5

save_path = "./models/bert-echr-frozen-dynamic-masks-modified-loss.pth"

for epoch in range(2):
    dt = time.time()

    current_LR = optimizer.param_groups[0]["lr"]

    ### training loop
    model.train()
    total_train_epoch_loss = 0.0

    optimizer.zero_grad()

    for train_step, batch in enumerate(echr_train_loader):

        inputs = batch["input_ids"].to(device)
        labels = batch["target"].to(device)

        ### mask a percentage of tokens
        mask_indices = torch.bernoulli(torch.full(labels.shape, 0.15)).bool()
        labels[~mask_indices] = (
            -100
        )  # ignore non-masked tokens, we only compute loss on masked tokens

        if use_mask_map:
            mask_map = batch["mask_map"]
            masked_pii = torch.logical_and(mask_map, mask_indices).to(device)
            synonyms = get_random_synonyms(mask_map).to(device)
            old_labels = labels.clone()
            labels = torch.where(masked_pii, synonyms, labels)

        outputs = model(inputs, labels=labels)
        train_loss = outputs.loss

        if modify_loss:
            loss_fct = CrossEntropyLoss()
            old_labels[~(mask_map > 0)] = ( -100 )
            train_loss -= loss_fct(
                outputs.logits.view(-1, model.config.vocab_size),
                old_labels.view(-1),
            )

        train_loss.requires_grad = True
        train_loss.backward()

        ### Accumulate gradients for a certain number of steps
        if (train_step + 1) % 4 == 0:
            ### Update parameters
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        total_train_epoch_loss += train_loss.item()

        print(
            f"Epoch: {epoch + 1}/{2} | Train Step: {train_step + 1}/{len(echr_train_loader)} | "
            f"Train Loss: {round(train_loss.item(), 4)} | "
            f"LR: {current_LR} | "
            f"Time: {round(time.time() - dt, 2)}s"
        )

    train_epoch_loss_mean = round(total_train_epoch_loss / len(echr_test_loader), 4)

    ### evaluation loop
    model.eval()
    total_valid_epoch_loss = 0.0
    for valid_step, batch in enumerate(echr_test_loader):

        with torch.no_grad():
            inputs = batch["input_ids"].to(device)
            labels = batch["target"].to(device)

            ### mask a percentage of tokens
            mask_indices = torch.bernoulli(torch.full(labels.shape, 0.15)).bool()
            labels[~mask_indices] = (
                -100
            )  # ignore non-masked tokens, we only compute loss on masked tokens

            if use_mask_map:
                mask_map = batch["mask_map"]
                masked_pii = torch.logical_and(mask_map, mask_indices).to(device)
                synonyms = get_random_synonyms(mask_map).to(device)
                old_labels = labels.clone()
                labels = torch.where(masked_pii, synonyms, labels)

            pred = model(inputs, labels=labels)
            valid_loss = pred.loss  # default MLM loss function

            if modify_loss:
                loss_fct = CrossEntropyLoss()
                old_labels[~(mask_map > 0)] = ( -100 )
                valid_loss -= loss_fct(
                    pred.logits.view(-1, model.config.vocab_size),
                    old_labels.view(-1),
                )

            total_valid_epoch_loss += valid_loss.item()

    valid_epoch_loss_mean = round(total_valid_epoch_loss / len(echr_test_loader), 4)

    print(
        f"Epoch: {epoch + 1}/{2} | "
        f"Train Loss: {round(train_epoch_loss_mean, 4)} | "
        f"Valid Loss: {round(valid_epoch_loss_mean, 4)} | "
        f"Time: {round(time.time() - dt, 2)}s"
    )

    ### save checkpoint if valid loss improved
    if valid_epoch_loss_mean < best_valid_loss:
        best_valid_loss = valid_epoch_loss_mean
        torch.save(
            model.state_dict(),
            save_path,
        )
        es_step = 0
    else:
        es_step += 1
        if es_step >= 2:
            break

    dt = time.time() - dt
