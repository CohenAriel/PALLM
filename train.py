import torch

_ = torch.set_grad_enabled(False)
from datasets import load_dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from dataset import MaskedDataset
from transformers import (
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
import time
from torch.optim import AdamW
from base_models import model

bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

echr = load_dataset("json", data_files="data/echr.jsonl", split="train")[:]["text"]
echr_train, echr_test = train_test_split(echr, test_size=0.1, random_state=124)

echr_train_dataset = MaskedDataset(text=echr_train, tokenizer=bert_tokenizer)
echr_test_dataset = MaskedDataset(text=echr_test, tokenizer=bert_tokenizer)

echr_train_loader = DataLoader(echr_train_dataset, batch_size=16, shuffle=True)
echr_test_loader = DataLoader(echr_test_dataset, batch_size=16, shuffle=True)

steps = len(echr_train_loader) * 3

optimizer = AdamW(model.parameters(), lr=2e-5)

scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer, num_warmup_steps=0.1, num_training_steps=steps, num_cycles=0.49
)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
model.to(device)

history = {
    "train_step": [],
    "train_loss": [],
    "valid_step": [],
    "valid_loss": [],
    "lr": [],
}

best_valid_loss = 1e5

# save_path = "./models/bert-echr-normal-loss.pth"
save_path = "./models/bert-echr-normal-loss-frozen.pth"

for epoch in range(3):
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

        outputs = model(inputs, labels=labels)
        train_loss = outputs.loss

        train_loss.requires_grad = True
        train_loss.backward()

        ### Accumulate gradients for a certain number of steps
        if (train_step + 1) % 4 == 0:
            ### Update parameters
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            ### append step results for train loop
            history["lr"].append(optimizer.param_groups[0]["lr"])
            history["train_step"].append(train_step + 1)
            history["train_loss"].append(train_loss)

        total_train_epoch_loss += train_loss.item()

        print(
            f"Epoch: {epoch + 1}/{3} | Train Step: {train_step + 1}/{len(echr_train_loader)} | "
            f"Train Loss: {round(train_loss.item(), 4)} | "
            f"LR: {round(current_LR, 8)} | "
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

            pred = model(inputs, labels=labels)
            valid_loss = pred.loss  # default MLM loss function

            total_valid_epoch_loss += valid_loss.item()

            ### append step results for valid loop
            history["valid_step"].append(valid_step + 1)
            history["valid_loss"].append(valid_loss)

    valid_epoch_loss_mean = round(total_valid_epoch_loss / len(echr_test_loader), 4)

    print(
        f"Epoch: {epoch + 1}/{3} | "
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
        if es_step >= 3:
            break

    dt = time.time() - dt
