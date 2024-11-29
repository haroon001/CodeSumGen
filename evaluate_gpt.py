from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import os

#os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
from torch.utils.data import DataLoader
from evaluate import load
import numpy as np
from tqdm import tqdm
from Levenshtein import distance as levenshtein_distance
from codebleu import calc_codebleu
import random


def prepare_data(examples, tkzr, max_length=256):
    texts = ["".join(example) for example in examples["code"]]

    encodings = tkzr(
        texts,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    )

    encodings["labels"] = encodings["input_ids"].clone()
    return encodings


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token  # Use EOS as padding token
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Add padding token
model = AutoModelForCausalLM.from_pretrained("distilgpt2_local")
model.resize_token_embeddings(len(tokenizer))  # Resize for padding token
model = model.to(device)

ds = load_dataset("Fraol/Py150-processed")
# train_ds = ds["train"].select(range(60000))
val_ds = ds["val"].select(range(7000))
test_ds = ds["test"].select(range(7000))

# for testing
val_ds = val_ds.select(range(5))
test_ds = test_ds.select(range(5))

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

print("Preparing datasets...")
val_ds = val_ds.map(
    lambda x: prepare_data(x, tokenizer),
    batched=True,
    remove_columns=val_ds.column_names,
)
test_ds = test_ds.map(
    lambda x: prepare_data(x, tokenizer),
    batched=True,
    remove_columns=test_ds.column_names,
)
val_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
test_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

#Load metrics
rouge = load("rouge")


def compute_metrics(preds, labels):
    total_dist, cut_dist, total_chars = 0, 0, 0
    for pred, label in zip(preds, labels):
        total_dist += levenshtein_distance(pred, label)
        min_len = min(len(pred), len(label))
        cut_dist += levenshtein_distance(pred[:min_len], label[:min_len])
        total_chars += len(labels)
    avg_lev = total_dist / len(preds)
    avg_cut_lev = cut_dist / len(preds)
    normed_lev = total_dist / total_chars if total_chars else 0

    # codebleu
    cb_res = calc_codebleu(labels, preds, lang="python")

    # ROUGE score
    rouge_score = rouge.compute(predictions=preds,
                                references=labels,
                                use_stemmer=True)

    return {
        "rouge1": rouge_score["rouge1"],
        "rougeL": rouge_score["rougeL"],
        "avg_lev": avg_lev,
        "normalized_lev": normed_lev,
        "cut_lev": avg_cut_lev,
        "codebleu": cb_res['codebleu'],
    }


eval_batch_size = 8

training_args = TrainingArguments(
    output_dir="./code-gen-results",
    eval_strategy="steps",
    eval_steps=750,
    num_train_epochs=3,
    learning_rate=1e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=eval_batch_size,
    save_total_limit=3,
    warmup_steps=500,
    save_steps=750,
    fp16=False,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=val_ds,
)

model.eval()
val_loader = DataLoader(val_ds, batch_size=eval_batch_size)
test_loader = DataLoader(test_ds, batch_size=eval_batch_size)


def evaluate_completion(model, tokenizer, eval_loader, num_tokens_to_predict=5):
    all_preds = []
    all_targets = []

    for batch in eval_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # For each sequence in batch
        for i in range(input_ids.shape[0]):
            # Get sequence length excluding padding
            seq_len = (input_ids[i] != tokenizer.pad_token_id).sum()

            # Add checks for sequence length
            if seq_len <= num_tokens_to_predict + 1:
                print(f"Skipping sequence {i} - too short (length {seq_len})")
                continue

            # Select a random position to cut the sequence
            cut_point = random.randint(1,
                                       seq_len - num_tokens_to_predict - 1)  # Start from 1 to ensure non-empty prefix

            # Get the input prefix and its attention mask
            prefix = input_ids[i, :cut_point].unsqueeze(0)
            prefix_attention_mask = attention_mask[i, :cut_point].unsqueeze(0)

            # Debug prints
            # print(f"Sequence length: {seq_len}")
            # print(f"Cut point: {cut_point}")
            # print(f"Prefix shape: {prefix.shape}")
            # print(f"Prefix attention mask shape: {prefix_attention_mask.shape}")

            # Additional check for empty prefix
            if prefix.shape[1] == 0:
                print(f"Skipping sequence {i} - empty prefix")
                continue

            # Get the target tokens
            target = labels[i, cut_point:cut_point + num_tokens_to_predict]

            # Generate completion
            with torch.no_grad():
                outputs = model.generate(
                    prefix,
                    attention_mask=prefix_attention_mask,
                    max_new_tokens=num_tokens_to_predict,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            # Get the predicted tokens (excluding the prefix)
            pred = outputs[0, cut_point:cut_point + num_tokens_to_predict]

            # Decode and save predictions and targets
            pred_text = tokenizer.decode(pred)
            target_text = tokenizer.decode(target)

            print(f"Target: {target_text}")
            print(f"Pred: {pred_text}")

            all_preds.append(pred_text)
            all_targets.append(target_text)

    if not all_preds:
        print("Warning: No valid predictions generated!")
        return None

    return compute_metrics(all_preds, all_targets)


with torch.no_grad():
    torch.cuda.empty_cache()
    print('Evaluating on validation set')
    val_metrics = evaluate_completion(model, tokenizer, val_loader)
    print('Validation metrics:')
    print(val_metrics)
    print('Evaluating on test set')
    test_metrics = evaluate_completion(model, tokenizer, test_loader)
    print('Test metrics:')
    print(test_metrics)
