# This file will evaluate our distilgpt2 fine-tuned model with the Py150 dataset.
# In this file, we generate k beam predictions and use that for a simple form of pass@k
# We also calculate the BLEU, ROUGE, and Levenshtein distance metrics for the model.
#

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
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
val_ds = val_ds.select(range(50))
test_ds = test_ds.select(range(50))

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

# Load metrics
bleu = load("bleu")
rouge = load("rouge")


def compute_metrics(preds, labels, beam_preds=None, top_k=None):
    if top_k is None:
        top_k = [1, 3, 5]


    metrics = {}

    # Calculate pass@k if beam predictions are provided
    if beam_preds is not None:
        for k in top_k:
            correct = 0
            for beams, target in zip(beam_preds, labels):
                candidates = [pred.strip() for pred in beams[:k]]
                #print(candidates)
                if target.strip() in candidates:
                    correct += 1
            metrics[f'pass@{k}'] = correct / len(labels)

    # Calculate other metrics using the best prediction (first beam)
    total_dist, cut_dist, total_chars = 0, 0, 0
    exact_matches = 0

    for pred, label in zip(preds, labels):
        if pred.strip() == label.strip():
            exact_matches += 1
        total_dist += levenshtein_distance(pred, label)
        min_len = min(len(pred), len(label))
        cut_dist += levenshtein_distance(pred[:min_len], label[:min_len])
        total_chars += len(labels)

    avg_lev = total_dist / len(preds)
    avg_cut_lev = cut_dist / len(preds)
    normed_lev = total_dist / total_chars if total_chars else 0

    bleu_score = bleu.compute(predictions=preds, references=labels)

    # codebleu
    cb_res = calc_codebleu(labels, preds, lang="python")

    # ROUGE score
    rouge_score = rouge.compute(predictions=preds,
                                references=labels,
                                use_stemmer=True)

    metrics.update({
        "accuracy": exact_matches / len(preds),
        "bleu": bleu_score['bleu'],
        "rouge1": rouge_score["rouge1"],
        "rouge2": rouge_score["rouge2"],
        "rougeL": rouge_score["rougeL"],
        "avg_lev": avg_lev,
        "normalized_lev": normed_lev,
        "cut_lev": avg_cut_lev,
        "codebleu": cb_res['codebleu'],
    })

    return metrics


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

def evaluate_completion(model, tokenizer, eval_loader, num_tokens_to_predict=3, num_beams=5):
    all_preds = []
    all_beam_preds = []
    all_targets = []

    for batch in tqdm(eval_loader, desc="Evaluating"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        for i in range(input_ids.shape[0]):
            seq_len = (input_ids[i] != tokenizer.pad_token_id).sum()

            if seq_len <= num_tokens_to_predict + 1:
                print(f"Skipping sequence {i} - too short (length {seq_len})")
                continue

            cut_point = random.randint(1, seq_len - num_tokens_to_predict - 1)
            
            prefix = input_ids[i, :cut_point].unsqueeze(0)
            prefix_attention_mask = attention_mask[i, :cut_point].unsqueeze(0)

            if prefix.shape[1] == 0:
                print(f"Skipping sequence {i} - empty prefix")
                continue

            target = labels[i, cut_point:cut_point + num_tokens_to_predict]

            with torch.no_grad():
                outputs = model.generate(
    prefix,
    attention_mask=prefix_attention_mask,
    max_new_tokens=num_tokens_to_predict,
    min_new_tokens=num_tokens_to_predict,
    num_beams=num_beams,
    num_return_sequences=num_beams,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    output_scores=True,
    return_dict_in_generate=True,
    no_repeat_ngram_size=0,
    length_penalty=0.0
)

                sequences = outputs.sequences

                beam_preds = []
                for beam_idx in range(num_beams):
                    pred = sequences[beam_idx, prefix.shape[1]:prefix.shape[1] + num_tokens_to_predict]
                    pred_text = tokenizer.decode(pred, skip_special_tokens=True)
                    beam_preds.append(pred_text) 
                print(beam_preds)

                all_beam_preds.append(beam_preds)
                all_preds.append(beam_preds[0])
                
                target_text = tokenizer.decode(target, skip_special_tokens=True)
                all_targets.append(target_text)

    if not all_preds:
        print("Warning: No valid predictions generated!")
        return None

    return compute_metrics(all_preds, all_targets, all_beam_preds)


with torch.no_grad():
    torch.cuda.empty_cache()
    print('Evaluating on validation set')
    val_metrics = evaluate_completion(model, tokenizer, val_loader, num_tokens_to_predict=5, num_beams=5)
    print('Validation metrics:')
    print(val_metrics)
    print('Evaluating on test set')
    test_metrics = evaluate_completion(model, tokenizer, test_loader, num_tokens_to_predict=5, num_beams=5)
    print('Test metrics:')
    print(test_metrics)
