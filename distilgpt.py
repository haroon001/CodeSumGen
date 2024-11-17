from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import numpy as np
from evaluate import load
import torch
from huggingface_hub import login


# from train_sum_mps_and_cpu import compute_metrics


def prepare_data(examples, tkzr, max_length=1024):
    texts = ["".join(example) for example in examples["code"]]

    encodings = tkzr(
        texts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    encodings["labels"] = encodings["input_ids"].clone()

    return encodings


cuda = torch.cuda.is_available()
mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
device = "cuda" if cuda \
    else "mps" if mps \
    else "cpu"
print(f'Using {device}')

# Load model directly
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2").to(device)

# tokenizer = AutoTokenizer.from_pretrained("microsoft/CodeGPT-small-py")
# model = AutoModelForCausalLM.from_pretrained("microsoft/CodeGPT-small-py").to(device)

# ds = load_dataset("google/code_x_glue_cc_code_completion_token", "python")
# temp = ds["train"].train_test_split(test_size=0.1, seed=42)
# train_ds, val_ds = temp["train"], temp["test"]
# del temp
ds = load_dataset("Fraol/Py150-processed")
train_ds = ds["train"].select(range(60000))
val_ds = ds["val"].select(range(7500))
test_ds = ds["test"].select(range(7500))

print("mapping train")
train_ds = train_ds.map(
    lambda x: prepare_data(x, tokenizer),
    batched=True,
    remove_columns=train_ds.column_names
)

print("mapping validation")
val_ds = val_ds.map(
    lambda x: prepare_data(x, tokenizer),
    batched=True,
    remove_columns=val_ds.column_names
)

print("mapping test")
test_ds = test_ds.map(
    lambda x: prepare_data(x, tokenizer),
    batched=True,
    remove_columns=test_ds.column_names
)

# # save datasets to my drive
# train_ds.save_to_disk("/content/drive/MyDrive/train_ds")
# val_ds.save_to_disk("/content/drive/MyDrive/val_ds")
# test_ds.save_to_disk("/content/drive/MyDrive/test_ds")

# Load metrics
# rouge = load("rouge")
# bleu = load("bleu")
#
# def compute_metrics(eval_pred):
#     # Unpack predictions and labels
#     logits, labels = eval_pred
#
#     # Get the predictions
#     predictions = np.argmax(logits, axis=-1)
#
#     # Decode tokens to strings
#     decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
#     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
#
#     # Postprocess to align text (strip extra spaces, etc.)
#     decoded_preds = [pred.strip() for pred in decoded_preds]
#     decoded_labels = [label.strip() for label in decoded_labels]
#
#     tokenized_preds = [pred.split() for pred in decoded_preds]
#     tokenized_labels = [label.split() for label in decoded_labels]
#
#
#
#     # BLEU score
#     bleu_score = bleu.compute(predictions=tokenized_preds,
#                               references=tokenized_labels)
#
#     # ROUGE score
#     rouge_score = rouge.compute(predictions=decoded_preds,
#                                  references=decoded_labels,
#                                  use_stemmer=True)
#
#     # Optionally add custom metrics like perplexity
#     # For perplexity, use loss values from the Trainer logs
#
#     return {
#         "bleu": bleu_score["bleu"],
#         "rouge1": rouge_score["rouge1"].mid.fmeasure,
#         "rouge2": rouge_score["rouge2"].mid.fmeasure,
#         "rougeL": rouge_score["rougeL"].mid.fmeasure
#     }


training_args = TrainingArguments(
    output_dir="./code-gen-results",
    eval_strategy="steps",
    eval_steps=750,
    num_train_epochs=3,
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    save_total_limit=3,
    warmup_steps=500,
    save_steps=750,
    fp16=False,
    gradient_accumulation_steps=4,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    # compute_metrics=compute_metrics
)
torch.cuda.empty_cache()
trainer.train()
trainer.save_model("./code-token-distilgpt-final")
print(trainer.evaluate(test_ds))

# need to add further evaluation metrics, but want to check to see how this works first


