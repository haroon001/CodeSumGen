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
    texts = [" ".join(example) for example in examples["code"]]

    encodings = tkzr(
        texts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    encodings["labels"] = encodings["input_ids"].clone()

    return encodings


login('')
cuda = torch.cuda.is_available()
mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
device = "cuda" if cuda \
    else "mps" if mps \
    else "cpu"
print(f'Using {device}')

tokenizer = AutoTokenizer.from_pretrained("microsoft/CodeGPT-small-py")
model = AutoModelForCausalLM.from_pretrained("microsoft/CodeGPT-small-py").to(device)

# ds = load_dataset("google/code_x_glue_cc_code_completion_token", "python")
# temp = ds["train"].train_test_split(test_size=0.1, seed=42)
# train_ds, val_ds = temp["train"], temp["test"]
# del temp
ds = load_dataset("Fraol/Py150-processed")
train_ds = ds["train"]
val_ds = ds["val"]
test_ds = ds["test"]
# train_ds = ds["train"].select(range(30000))
# val_ds = ds["val"].select(range(2000))
# test_ds = ds["test"].select(range(2000))

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

rouge = load("rouge")
bleu = load("bleu")

collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

training_args = TrainingArguments(
    output_dir="./code-gen-results",
    eval_strategy="steps",
    eval_steps=500,
    num_train_epochs=3,
    learning_rate=1e-5,
    per_device_train_batch_size=4 if not cuda else 16,
    per_device_eval_batch_size=4 if not cuda else 16,
    save_total_limit=3,
    warmup_steps=500,
    save_steps=500,
    fp16=cuda,
    bf16=mps,
    gradient_accumulation_steps=8,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collator,
    # compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model("./code-token-gen-final")

# need to add further evaluation metrics, but want to check to see how this works first

