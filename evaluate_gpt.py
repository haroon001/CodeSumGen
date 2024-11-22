from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import torch
from evaluate import load
import numpy as np


def prepare_data(examples, tkzr, max_length=512):
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
model = AutoModelForCausalLM.from_pretrained("distilgpt2_local")
model.resize_token_embeddings(len(tokenizer))  # Resize for padding token
model = model.to(device)

ds = load_dataset("Fraol/Py150-processed")
# train_ds = ds["train"].select(range(60000))
val_ds = ds["val"].select(range(7500))
test_ds = ds["test"].select(range(7500))

# for testing

val_ds = val_ds.select(range(15))
test_ds = test_ds.select(range(15))

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

print("Preparing datasets...")
val_ds = val_ds.map(
    lambda x: prepare_data(x, tokenizer),
    batched=True,
    remove_columns=val_ds.column_names,
    num_proc=4
)
test_ds = test_ds.map(
    lambda x: prepare_data(x, tokenizer),
    batched=True,
    remove_columns=test_ds.column_names,
    num_proc=4
)

#Load metrics
rouge = load("rouge")
bleu = load("bleu")


def compute_metrics(eval_pred):
    # Unpack predictions and labels
    logits, labels = eval_pred

    # Get the predictions
    predictions = np.argmax(logits, axis=-1)

    # Decode tokens to strings
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Postprocess to align text (strip extra spaces, etc.)
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    # BLEU score
    bleu_score = bleu.compute(predictions=decoded_preds,
                              references=decoded_labels)

    # ROUGE score
    rouge_score = rouge.compute(predictions=decoded_preds,
                                references=decoded_labels,
                                use_stemmer=True)

    # Optionally add custom metrics like perplexity
    # For perplexity, use loss values from the Trainer logs

    return {
        "bleu": bleu_score["bleu"],
        "rouge1": rouge_score["rouge1"],
        "rouge2": rouge_score["rouge2"],
        "rougeL": rouge_score["rougeL"],
    }


training_args = TrainingArguments(
    output_dir="./code-gen-results",
    eval_strategy="steps",
    eval_steps=750,
    num_train_epochs=3,
    learning_rate=1e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
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
    # train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics
)

model.eval()

print('Evaluation on validation dataset')
print(trainer.evaluate())
print('Evaluation on test dataset')
print(trainer.evaluate(test_ds))
