from datasets import load_dataset
from transformers import (
    T5ForConditionalGeneration,
    RobertaTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
import numpy as np
from evaluate import load
import torch
from sklearn.model_selection import train_test_split
from huggingface_hub import login

login('')
cuda = torch.cuda.is_available()
mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

model_name = "Salesforce/codet5-small"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = RobertaTokenizer.from_pretrained(model_name)

dataset = load_dataset("mbpp")

train_dataset = dataset["train"]
test_dataset = dataset["test"]
validation_dataset = dataset["validation"]


def clean_text(text):
    return text.strip().replace("\n", " ").replace("\r", " ")


def preprocess_function(examples):
    inputs = [clean_text(str(code)) for code in examples["code"]]
    targets = [clean_text(str(text)) for text in examples["text"]]

    inputs = ["summarize: " + inp for inp in inputs]

    model_inputs = tokenizer(
        inputs,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    labels = tokenizer(
        targets,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    model_inputs["labels"] = labels["input_ids"]

    labels = labels["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels

    return model_inputs


train_tokenized = train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=train_dataset.column_names
)

test_tokenized = test_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=test_dataset.column_names
)

validation_tokenized = validation_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=validation_dataset.column_names
)

rouge = load("rouge")
bleu = load("bleu")


def compute_metrics(eval_preds):
    predictions, labels = eval_preds
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    rouge_result = rouge.compute(predictions=decoded_preds, references=decoded_labels)

    bleu_result = bleu.compute(predictions=decoded_preds, references=[[label] for label in decoded_labels])

    return {
        "rouge1": rouge_result["rouge1"],
        "rouge2": rouge_result["rouge2"],
        "rougeL": rouge_result["rougeL"],
        "bleu": bleu_result["bleu"]
    }


training_args = Seq2SeqTrainingArguments(
    output_dir="./codet5-finetuned-mbpp",
    evaluation_strategy="steps",
    eval_steps=100,
    learning_rate=2e-5,
    per_device_train_batch_size=4 if not cuda else 8,
    per_device_eval_batch_size=4 if not cuda else 8,
    num_train_epochs=5,
    weight_decay=0.01,
    save_total_limit=3,
    predict_with_generate=True,
    logging_dir="./logs",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="bleu",
    greater_is_better=True,
    fp16=cuda,
    gradient_accumulation_steps=2,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=validation_tokenized,
    # tokenizer=tokenizer,
    processing_class=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
    compute_metrics=compute_metrics,
)

trainer.train()

model.save_pretrained("./codet5-finetuned-mbpp-final")
tokenizer.save_pretrained("./codet5-finetuned-mbpp-final")

test_results = trainer.evaluate(test_tokenized)
print("\nTest Results:", test_results)


def generate_summary(code):
    inputs = tokenizer("summarize: " + code, return_tensors="pt", max_length=512, truncation=True)
    device = "cuda" if cuda else "mps" if mps else "cpu"
    summary_ids = model.generate(
        inputs["input_ids"].to(device),
        max_length=128,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True,
        no_repeat_ngram_size=2,
        top_p=0.95,
        temperature=0.7
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


prompt_dataset = dataset["prompt"]
for idx in range(min(3, len(prompt_dataset))):
    example = prompt_dataset[idx]
    print(f"\nExample {idx + 1}:")
    print("Original Code:", example["code"])
    print("Original Description:", example["text"])
    print("Generated Summary:", generate_summary(example["code"]))