from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import numpy as np
import torch
from huggingface_hub import login

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
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
model.resize_token_embeddings(len(tokenizer))  # Resize for padding token
model = model.to(device)


ds = load_dataset("Fraol/Py150-processed")
train_ds = ds["train"].select(range(60000))
val_ds = ds["val"].select(range(7500))
test_ds = ds["test"].select(range(7500))


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  
)


print("Preparing datasets...")
train_ds = train_ds.map(
    lambda x: prepare_data(x, tokenizer),
    batched=True,
    remove_columns=train_ds.column_names,
    num_proc=4  
)
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


training_args = TrainingArguments(
    output_dir="./code-gen-results",
    evaluation_strategy="steps",
    eval_steps=500,
    num_train_epochs=3,
    learning_rate=5e-5,  
    per_device_train_batch_size=4,  
    per_device_eval_batch_size=4,
    save_total_limit=3,
    warmup_steps=1000,
    save_steps=500,
    fp16=False, 
    gradient_accumulation_steps=8,  
    load_best_model_at_end=True,
    max_grad_norm=1.0,  
    logging_steps=100,
    metric_for_best_model="eval_loss",
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=data_collator
)


if torch.cuda.is_available():
    torch.cuda.empty_cache()


try:
    print("Starting training...")
    trainer.train()
    trainer.save_model("./code-token-distilgpt-final")
    print("Evaluating final model...")
    eval_results = trainer.evaluate(test_ds)
    print(f"Evaluation results: {eval_results}")
except RuntimeError as e:
    print(f"Training error occurred: {str(e)}")
    if "CUDA out of memory" in str(e):
        print("Suggestion: Try reducing batch size or model size")
    raise