from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
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

tokenizer = AutoTokenizer.from_pretrained("microsoft/CodeGPT-small-py")
model = AutoModelForCausalLM.from_pretrained("microsoft/CodeGPT-small-py")

dataset = load_dataset("eth_py150_open")

train_dataset = dataset["train"]
test_dataset = dataset["test"]
validation_dataset = dataset["validation"]
