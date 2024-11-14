import json
from collections import defaultdict

import numpy as np
import torch
from datasets import load_dataset
from evaluate import load
from huggingface_hub import login
from responses import target
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model_and_data():
    login('')
    model = AutoModelForCausalLM.from_pretrained("./code-token-gen-final")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/CodeGPT-small-py")

    ds = load_dataset("google/code_x_glue_cc_code_completion_token", "python")
    test_ds = ds["test"]

    return model, tokenizer, test_ds


def prepare_input_and_target(example, target_size=1):
    code_sequence = example["code"]
    code_sequence.pop()
    return code_sequence[:-target_size], code_sequence[-target_size:]

def generate_completion(model, tokenizer, input_seq, max_new_tokens=50):
    input_text = " ".join(input_seq)
    inputs = tokenizer(input_text, return_tensors="pt", add_special_tokens=False, max_length=512, truncation=True)

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs,
                                 max_new_tokens=max_new_tokens,
                                 temperature=0.8,
                                 # top_p=0.95,
                                 do_sample=True,
                                 no_repeat_ngram_size=2,
                                 )
    completion_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    completion = tokenizer.decode(completion_tokens, skip_special_tokens=True)

    return completion.split()


def compute_metrics(generated, target):
    if not generated:
        return {
            "rouge1": 0,
            "rouge2": 0,
            "rougeL": 0,
            "rougeLsum": 0,
            "bleu": 0,
            "accuracy": 0
        }
    rouge = load("rouge")
    bleu = load("bleu")

    generated_str = " ".join(generated)
    target_str = " ".join(target)

    rouge_score = rouge.compute(predictions=[generated_str], references=[[target_str]])

    bleu_score = bleu.compute(predictions=[generated_str], references=[[target_str]])

    min_len = min(len(generated), len(target))
    matching = sum(1 for i in range(min_len) if generated[i] == target[i])
    accuracy = matching / len(target) if target else 0

    return {
        "rouge1": rouge_score["rouge1"],
        "rouge2": rouge_score["rouge2"],
        "rougeL": rouge_score["rougeL"],
        "rougeLsum": rouge_score["rougeLsum"],
        "bleu": bleu_score['bleu'],
        "accuracy": accuracy
    }


def evaluate(model, tokenizer, test_ds, target_size=None, num_samples=None):
    model.eval()
    model.to(model.device)

    all_metrics = defaultdict(list)

    if num_samples:
        test_ds = test_ds.select(range(num_samples))

    for example in tqdm(test_ds, desc="Evaluating"):
        input_seq, target_seq = prepare_input_and_target(example, target_size)
        generated = generate_completion(model, tokenizer, input_seq, len(target_seq))
        metrics = compute_metrics(generated, target_seq)
        for metric, value in metrics.items():
            all_metrics[metric].append(value)

    average_metrics = {metric: np.mean(values) for metric, values in all_metrics.items()}

    return average_metrics

def main():
    model, tokenizer, test_ds = load_model_and_data()

    print('Running evaluation')
    metrics = evaluate(model, tokenizer, test_ds, target_size=1, num_samples=100)

    print("Done. Results:")
    print('-'*50)
    for metric, value in metrics.items():
        print(f"{metric}: {value}")

    with open("evaluation_results.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
