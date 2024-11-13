from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import numpy as np
from evaluate import load
import json
from collections import defaultdict
from huggingface_hub import login
from transformers.models.marian.convert_marian_to_pytorch import add_special_tokens_to_vocab


def load_model_and_data():
    login('')
    model = AutoModelForCausalLM.from_pretrained("./code-token-gen-final")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/CodeGPT-small-py")

    ds = load_dataset("google/code_x_glue_cc_code_completion_token", "python")
    test_ds = ds["test"]

    return model, tokenizer, test_ds


def prepare_input_and_target(example, context_length=None):
    code_sequence = example["code"]

    if context_length is None:
        context_length = int(len(code_sequence) * 0.8)

    input_sequence = code_sequence[:context_length]
    target_sequence = code_sequence[context_length:]

    return input_sequence, target_sequence

def generate_completion(model, tokenizer, input_seq, max_new_tokens=50):
    input_text = " ".join(input_seq)
    inputs = tokenizer(input_text, return_tensors="pt", add_special_tokens=False)

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs,
                                 max_new_tokens=max_new_tokens,
                                 temperature=0.7,
                                 top_p=0.95,
                                 do_sample=True)
    completion_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    completion = tokenizer.decode(completion_tokens, skip_special_tokens=True)

    return completion.split()


def compute_metrics(generated, target):
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
        "rouge": rouge_score,
        "bleu": bleu_score['bleu'],
        "accuracy": accuracy
    }


def evaluate(model, tokenizer, test_ds, context_length, num_samples=None):
    model.eval()
    model.to(model.device)

    all_metrics=defaultdict(list)

    if num_samples:
        test_ds = test_ds.select(range(num_samples))

    for example in tqdm(test_ds, desc="Evaluating"):
        input_seq, target_seq = prepare_input_and_target(example, context_length)
        generated = generate_completion(model, tokenizer, input_seq)

        metrics = compute_metrics(generated, target_seq)
        for metric, value in metrics.items():
            all_metrics[metric].append(value)

    average_metrics = {metric: np.mean(values) for metric, values in all_metrics.items()}

    return average_metrics

def main():
    model, tokenizer, test_ds = load_model_and_data()

    print('Running evaluation')
    metrics = evaluate(model, tokenizer, test_ds, context_length=10, num_samples=100)

    print("Done. Results:")
    print('-'*50)
    for metric, value in metrics.items():
        print(f"{metric}: {value}")

    with open("evaluation_results.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
