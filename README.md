# DocuCode

**DocuCode: Code Summarization and Generation evaluation**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)  
[![Paper](https://img.shields.io/badge/Paper-PDF-blue)](CSE_847_Project_Report.pdf)

---

## Overview

CodeSumGen is a research project exploring the fine-tuning of large language models (LLMs) for two core tasks:
- **Code Summarization**: Generating natural language descriptions from source code.
- **Code Generation**: Predicting code continuations from partial code snippets.

We evaluate transformer-based models such as **CodeT5**, **CodeT5+**, and **DistilGPT2** on benchmark datasets including **MBPP**, **CodeSearchNet (CodeXGLUE)**, and **Py150**.

---

## 🔍 Motivation

As codebases grow in size and complexity, tools that summarize or generate code can assist with software comprehension, documentation, and development automation. We assess the impact of transfer learning and model distillation on the performance of code-focused tasks.

---

## 📦 Datasets Used

- **MBPP**: Mostly Basic Python Programming Problems  
- **CodeXGLUE (CodeSearchNet)**: Benchmark for code summarization  
- **Py150**: 150k Python files for code completion and repair

---

## 📊 Tasks and Models

| Task                | Model        | Dataset             |
|---------------------|--------------|---------------------|
| Code Summarization  | CodeT5       | MBPP, CodeXGLUE     |
| Code Summarization  | CodeT5+      | CodeXGLUE           |
| Code Generation     | DistilGPT2   | Py150               |

---

## 🔧 Methodology

### 🧠 Code Summarization
- **Models**: CodeT5, CodeT5+  
- **Technique**: Fine-tuned using Hugging Face `Trainer` API  
- **Preprocessing**:
  - Prefixing code with `"summarize:"`
  - Byte Pair Encoding (BPE) tokenization
  - Input max length: 512
  - Output max length: 128  
- **Evaluation**: ROUGE-1/2/L, BLEU

### 🔨 Code Generation
- **Model**: DistilGPT2 (lightweight version of GPT-2 via knowledge distillation)  
- **Training Setup**:
  - Input max length: 512
  - 3 epochs, batch size 4
  - Token-level prediction using causal LM  
- **Evaluation**: BLEU, ROUGE‑1, CodeBLEU, Levenshtein Distance, pass@k

---

## 📈 Results

### Code Summarization (MBPP, CodeT5)

| Metric     | Score |
|------------|-------|
| ROUGE-1    | 0.72  |
| ROUGE-2    | 0.55  |
| ROUGE-L    | 0.70  |
| BLEU       | 0.46  |

### Code Summarization (CodeXGLUE)

| Model   | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU  |
|---------|---------|---------|---------|-------|
| CodeT5  | 0.474   | 0.192   | 0.419   | 0.251 |
| CodeT5+ | 0.488   | 0.206   | 0.439   | 0.266 |

### Code Generation (DistilGPT2)

| Tokens | Accuracy | BLEU   | CodeBLEU | Levenshtein |
|--------|----------|--------|----------|-------------|
| 1      | 70.6%    | 0.7716 | 0.4668   | 1.1479      |
| 3      | 48.1%    | 0.4902 | 0.2964   | 3.9061      |
| 5      | 36.7%    | 0.4146 | 0.3472   | 7.0340      |

| k (Beam Size) | pass@k |
|---------------|--------|
| 1             | 0.377  |
| 3             | 0.443  |
| 5             | 0.470  |

---


## 🚀 Future Work

- Experiment with larger and more diverse code datasets  
- Extend to more programming languages  
- Evaluate with additional metrics (e.g., METEOR, CIDEr)  
- Explore line-level code completion and ensemble models  
- Apply interpretability techniques to better understand model behavior

---

## 📄 Citation

If you use this work, please cite:

```bibtex
@misc{haroon2024codesumgen,
  author = {Muhammad Haroon, Michael Ewnetu, Matthew DiRisio},
  title = {Code Summarization and Generation with Large Language Models},
  year = {2024},
  note = {Course project at Michigan State University},
  url = {https://github.com/haroon001/CodeSumGen}
}

