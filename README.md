# 🧮 AI Math Solver (Fine-Tuned Qwen2.5-1.5B)

![Status](https://img.shields.io/badge/Status-Public%20research%20demo-blue)
![Model](https://img.shields.io/badge/Model-Qwen2.5--1.5B-blue)
![Tech](https://img.shields.io/badge/Tech-Unsloth%20%7C%20LoRA-orange)

A specialized **Small Language Model (SLM)** pipeline capable of solving grade-school math word problems with step-by-step reasoning.  
Trained by fine-tuning **Qwen2.5-Math-1.5B** on the **GSM8K** dataset using **Unsloth** + **QLoRA**.

It also includes an experimental inference-time mode that compares three sampled answers and selects the most common explicitly marked final result. Agreement can still be wrong.

---

## 🔗 Live Demos

| Platform | Type | Speed | Link |
|---------|------|--------|------|
| **Hugging Face** | Web App (may sleep when idle) | CPU | [➡️ Try the Chat App](https://huggingface.co/spaces/justhariharan/Math-Solver) |
| **Google Colab** | Notebook | **Fast (GPU)** | [➡️ Run on Free GPU](https://colab.research.google.com/github/HariHaran9597/Math-solver/blob/main/Try_Math_Solver.ipynb) |

---

## 📊 Performance & Metrics

The repository does not currently include a versioned evaluation set with per-question
predictions sufficient to reproduce an accuracy percentage. No numerical accuracy claim is
made here. Training used GSM8K; this is not evidence of performance on the full GSM8K test set
or equivalence to larger models. Self-consistency compares three sampled outputs and selects
the most common explicitly marked final answer. Agreement is not a correctness guarantee.

---

## 🛠️ Technical Stack

- **Model:** Qwen2.5-Math-1.5B-Instruct  
- **Training:** Unsloth + PyTorch  
- **Fine-Tuning:** QLoRA (rank=16, alpha=16)  
- **Dataset:** GSM8K  
- **Inference:** Hugging Face Transformers  
- **Frontend:** Gradio  

---

## 🚀 Key Features

### **1. Teacher Persona (NLP Prompting)**  
Model is instructed to *teach* instead of just outputting the answer.

> System Prompt: "You are a patient and friendly math teacher. Explain the logic simply so a student can understand."

---

### **2. Three-attempt comparison (experimental)**
During inference:

- **Standard Mode:** 1 answer (fast)  
- **Comparison mode:** generates up to **3 reasoning attempts** (temperature=0.8), extracts
  only answers marked with `####`, and returns the most common marked answer. If no attempt
  provides a marked answer, the app says so instead of guessing from an arbitrary number in
  the reasoning. This mode is slower and can still be wrong.

---

## 💻 Run Locally

### **1. Install Dependencies**

```bash
pip install torch transformers accelerate peft

```
## Python Inference Script
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "justhariharan/Qwen2.5-Math-1.5B-Solver"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

def solve(question):
    messages = [
        {"role": "system", "content": "You are a helpful math tutor. Solve step-by-step."},
        {"role": "user", "content": question}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    outputs = model.generate(**inputs, max_new_tokens=512)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print(solve("If I have 30 apples and eat 2, how many do I have left?"))

```

## Project Structure
```bash
├── README.md
├── Try_Math_Solver.ipynb       # Inference notebook
└── MATH_agent.ipynb            # Training / exploration notebook
```

The hosted application's files live in the [Hugging Face Space repository](https://huggingface.co/spaces/justhariharan/Math-Solver/tree/main), separately from these notebooks.

### 🔮 Future Improvements

RAG Integration: Query textbook examples for hard problems
Vision Support: Use Qwen-VL to solve image-based math questions
Model Optimization: Convert to GGUF for llama.cpp CPU inference

Built with ❤️ by Hariharan

