# 🧮 AI Math Solver (Fine-Tuned Qwen2.5-1.5B)

![Status](https://img.shields.io/badge/Status-Research%20prototype-blue)
![Model](https://img.shields.io/badge/Model-Qwen2.5--1.5B-blue)
![Tech](https://img.shields.io/badge/Tech-Unsloth%20%7C%20LoRA-orange)

A specialized **Small Language Model (SLM)** pipeline capable of solving grade-school math word problems with step-by-step reasoning.  
Trained by fine-tuning **Qwen2.5-Math-1.5B** on the **GSM8K** dataset using **Unsloth** + **QLoRA**.

It also features an inference-time **Smart Mode** using Majority Voting (Self-Consistency), boosting accuracy significantly without additional training.

---

## 🔗 Live Demos

| Platform | Type | Speed | Link |
|---------|------|--------|------|
| **Hugging Face** | Web App (may sleep when idle) | CPU | [➡️ Try the Chat App](https://huggingface.co/spaces/justhariharan/Math-Solver) |
| **Google Colab** | Notebook | **Fast (GPU)** | [➡️ Run on Free GPU](https://colab.research.google.com/github/HariHaran9597/Math-solver/blob/main/Try_Math_Solver.ipynb) |

---

## 📊 Performance & Metrics

These are previously reported sample results. The repository does not currently provide
the sample size, per-question predictions, or a versioned evaluation report sufficient to
reproduce these percentages. Treat them as provisional, not as full GSM8K test-set scores
or evidence of equivalence to larger models. The training dataset size is not the evaluation denominator.

| Method | Accuracy (GSM8K Sample) | Notes |
|--------|--------------------------|-------|
| **Base Model (Zero-Shot)** | ~45% | Struggled with formatting & reasoning consistency. |
| **Fine-Tuned (Greedy)** | 70% | Stable step-by-step logic, fewer hallucinations. |
| **Fine-Tuned + Voting** | **82%** | Experimental mode; generates 3 answers and votes. |

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

### **2. Smart Mode (Self-Consistency Voting)**  
During inference:

- **Standard Mode:** 1 answer (fast)  
- **Smart Mode:**  
  - Generates **3 different reasoning paths** (temp=0.8)  
  - Extracts final answers  
  - Performs **Majority Vote**  
  - Output becomes far more reliable  

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

