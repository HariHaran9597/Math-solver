# 🧮 AI Math Solver (Fine-Tuned Qwen2.5-1.5B)

![Status](https://img.shields.io/badge/Status-Live-success)
![Accuracy](https://img.shields.io/badge/Accuracy-82%25-brightgreen)
![Model](https://img.shields.io/badge/Model-Qwen2.5--1.5B-blue)
![Tech](https://img.shields.io/badge/Tech-Unsloth%20%7C%20LoRA-orange)

A specialized **Small Language Model (SLM)** pipeline capable of solving grade-school math word problems with step-by-step reasoning.  
Trained by fine-tuning **Qwen2.5-Math-1.5B** on the **GSM8K** dataset using **Unsloth** + **QLoRA**.

It also features an inference-time **Smart Mode** using Majority Voting (Self-Consistency), boosting accuracy significantly without additional training.

---

## 🔗 Live Demos

| Platform | Type | Speed | Link |
|---------|------|--------|------|
| **Hugging Face** | Web App | Slow (CPU) | [➡️ Try the Chat App](https://huggingface.co/spaces/justhariharn/Math-Solver-Portfolio) |
| **Google Colab** | Notebook | **Fast (GPU)** | [➡️ Run on Free GPU](https://colab.research.google.com/github/HariHaran9597/Math-solver/blob/main/Try_Math_Solver.ipynb) |

---

## 📊 Performance & Metrics

Pipeline engineered to push the limits of a tiny 1.5B model.  
Using **Self-Consistency (Majority Voting)** yields performance comparable to much larger models.

| Method | Accuracy (GSM8K Sample) | Notes |
|--------|--------------------------|-------|
| **Base Model (Zero-Shot)** | ~45% | Struggled with formatting & reasoning consistency. |
| **Fine-Tuned (Greedy)** | 70% | Stable step-by-step logic, fewer hallucinations. |
| **Fine-Tuned + Voting** | **82%** 🏆 | Production mode; generates 3 answers and votes. |

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
pip install torch transformers accelerate peft unsloth

```
## Python Inference Script
```bash
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
    inputs = tokenizer([text], return_tensors="pt").to("cuda")
    
    outputs = model.generate(**inputs, max_new_tokens=512)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print(solve("If I have 30 apples and eat 2, how many do I have left?"))

```
## Project Structure
```bash
├── app.py                     # Gradio app (UI + inference logic)
├── requirements.txt           # Dependencies for deployment
├── Try_Math_Solver.ipynb      # Clean inference notebook for users
└── Math_Solver_Training.ipynb # Full Unsloth + LoRA training pipeline
```
###🔮 Future Improvements

RAG Integration: Query textbook examples for hard problems
Vision Support: Use Qwen-VL to solve image-based math questions
Model Optimization: Convert to GGUF for llama.cpp CPU inference

Built with ❤️ by Hariharan

