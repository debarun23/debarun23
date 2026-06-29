# Debarun Das — AI Engineer

**Building intelligent systems from first principles.**

AI Engineer at Mind & Matter UK. I specialize in the full LLM pipeline — architecture design, pretraining, fine-tuning, and production deployment. Two models live on HuggingFace, both built and trained on a single consumer GPU.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-debarun--das-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://linkedin.com/in/debarun-das)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Debarun12-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/Debarun12)
[![Email](https://img.shields.io/badge/Gmail-debarundas237-EA4335?style=flat-square&logo=gmail&logoColor=white)](mailto:debarundas237@gmail.com)

---

## whoami

```python
class DebarunDas:
    role       = "AI Engineer @ Mind & Matter UK"
    location   = "Kolkata, India"
    focus      = ["LLM Architecture", "LoRA / QLoRA Fine-Tuning", "RAG Systems"]
    flagship   = "Built 95M-param LLM from scratch — beats GPT-3 Small (PPL 24.40 vs 26.0)"
    second     = "Fine-tuned Qwen2.5-3B via QLoRA on 8GB GPU — Val PPL 2.40, target was <10"
    philosophy = "Understand the internals. Build from scratch. Deploy with purpose."
    open_to    = "AI / ML Engineering roles — full-time or contract"
```

---

## Project 1 — ENG_llmV03

95M parameter transformer language model built entirely from scratch in PyTorch. No pretrained weights. No `from_pretrained()`.

```
Architecture   :  RoPE + SwiGLU Transformer  (same design as LLaMA / Mistral / Gemma)
Parameters     :  95M
Training Data  :  WikiText-103  (103M tokens)
Base PPL       :  24.40  —  beats GPT-3 Small (26.0) with 30% fewer parameters
Fine-tuned PPL :  20.83  (two-stage LoRA: R128 → merged → R64)
LoRA Params    :  ~1.6M trainable  (1.8% of base)
QA Dataset     :  355k clean pairs  (SciQ + ELI5 + FreebaseQA, cleaned from 405k raw)
Hardware       :  Single RTX 5050  —  8.5 GB VRAM
Status         :  Complete — deployed on HuggingFace
```

**What makes this different:**

- Rebuilt architecture V2 to V3 after diagnosing and fixing validation contamination (PPL spikes 43 to 73)
- Implemented RoPE, SwiGLU, Flash Attention, and LoRA from mathematical foundations — not copied from tutorials
- Hit catastrophic forgetting on full fine-tune (PPL rose from 24 to 35+), diagnosed root cause, switched to two-stage LoRA
- Engineered 355k clean QA pairs from 405k raw: regex deduplication, filler-phrase filtering, length enforcement
- Every epoch tracked. Every bug documented. Every architectural decision justified.

**V2 to V3 architecture upgrade:**

| | V2 | V3 |
|:--|:--:|:--:|
| Position Encoding | Absolute | RoPE |
| FFN Activation | GELU | SwiGLU |
| Transformer Layers | 8 | 12 |
| Parameters | 77M | 95M |
| Final PPL | ~28 | 20.83 |

Model: [huggingface.co/Debarun12/ENG-llmV03](https://huggingface.co/Debarun12/ENG-llmV03)
Code: [github.com/debarun23/LLM-from-scratch](https://github.com/debarun23/LLM-from-scratch)

---

## Project 2 — JavaExpert-Qwen2.5-3B

Domain-locked Java programming assistant, fine-tuned via QLoRA on a single 8GB consumer GPU. Answers Java questions accurately and refuses everything else — no external guardrails, refusal baked into training data.

```
Base Model     :  Qwen2.5-3B  (4-bit NF4 quantization via QLoRA)
Training VRAM  :  < 7 GB peak  (hard budget: 8 GB)
Inference VRAM :  < 2 GB  (merged model, no adapter dependency at runtime)
Val Perplexity :  2.40  —  target was < 10, beaten by 4x
Java Accuracy  :  8.5 / 10
Domain Refusal :  8.5 / 10  (refuses non-Java queries without post-processing)
QA Dataset     :  7,921 pairs generated from 42,000-line Java PDF corpus
Hardware       :  Single RTX 5050  —  8 GB VRAM
Status         :  Complete — deployed on HuggingFace
```

**VRAM optimization — every decision explicitly measured:**

| Component | Initial Config | Final Config | VRAM Saved |
|:--|:--|:--|:--:|
| LoRA rank | 32 across 7 modules | 16 on q_proj + v_proj only | -1.5 GB |
| Batch size | 4 | 1 + gradient accumulation x8 | -2.0 GB |
| Optimizer | AdamW | Adafactor | -1.0 GB |
| Compute dtype | fp16 | bf16 | Stable on Blackwell |

**Real bugs found and fixed — not tutorial problems:**

- SFTTrainer with pre-tokenized input inflated step count by 24x. Silent — no error, just wrong numbers. Fix: replaced with standard Trainer and explicit tokenization.
- PyTorch 2.6 changed `weights_only=True` default, breaking checkpoint resume with UnpicklingError. Fix: remove `rng_state.pth` before resuming.

Model: [huggingface.co/Debarun12/JavaExpert-Qwen2.5-3B](https://huggingface.co/Debarun12/JavaExpert-Qwen2.5-3B)
Code: [github.com/debarun23/JavaExpert-Qwen2.5-3B](https://github.com/debarun23/JavaExpert-Qwen2.5-3B)

---

## Other Projects

**Terratech — AI Agricultural Monitoring System**

IoT crop health monitoring with ESP32-CAM and multi-sensor integration. 95% plant disease detection accuracy using Edge Impulse and OpenCV. Automated irrigation reducing water consumption by 28.6%.

`ESP32` `OpenCV` `Edge Impulse` `IoT` `Computer Vision`
[github.com/debarun23/Terratech-Smart-Agriculture-with-IoT-Robotics-AI](https://github.com/debarun23/Terratech-Smart-Agriculture-with-IoT-Robotics-AI)

---

**Job Granite Guide — Agentic AI Interview Trainer**

IBM Granite + RAG pipeline for domain-specific interview preparation. Automated question generation with retrieval-augmented context. CI/CD deployed on IBM Cloud.

`IBM Granite` `RAG` `LangChain` `Agentic AI` `IBM Cloud`
[github.com/debarun23/job-granite-guide](https://github.com/debarun23/job-granite-guide)

---

**Git Repository Explainer — Intelligent Code Analyzer**

Analyzes any public GitHub repository in real time. 100% parsing accuracy via GitHub REST API. Auto-generates Mermaid.js architecture diagrams. Reduced developer onboarding time by 40%.

`React` `TailwindCSS` `OpenAI API` `GitHub API` `Mermaid.js`
[github.com/debarun23/Git-Repository-Explainer-App](https://github.com/debarun23/Git-Repository-Explainer-App)

---

## Technical Stack

**AI / ML**

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co)
[![PEFT](https://img.shields.io/badge/PEFT-FF6F00?style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/docs/peft)
[![TRL](https://img.shields.io/badge/TRL-7B2FBE?style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/docs/trl)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat-square&logo=langchain&logoColor=white)](https://langchain.com)
[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org)

Implemented from scratch: `Transformer` · `Multi-Head Attention` · `RoPE` · `SwiGLU` · `LoRA` · `QLoRA` · `Flash Attention` · `Cosine LR Scheduling` · `Gradient Accumulation`

Fine-tuning frameworks used: `PEFT` · `TRL` · `SFTTrainer` · `Adafactor` · `bitsandbytes`

**Infrastructure**

[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)
[![Git](https://img.shields.io/badge/Git-F05032?style=flat-square&logo=git&logoColor=white)](https://git-scm.com)
[![Linux](https://img.shields.io/badge/Linux-FCC624?style=flat-square&logo=linux&logoColor=black)](https://linux.org)

---

## Certifications

| Issuer | Certification |
|:--|:--|
| Hugging Face | The LLM Course — Fundamentals |
| IBM | Getting Started with Artificial Intelligence |
| Deloitte | Australia Technology Job Simulation |

---

## GitHub Stats

![GitHub Stats](https://github-readme-stats.vercel.app/api?username=debarun23&show_icons=true&theme=github_dark&hide_border=true&title_color=58A6FF&icon_color=58A6FF&text_color=c9d1d9&bg_color=0d1117)

![GitHub Streak](https://github-readme-streak-stats.herokuapp.com?user=debarun23&theme=github-dark-blue&hide_border=true&ring=58A6FF&fire=58A6FF&currStreakLabel=58A6FF)

---

*Building intelligence from first principles · Not `from_pretrained()`*

*Open to AI / ML Engineering roles · Kolkata, India · Remote-friendly*
