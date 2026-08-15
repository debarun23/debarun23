# Debarun Das

**I build language models from first principles, not from `from_pretrained()`.**

I specialize in the full LLM pipeline: architecture design, pretraining, fine-tuning, and low-resource deployment. Everything below was built and trained on a single consumer GPU (8-8.5GB VRAM), no cloud clusters, no borrowed compute.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-debarun--das-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://linkedin.com/in/debarun-das)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Debarun12-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/Debarun12)
[![Email](https://img.shields.io/badge/Gmail-debarundas237-EA4335?style=flat-square&logo=gmail&logoColor=white)](mailto:debarundas237@gmail.com)

---

## whoami

```python
class DebarunDas:
    focus      = ["LLM Architecture", "LoRA / QLoRA Fine-Tuning", "RAG Systems", "Agentic Workflows"]
    flagship   = "Built a 95M-param transformer from scratch — beats GPT-3 Small (PPL 24.40 vs 26.0)"
    second     = "Fine-tuned Qwen2.5-3B via QLoRA on 8GB VRAM — Val PPL 2.40, target was <10"
    constraint = "Everything trained on a single consumer GPU. No excuses about compute."
    philosophy = "Understand the internals. Build from scratch. Ship something that holds up."
    open_to    = "AI / ML Engineering roles — full-time or contract"
```

---

## What I actually know how to do

This isn't a list of libraries I've imported once. Everything below, I've implemented, debugged, and can explain from first principles:

- **Transformer internals** — Multi-Head Attention, RoPE, SwiGLU, Flash Attention, causal self-attention, weight tying, built by hand in PyTorch, not copied from a tutorial
- **LLM fine-tuning at the edge of consumer hardware** — LoRA, QLoRA, 4-bit NF4 quantization, gradient accumulation, completion-only loss masking, squeezing real fine-tuning jobs into 8GB of VRAM
- **RAG systems that don't hallucinate** — semantic chunking, embedding retrieval, confidence-scored routing between verified data and model generation, domain refusal layers with zero external guardrails
- **Debugging failures nobody documents** — silent step-count inflation bugs, checkpoint incompatibility across PyTorch versions, validation contamination, catastrophic forgetting, the kind of bugs that don't show up in a blog post

---

## Project 1 — TinyTalk-124M

A conversational language model built through a complete first-principles pipeline: a 95M-parameter transformer pretrained from scratch, then adapted into a chat model via two independent, staged LoRA fine-tuning passes. No pretrained weights used at any stage.

```
Base Model     :  ScratchLM-95M (88.7M measured params) — RoPE + SwiGLU transformer, 12 layers
Pretraining    :  WikiText-103, converged epoch 17, validation PPL 24.40 (beats GPT-3 Small's 26.0)
Stage A        :  Instruction adaptation — LoRA r=32, alpha=64, target: qkv_proj + out_proj
Stage B        :  Conversational refinement — independent LoRA adapter on the re-merged Stage A base
Trainable      :  1,179,648 params (1.02% of 115.57M) per stage
Final Val PPL  :  25.83 (monotonic decrease, plateaus epoch 4)
Hardware       :  Single RTX 5050 — 8.5 GB VRAM, mixed precision (AMP)
Status         :  Complete — deployed on HuggingFace
```

**What makes this different from a single-pass fine-tune:**

- Two-stage architecture instead of one fine-tuning pass: Stage A adapts the frozen backbone to the instruction/response format, Stage B loads that re-merged checkpoint as a new frozen base and trains an independently zero-initialized adapter for multi-turn conversation, keeping each stage's learning signal scoped to the capability it targets
- Every merge step verified against a live forward pass: 148 total keys merged, 0 missing, 0 unexpected, both for Stage A and Stage B
- Label-shift sanity check run before training to confirm next-token targets are correctly offset by exactly one position (`labels[0][:9] == input_ids[0][1:10]`) — the kind of silent bug that otherwise trains a model on the wrong task entirely
- `lora_B` zero-initialized so the adapted model is numerically identical to the frozen base before any gradient step, `lora_A` uses small-scale noise (or `kaiming_uniform_` in the Stage A refinement) so the first backward pass still has a well-conditioned gradient
- Full training curve logged step-by-step, not just a final number: perplexity tracked from 41.06 at step 1400 down to 25.83 by step 10600

**Real training curve (logged):**

| Step | Epoch | Train Loss | Val Loss | Val PPL |
|:--|:--:|:--:|:--:|:--:|
| 1400 | 0 | 3.59 | 3.72 | 41.06 |
| 3400 | 1 | 3.82 | 3.47 | 32.23 |
| 5600 | 2 | 3.26 | 3.34 | 28.16 |
| 7600 | 3 | 2.95 | 3.27 | 26.44 |
| 9800 | 4 | 3.52 | 3.25 | 25.82 |
| 10600 | 4 | 3.68 | 3.25 | 25.83 |

**Real, unedited model output** (greedy decoding, repetition penalty 1.3):

> **Prompt:** How can I improve my relationship with my family?
> **Response:** One way to improve your relationships with your family is by developing a strong bond. This will help you build trust and loyalty, which in turn helps you feel more connected to them.

Model: [huggingface.co/Debarun12/tinytalk-124m](https://huggingface.co/Debarun12/tinytalk-124m)
Code: [github.com/debarun23/tinytalk-114m](https://github.com/debarun23/tinytalk-114m)
Base model: [github.com/debarun23/LLM-from-scratch](https://github.com/debarun23/LLM-from-scratch)

---

## Project 2 — JavaExpert-Qwen2.5-3B

A domain-locked Java programming assistant, fine-tuned via QLoRA on a single 8GB consumer GPU. Answers Java questions accurately and refuses everything else, no external guardrails, refusal baked directly into training.

```
Base Model     :  Qwen2.5-3B  (4-bit NF4 quantization via QLoRA)
Training VRAM  :  < 7 GB peak  (hard budget: 8 GB)
Inference VRAM :  < 2 GB  (merged model, no adapter dependency at runtime)
Val Perplexity :  2.40  —  target was < 10, beaten by 4x
Java Accuracy  :  8.5 / 10
Domain Refusal :  8.5 / 10  (refuses non-Java queries without post-processing)
QA Dataset     :  7,921 pairs generated from a 42,000-line Java PDF corpus
Hardware       :  Single RTX 5050  —  8 GB VRAM
Status         :  Complete — deployed on HuggingFace
```

**VRAM optimization, every decision explicitly measured:**

| Component | Initial Config | Final Config | VRAM Saved |
|:--|:--|:--|:--:|
| LoRA rank | 32 across 7 modules | 16 on q_proj + v_proj only | -1.5 GB |
| Batch size | 4 | 1 + gradient accumulation x8 | -2.0 GB |
| Optimizer | AdamW | Adafactor | -1.0 GB |
| Compute dtype | fp16 | bf16 | Stable on Blackwell |

**Real bugs found and fixed, not tutorial problems:**

- SFTTrainer with pre-tokenized input silently inflated step count by 24x. No error thrown, just wrong numbers. Fixed by replacing it with a standard Trainer and explicit tokenization.
- PyTorch 2.6 changed the `weights_only=True` default, breaking checkpoint resume with an UnpicklingError. Fixed by removing `rng_state.pth` before resuming.

Model: [huggingface.co/Debarun12/JavaExpert-Qwen2.5-3B](https://huggingface.co/Debarun12/JavaExpert-Qwen2.5-3B)
Code: [github.com/debarun23/JavaExpert-Qwen2.5-3B](https://github.com/debarun23/JavaExpert-Qwen2.5-3B)

---

## Project 3 — Domain-Locked RAG Assistants

Two production-style RAG systems built end to end, from zero-dataset starting points to deployed, confidence-scored retrieval pipelines.

```
Pipeline       :  Web scraping → multi-pass corpus cleaning → LLM-assisted QA generation
                  → LoRA fine-tuning → semantic retrieval → confidence-based routing
Retrieval      :  FAISS vector store, cosine similarity, three-band confidence scoring
Deployment     :  FastAPI backend, streaming inference, Dockerized
Reliability    :  Domain refusal layer — 100% out-of-scope rejection accuracy
```

**What makes this different:** most RAG demos blend retrieval and generation into one blurry answer. These systems explicitly route between three confidence bands, high-confidence queries return verified ground truth, low-confidence queries fall back to generation, and everything in between is flagged rather than guessed at.

Code: [github.com/debarun23/aws-enterprise-ai-assistant](https://github.com/debarun23/aws-enterprise-ai-assistant)
Model: [huggingface.co/Debarun12/aws-enterprise-assistant-qwen2.5-7b-qlora](https://huggingface.co/Debarun12/aws-enterprise-assistant-qwen2.5-7b-qlora)

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

Analyzes any public GitHub repository in real time. 100% parsing accuracy via the GitHub REST API. Auto-generates Mermaid.js architecture diagrams. Reduced developer onboarding time by 40%.

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

**Implemented from scratch:** `Transformer` · `Multi-Head Attention` · `RoPE` · `SwiGLU` · `LoRA` · `QLoRA` · `Flash Attention` · `Cosine LR Scheduling` · `Gradient Accumulation`

**Fine-tuning frameworks:** `PEFT` · `TRL` · `SFTTrainer` · `Adafactor` · `bitsandbytes`

**Retrieval & Agents:** `FAISS` · `Vector Databases` · `Sentence-Transformers` · `LangChain` · `LangGraph`

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

*Building intelligence from first principles. Not `from_pretrained()`.*
