# Debarun Das

### LLM Engineer | AI Developer | Building Intelligence from First Principles

[![Portfolio](https://img.shields.io/badge/Portfolio-debarun--s--portfolio.vercel.app-000000?style=flat-square\&logo=vercel\&logoColor=white)](https://debarun-s-portfolio.vercel.app/)
[![GitHub](https://img.shields.io/badge/GitHub-debarun23-181717?style=flat-square\&logo=github\&logoColor=white)](https://github.com/debarun23)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Debarun12-FFD21E?style=flat-square\&logo=huggingface\&logoColor=black)](https://huggingface.co/Debarun12)

---

## About

I build LLMs—not just use them.

My work focuses on training, fine-tuning, evaluating, and deploying language models under real-world hardware constraints. I enjoy working across the entire stack: dataset engineering, transformer architectures, low-resource training, retrieval systems, and production deployment.

Every project includes documented engineering decisions, debugging notes, evaluation results, and known limitations.

**Current Focus:** LLM Fine-Tuning · RAG Systems · Agentic AI · Low-Resource ML

---

# Core Projects

## ENG_llmV03 — 95M Parameter LLM Built From Scratch

> No `from_pretrained()`. Built from mathematical foundations.

| Metric          | Value                      |
| --------------- | -------------------------- |
| Architecture    | RoPE + SwiGLU Transformer  |
| Parameters      | 95M                        |
| Training Data   | WikiText-103 (103M tokens) |
| Base PPL        | 24.40                      |
| Fine-Tuned PPL  | 20.83                      |
| LoRA Parameters | ~1.6M                      |
| QA Dataset      | 355K Clean QA Pairs        |
| Hardware        | RTX 5050 (8.5GB VRAM)      |

### Highlights

* Implemented Transformer architecture from scratch
* Implemented RoPE positional encoding
* Implemented SwiGLU feed-forward blocks
* Diagnosed and fixed validation contamination
* Rebuilt model pipeline from V2 → V3
* Solved catastrophic forgetting using staged LoRA fine-tuning
* Created and cleaned 355K QA examples from 405K raw samples

### Architecture Evolution

| Feature           | V2       | V3     |
| ----------------- | -------- | ------ |
| Position Encoding | Absolute | RoPE   |
| Activation        | GELU     | SwiGLU |
| Layers            | 8        | 12     |
| Parameters        | 77M      | 95M    |
| Final PPL         | ~28      | 20.83  |

**Links**

* GitHub: https://github.com/debarun23/LLM-from-scratch
* Hugging Face: https://huggingface.co/Debarun12

---

## JavaExpert-Qwen2.5-3B — Domain-Locked Java QA via QLoRA

> Fine-tuned Qwen2.5-3B on a single 8GB GPU.

| Metric                | Value               |
| --------------------- | ------------------- |
| Base Model            | Qwen2.5-3B-Instruct |
| Method                | QLoRA               |
| Dataset               | 7,921 QA Pairs      |
| Validation Perplexity | 2.40                |
| Peak Training VRAM    | < 7 GB              |
| Inference VRAM        | < 2 GB              |
| Java Correctness      | 8.5 / 10            |
| Domain Restriction    | 8.5 / 10            |
| Hardware              | RTX 5050 (8GB VRAM) |

### Engineering Decisions

| Component              | Initial             | Final                | VRAM Saved |
| ---------------------- | ------------------- | -------------------- | ---------- |
| LoRA Rank              | 32 across 7 modules | 16 on q_proj, v_proj | −1.5 GB    |
| Batch Size             | 4                   | 1 + Grad Accum ×8    | −2.0 GB    |
| Optimizer              | AdamW               | Adafactor            | −1.0 GB    |
| Gradient Checkpointing | Enabled             | Disabled             | −1.0 GB    |

### Bugs Found and Fixed

* SFTTrainer + packing caused 24× step inflation
* PyTorch 2.6 checkpoint incompatibility due to `weights_only=True`
* Fixed checkpoint resumption pipeline

**Links**

* GitHub: https://github.com/debarun23/java-expert-qlora
* Model: https://huggingface.co/Debarun12/JavaExpert-Qwen2.5-3B

---

## Terratech — AI Agricultural Monitoring System

AI-powered crop monitoring and automated irrigation platform.

### Results

* 95% disease detection accuracy
* 28.6% reduction in water consumption
* Real-time sensor monitoring and actuation

### Technologies

`ESP32` `OpenCV` `Edge Impulse` `IoT` `Computer Vision`

**GitHub**

https://github.com/debarun23/Terratech-Smart-Agriculture-with-IoT-Robotics-AI

---

## Job Granite Guide — Agentic AI Interview Trainer

Interview preparation assistant built using IBM Granite and Retrieval-Augmented Generation.

### Features

* Domain-specific interview preparation
* RAG-powered contextual responses
* CI/CD deployment on IBM Cloud

### Technologies

`IBM Granite` `RAG` `LangChain` `Agentic AI`

**GitHub**

https://github.com/debarun23/job-granite-guide

---

## Git Repository Explainer

Analyzes public GitHub repositories and automatically generates architecture explanations.

### Results

* 100% repository parsing success
* Automated Mermaid.js architecture diagrams
* Reduced developer onboarding effort

### Technologies

`React` `TailwindCSS` `OpenAI API` `GitHub API` `Mermaid.js`

**GitHub**

https://github.com/debarun23/Git-Repository-Explainer-App

---

# Technical Stack

## AI / ML

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square\&logo=pytorch\&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat-square\&logo=huggingface\&logoColor=black)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat-square\&logo=langchain\&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square\&logo=numpy\&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat-square\&logo=nvidia\&logoColor=white)

### Implemented From Scratch

* Transformer
* Multi-Head Attention
* RoPE
* SwiGLU
* LoRA
* Flash Attention
* Gradient Accumulation
* Cosine Learning Rate Scheduling

---

# Infrastructure

![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square\&logo=docker\&logoColor=white)
![Git](https://img.shields.io/badge/Git-F05032?style=flat-square\&logo=git\&logoColor=white)
![Linux](https://img.shields.io/badge/Linux-FCC624?style=flat-square\&logo=linux\&logoColor=black)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square\&logo=python\&logoColor=white)

---

# Certifications

* DeepLearning.AI — LangChain for LLM Application Development
* DeepLearning.AI — Finetuning Large Language Models
* Hugging Face — NLP Course
* Meta — Version Control
* IBM — Getting Started with AI
* Deloitte — Technology Job Simulation

---

# GitHub Stats

<div align="center">

<img height="170" src="https://github-readme-stats.vercel.app/api?username=debarun23&show_icons=true&theme=transparent&hide_border=true"/>

<img height="170" src="https://github-readme-streak-stats.herokuapp.com?user=debarun23&theme=transparent&hide_border=true"/>

</div>

---

<div align="center">

Building intelligence from first principles.

LLM Engineering · Fine-Tuning · RAG · Agentic AI

Kolkata, India

</div>
