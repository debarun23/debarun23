<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0a0a0a,50:0d1117,100:161b22&height=140&section=header&text=Debarun%20Das&fontSize=42&fontColor=ffffff&fontAlignY=55&desc=AI%20Engineer%20%7C%20LLM%20Architect%20%7C%20Building%20from%20First%20Principles&descAlignY=78&descSize=14&animation=fadeIn" />

[![Typing SVG](https://readme-typing-svg.herokuapp.com?font=JetBrains+Mono&weight=600&size=16&duration=3000&pause=1200&color=58A6FF&center=true&vCenter=true&width=600&lines=Built+a+95M+LLM+from+scratch+%E2%80%94+beats+GPT-3+Small;Fine-tuned+Qwen2.5-3B+on+8GB+consumer+GPU;RAG+%7C+LoRA+%7C+QLoRA+%7C+PyTorch+%7C+LangChain;Open+to+AI+%2F+ML+Engineering+roles)](https://github.com/debarun23)

<br/>

[![LinkedIn](https://img.shields.io/badge/LinkedIn-debarun--das-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://linkedin.com/in/debarun-das)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Debarun12-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/Debarun12)
[![Portfolio](https://img.shields.io/badge/Portfolio-Live-00D4FF?style=flat-square&logo=vercel&logoColor=white)](https://debarun-s-portfolio.vercel.app/)
[![Email](https://img.shields.io/badge/Email-debarundas237-EA4335?style=flat-square&logo=gmail&logoColor=white)](mailto:debarundas237@gmail.com)

</div>

---

## `whoami`

```python
class DebarunDas:
    role        = "AI Engineer @ Mind & Matter UK"
    location    = "Kolkata, India"
    focus       = ["LLM Architecture", "LoRA / QLoRA Fine-Tuning", "RAG Systems", "MLOps"]
    flagship    = "Built 95M-param LLM from scratch — beats GPT-3 Small (PPL 24.40 vs 26.0)"
    second      = "Fine-tuned Qwen2.5-3B via QLoRA on 8GB GPU — Val PPL 2.40, target was <10"
    philosophy  = "Understand the internals. Build from scratch. Deploy with purpose."
    open_to     = "AI / ML Engineering roles — full-time or contract"
```

> I don't just *use* AI — I architect, train, and deploy it.
> Two production models on HuggingFace. Built on a single consumer GPU. Fully documented.

---

## 🏆 Project 1 — ENG_llmV03

> **95M Parameter Transformer · Built Entirely From Scratch · No `from_pretrained()`**

```
Architecture   :  RoPE + SwiGLU Transformer  (same design as LLaMA / Mistral / Gemma)
Parameters     :  95M
Training Data  :  WikiText-103  (103M tokens)
Base PPL       :  24.40  ←  beats GPT-3 Small (26.0) with 30% fewer parameters
Fine-tuned PPL :  20.83  (two-stage LoRA: R128 → merged → R64)
LoRA Params    :  ~1.6M trainable  (1.8% of base)
QA Dataset     :  355k clean pairs  (SciQ + ELI5 + FreebaseQA, cleaned from 405k raw)
Hardware       :  Single RTX 5050  ·  8.5 GB VRAM
Status         :  ✅ Complete  ·  Deployed on HuggingFace
```

**Why this project is different:**

- Rebuilt architecture V2 → V3 after diagnosing and fixing **validation contamination** (PPL spikes 43→73)
- Implemented **RoPE, SwiGLU, Flash Attention, and LoRA from mathematical foundations** — not copied
- Hit **catastrophic forgetting** on full fine-tune (PPL 24→35+) → diagnosed root cause → switched to two-stage LoRA
- **Engineered 355k clean QA pairs** from 405k raw: regex deduplication, filler-phrase filtering, length enforcement
- Every epoch tracked. Every bug documented. Every architectural decision justified.

| | V2 | V3 |
|:--|:--:|:--:|
| Position Encoding | Absolute | **RoPE** |
| FFN Activation | GELU | **SwiGLU** |
| Layers | 8 | **12** |
| Parameters | 77M | **95M** |
| Final PPL | ~28 | **20.83** |

[![HuggingFace](https://img.shields.io/badge/Model-Debarun12%2FENG--llmV03-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/Debarun12/ENG-llmV03)
[![GitHub](https://img.shields.io/badge/Code-LLM--from--scratch-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/debarun23/LLM-from-scratch)

---

## 🔒 Project 2 — JavaExpert-Qwen2.5-3B

> **Domain-Locked Java QA · QLoRA on Consumer Hardware · Production-Deployed**

```
Base Model     :  Qwen2.5-3B  (4-bit NF4 quantization via QLoRA)
Training VRAM  :  < 7 GB peak  (hard budget: 8 GB)
Inference VRAM :  < 2 GB  (merged model, no adapter dependency at runtime)
Val Perplexity :  2.40  ←  target was < 10, beaten by 4×
Domain Refusal :  8.5 / 10  (refuses non-Java queries without external guardrails)
Java Accuracy  :  8.5 / 10
QA Dataset     :  7,921 pairs generated from 42,000-line Java PDF corpus
Hardware       :  Single RTX 5050  ·  8 GB VRAM
Status         :  ✅ Complete  ·  Deployed on HuggingFace
```

**VRAM optimization decisions — every choice measured, not assumed:**

| Component | Initial | Final | VRAM Saved |
|:--|:--|:--|:--:|
| LoRA rank | 32 across 7 modules | 16 on q_proj + v_proj | −1.5 GB |
| Batch size | 4 | 1 + grad accum ×8 | −2.0 GB |
| Optimizer | AdamW | Adafactor | −1.0 GB |
| Compute dtype | fp16 | bf16 | Stable on Blackwell |

**Bugs found and fixed that don't appear in tutorials:**
- `SFTTrainer` step-count inflation: 2,139 steps on 90 samples (24× expected) — silent, no error
- PyTorch 2.6 `weights_only=True` default change breaking checkpoint resume

[![HuggingFace](https://img.shields.io/badge/Model-Debarun12%2FJavaExpert--Qwen2.5--3B-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/Debarun12/JavaExpert-Qwen2.5-3B)

---

## 📌 Other Projects

**🌱 Terratech — AI Agricultural Monitoring**

IoT crop health system with ESP32-CAM + Edge AI inference. 95% plant disease detection accuracy. Automated irrigation reducing water use by 28.6%.

`ESP32` `OpenCV` `Edge Impulse` `IoT` `Computer Vision` — [GitHub →](https://github.com/debarun23/Terratech-Smart-Agriculture-with-IoT-Robotics-AI)

---

**🪨 Job Granite Guide — Agentic AI Interview Trainer**

IBM Granite + RAG pipeline for domain-specific interview preparation. CI/CD deployed on IBM Cloud.

`IBM Granite` `RAG` `LangChain` `Agentic AI` — [GitHub →](https://github.com/debarun23/job-granite-guide)

---

**📊 Git Repository Explainer**

Real-time GitHub repo analyzer: 100% parsing accuracy, auto-generates Mermaid.js architecture diagrams, 40% faster onboarding.

`React` `OpenAI API` `GitHub API` `Mermaid.js` — [GitHub →](https://github.com/debarun23/Git-Repository-Explainer-App)

---

**📈 Crypto Price Tracker**

Real-time dashboard for 50+ cryptocurrencies. 40% faster load time post-Vite migration.

`React` `Google Charts` `CoinGecko API` — [GitHub →](https://github.com/debarun23/Crypto-Price-Tracking-App)

---

## ⚙️ Technical Stack

**AI / ML — Core**

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat-square&logo=langchain&logoColor=white)](https://langchain.com)
[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org)

**Implemented from scratch:**
`Transformer` · `Multi-Head Attention` · `RoPE` · `SwiGLU` · `LoRA` · `QLoRA` · `Flash Attention` · `Cosine LR Scheduling` · `Gradient Accumulation` · `Mixed Precision Training`

**Infrastructure**

[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)
[![Git](https://img.shields.io/badge/Git-F05032?style=flat-square&logo=git&logoColor=white)](https://git-scm.com)
[![Linux](https://img.shields.io/badge/Linux-FCC624?style=flat-square&logo=linux&logoColor=black)](https://linux.org)

---

## 📊 GitHub Stats

<div align="center">

<img src="https://github-readme-stats.vercel.app/api?username=debarun23&show_icons=true&theme=github_dark&hide_border=true&title_color=58A6FF&icon_color=58A6FF&text_color=c9d1d9&bg_color=0d1117&include_all_commits=true&count_private=true" height="165"/>
<img src="https://github-readme-stats.vercel.app/api/top-langs/?username=debarun23&layout=compact&theme=github_dark&hide_border=true&title_color=58A6FF&text_color=c9d1d9&bg_color=0d1117&langs_count=6" height="165"/>

<img src="https://github-readme-streak-stats.herokuapp.com?user=debarun23&theme=github-dark-blue&hide_border=true&ring=58A6FF&fire=58A6FF&currStreakLabel=58A6FF&sideLabels=c9d1d9&dates=8b949e" />

<img src="https://github-readme-activity-graph.vercel.app/graph?username=debarun23&theme=github-compact&hide_border=true&color=58A6FF&line=58A6FF&point=ffffff&area=true&area_color=58A6FF" />

</div>

---

## 🎓 Certifications

| Issuer | Certification |
|:--|:--|
| `DeepLearning.AI` | LangChain for LLM Application Development |
| `DeepLearning.AI` | Finetuning Large Language Models |
| `Hugging Face` | NLP Course |
| `Deloitte` | Technology Job Simulation |
| `Meta` | Version Control |
| `IBM` | Getting Started with Artificial Intelligence |

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:161b22,50:0d1117,100:0a0a0a&height=100&section=footer&animation=fadeIn"/>

**`Building intelligence from first principles · Not from_pretrained()`**

*Available for AI / ML Engineering roles · Kolkata, India · Remote-friendly*

</div>
