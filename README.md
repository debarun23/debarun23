<!-- Typing SVG header -->
<div align="center">
  <img src="https://readme-typing-svg.herokuapp.com?font=JetBrains+Mono&weight=700&size=28&duration=3500&pause=1000&color=00D4FF&center=true&vCenter=true&width=700&lines=Debarun+Das;AI+%2F+ML+Engineer;LLM+Architect;Building+Intelligence+from+Scratch" alt="Typing SVG" />
  
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=120&section=header&animation=twinkling" />
</div>

---

## `whoami`

```python
class DebarunDas:
    role        = "AI / ML Engineer"
    focus       = ["LLM Architecture", "LoRA Fine-Tuning", "RAG Systems", "MLOps"]
    current     = "AI Engineer @ Mind & Matter UK"
    flagship    = "Built 95M-param LLM from scratch — beats GPT-3 Small (PPL 24.40 vs 26.0)"
    philosophy  = "Understand the internals. Build from scratch. Deploy with purpose."
```

> I don't just *use* AI — I architect, train, and deploy it.
> Built a **95M parameter transformer** from first principles using LLaMA-style architecture (RoPE + SwiGLU), outperforming GPT-3 Small on WikiText-103 with **30% fewer parameters**.

---

## Flagship Project — ENG_llmV03

> 95M Parameter Language Model · Built Entirely From Scratch · No `from_pretrained()`

```
Architecture  :  RoPE + SwiGLU Transformer (LLaMA / Mistral style)
Parameters    :  95M
Training Data :  WikiText-103 (103M tokens)
Base PPL      :  24.40  ← beats GPT-3 Small (26.0) with 30% fewer params
Fine-tuned PPL:  20.83  (two-stage LoRA: R128 → R64)
LoRA Params   :  ~1.6M trainable (1.8% of base)
QA Dataset    :  355k clean pairs (SciQ + ELI5 + FreebaseQA)
Hardware      :  Single RTX 5050 (8.5GB VRAM)
Status        :  Complete
```

**What makes this different from every other "LLM from scratch" tutorial:**

- Diagnosed and fixed **validation contamination** mid-training (V2 → V3 full rebuild)
- Implemented **RoPE and SwiGLU from mathematical foundations** — not copy-paste
- Hit and fixed **catastrophic forgetting** — switched full fine-tune → two-stage LoRA
- Engineered **355k clean QA pairs** from 405k raw (regex filtering, deduplication, length enforcement)
- Every epoch tracked, every bug documented, every architectural decision justified

**V2 → V3 Architecture Upgrade:**

| | V2 | V3 |
|---|---|---|
| Position Encoding | Absolute | **RoPE** |
| FFN Activation | GELU | **SwiGLU** |
| Layers | 8 | **12** |
| Parameters | 77M | **95M** |
| Final PPL | ~28 | **20.83** |

📄 [Full technical documentation](https://github.com/debarun23/LLM-from-scratch) · 🤗 [Model on HuggingFace](https://huggingface.co/Debarun12)

---

## 📌 Featured Projects

**🌱 Terratech — AI Agricultural Monitoring System**

IoT-enabled crop health monitoring with ESP32-CAM and multi-sensor integration.

- **95% accuracy** in plant disease detection using Edge Impulse and OpenCV
- Automated irrigation system reducing water consumption by **28.6%**
- End-to-end pipeline: sensor data → edge AI inference → automated actuation

`ESP32` `OpenCV` `Edge Impulse` `IoT` `Computer Vision`

[GitHub →](https://github.com/debarun23/Terratech-Smart-Agriculture-with-IoT-Robotics-AI)

---

**🪨 Job Granite Guide — Agentic AI Interview Trainer**

Agentic AI assistant for interview preparation powered by IBM Granite and RAG.

- Automated domain-specific question generation with retrieval-augmented context
- CI/CD deployment pipeline on IBM Cloud

`IBM Granite` `RAG` `LangChain` `Agentic AI` `IBM Cloud`

[GitHub →](https://github.com/debarun23/job-granite-guide)

---

**📊 Git Repository Explainer — Intelligent Code Analyzer**

Analyzes any public GitHub repository in real time and generates developer-friendly insights.

- GitHub REST API pipeline with **100% parsing accuracy**
- Auto-generates Mermaid.js architecture and flow diagrams
- Reduced developer onboarding time by **40%**

`React` `TailwindCSS` `OpenAI API` `GitHub API` `Mermaid.js`

[GitHub →](https://github.com/debarun23/Git-Repository-Explainer-App)

---

**📈 Crypto Price Tracker Dashboard**

Real-time tracking for 50+ cryptocurrencies with interactive charts.

- CoinGecko API integration with React Hooks and Google Charts
- **40% faster** load times post-Vite migration

`React` `Google Charts` `CoinGecko API` `Vite`

[GitHub →](https://github.com/debarun23/Crypto-Price-Tracking-App)

---

## ⚙️ Technical Stack

### AI / ML
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat-square&logo=langchain&logoColor=white)](https://langchain.com)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org)
[![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)

**Implemented from scratch:** `Transformer` · `Multi-Head Attention` · `RoPE` · `SwiGLU` · `LoRA` · `Flash Attention` · `Cosine LR Scheduling` · `Gradient Accumulation`

### Infrastructure & Tools
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)
[![Git](https://img.shields.io/badge/Git-F05032?style=flat-square&logo=git&logoColor=white)](https://git-scm.com)
[![Linux](https://img.shields.io/badge/Linux-FCC624?style=flat-square&logo=linux&logoColor=black)](https://linux.org)
[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)

---

## 📊 GitHub Stats

<div align="center">

![Debarun's GitHub Stats](https://github-readme-stats.vercel.app/api?username=debarun23&show_icons=true&theme=transparent&hide_border=true&title_color=00D4FF&icon_color=00D4FF&text_color=ffffff&bg_color=0d1117)
![GitHub Streak](https://github-readme-streak-stats.herokuapp.com?user=debarun23&theme=transparent&hide_border=true&ring=00D4FF&fire=00D4FF&currStreakLabel=00D4FF)

![Activity Graph](https://github-readme-activity-graph.vercel.app/graph?username=debarun23&theme=react-dark&hide_border=true&color=00D4FF&line=00D4FF&point=ffffff&area=true)

</div>

---

## 🎓 Certifications

`DeepLearning.AI` LangChain for LLM Application Development  
`DeepLearning.AI` Finetuning Large Language Models  
`Hugging Face` NLP Course  
`Meta` Version Control · `Deloitte` Technology Job Simulation · `IBM` Getting Started with AI

---

## 📬 Contact

[![Email](https://img.shields.io/badge/Email-0078D4?style=flat-square&logo=microsoft-outlook&logoColor=white)](mailto:debarundas237@gmail.com)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://linkedin.com/in/debarun-das-14b3a4259)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/debarun23)
[![Portfolio](https://img.shields.io/badge/Portfolio-000000?style=flat-square&logo=vercel&logoColor=white)](https://debarun-s-portfolio.vercel.app/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/Debarun12)

---

<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer&animation=twinkling" />
  
  <sub>Building intelligence from first principles · Not <code>from_pretrained()</code></sub>
</div>
