# <img src="https://readme-typing-svg.herokuapp.com?font=JetBrains+Mono&weight=700&size=32&duration=3500&pause=1000&color=00D4FF&center=false&vCenter=true&width=600&lines=Debarun+Das;AI+%2F+ML+Engineer;LLM+Architect;Building+Intelligence+from+Scratch" alt="Typing SVG" />

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=120&section=header&animation=twinkling" width="100%"/>

---

## `whoami`

```python
class DebarunDas:
    role        = "AI / ML Engineer"
    focus       = ["LLM Architecture", "Model Training", "Medical NLP", "MLOps"]
    currently   = "Training a 95M param LLM from scratch (RoPE + SwiGLU)"
    building    = "Domain-adapted Medical QA System"
    philosophy  = "Understand the internals. Build from scratch. Deploy with purpose."
```

> I don't just *use* AI — I architect, train, and deploy it.  
> Currently building a **95M parameter transformer** from scratch using modern LLaMA-style architecture, outperforming GPT-3 Small on WikiText-103 with 22M fewer parameters.

---

## 🧠 AI / ML Engineering

### Flagship Project — ENG-LLM (In Progress)

```
Architecture  :  RoPE + SwiGLU Transformer (LLaMA-style)
Parameters    :  95M
Training Data :  WikiText-103
Current PPL   :  24.4 → target 20-22
Status        :  Epoch 16/20 base training
Fine-tune     :  Medical QA (198K pairs — MedQA + MedMCQA + ChatDoctor)
Benchmark     :  Outperforms GPT-3 Small (125M) at epoch 8
```

**What makes this different from every other "LLM from scratch" project:**
- Diagnosed and fixed validation contamination mid-training (V2 → V3 rebuild)
- Implemented RoPE and SwiGLU from mathematical foundations — not copy-paste
- Documented every epoch, every bug, every architectural decision
- 16+ epochs of stable training — no NaN, no explosion, no shortcuts

**V2 → V3 Architecture Upgrade:**

| | V2 | V3 |
|---|---|---|
| Position Encoding | Absolute | **RoPE** |
| FFN Activation | GELU | **SwiGLU** |
| Layers | 8 | **12** |
| Parameters | 77M | **95M** |
| Final PPL | ~28 | **~20-22** |

---

## ⚙️ Technical Stack

### AI / ML
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat-square&logo=nvidia&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square&logo=python&logoColor=white)

**Architectures implemented from scratch:**
`Transformer` · `Multi-Head Attention` · `RoPE` · `SwiGLU` · `Cosine LR Scheduling` · `Gradient Accumulation` · `Mixed Precision Training`

### Full-Stack & DevOps
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=flat-square&logo=javascript&logoColor=black)
![React](https://img.shields.io/badge/React-20232A?style=flat-square&logo=react&logoColor=61DAFB)
![Next.js](https://img.shields.io/badge/Next.js-000000?style=flat-square&logo=next.js&logoColor=white)
![Java](https://img.shields.io/badge/Java-ED8B00?style=flat-square&logo=openjdk&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192?style=flat-square&logo=postgresql&logoColor=white)

---

## 📌 Featured Projects

<details>
<summary><strong>🤖 ENG-LLM — Medical QA Language Model (In Progress)</strong></summary>

**The real one.** Built a 95M parameter transformer from scratch using LLaMA-style architecture (RoPE + SwiGLU). Trained on WikiText-103 for 20 epochs, fine-tuning on 198K medical QA pairs (MedQA + MedMCQA + ChatDoctor).

- Implemented RoPE, SwiGLU, gradient accumulation, AMP training from scratch
- Rebuilt architecture V2 → V3 after diagnosing validation contamination
- PPL progression: 34.48 → 24.4 (epoch 16) — outperforming GPT-3 Small (125M)
- Fine-tuning target: Medical QA with clinical accuracy evaluation on MedQA benchmark

`PyTorch` `Transformers` `CUDA` `HuggingFace Datasets` `Medical NLP`

</details>

<details>
<summary><strong>🌱 Terratech — AI Agricultural Monitoring System</strong></summary>

IoT-enabled crop health monitoring with ESP32-CAM and multi-sensor integration.

- 95% accuracy in plant disease detection using Edge Impulse and OpenCV
- Automated irrigation system reducing water consumption by 28.6%
- End-to-end pipeline from sensor data to AI inference at the edge

`ESP32` `OpenCV` `Edge Impulse` `IoT` `Computer Vision`

[GitHub →](https://github.com/debarun23/Terratech-Smart-Agriculture-with-IoT-Robotics-AI)

</details>

<details>
<summary><strong>🪨 Job Granite Guide — Agentic AI Interview Trainer</strong></summary>

Agentic AI assistant for interview preparation powered by IBM Granite and RAG.

- Automated question generation with domain-specific retrieval
- CI/CD deployment pipeline on IBM Cloud

`IBM Granite` `RAG` `Agentic AI` `IBM Cloud`

[GitHub →](https://github.com/debarun23/job-granite-guide)

</details>

<details>
<summary><strong>📊 Git Repository Explainer — Intelligent Code Analyzer</strong></summary>

Analyzes any public GitHub repository in real time and generates developer-friendly insights, architecture diagrams, and workflow documentation.

- GitHub REST API pipeline with 100% parsing accuracy
- Auto-generates Mermaid.js architecture and flow diagrams
- OpenAI-powered structured project descriptions
- Reduced developer onboarding time by 40%

`React` `TailwindCSS` `OpenAI API` `GitHub API` `Mermaid.js`

[GitHub →](https://github.com/debarun23/Git-Repository-Explainer-App)

</details>

<details>
<summary><strong>📈 Crypto Price Tracker Dashboard</strong></summary>

Real-time tracking for 50+ cryptocurrencies with interactive charts.

- CoinGecko API integration with React Hooks and Google Charts
- 40% faster load times post Vite migration

`React` `Google Charts` `CoinGecko API` `Vite`

[GitHub →](https://github.com/debarun23/Crypto-Price-Tracking-App)

</details>

---

## 📊 GitHub Stats

<p align="center">
  <img src="https://github-readme-stats.vercel.app/api?username=debarun23&show_icons=true&theme=transparent&hide_border=true&title_color=00D4FF&icon_color=00D4FF&text_color=ffffff&bg_color=0d1117" height="150"/>
  <img src="https://github-readme-streak-stats.herokuapp.com?user=debarun23&theme=transparent&hide_border=true&ring=00D4FF&fire=00D4FF&currStreakLabel=00D4FF" height="150"/>
</p>

<p align="center">
  <img src="https://github-readme-activity-graph.vercel.app/graph?username=debarun23&theme=react-dark&hide_border=true&color=00D4FF&line=00D4FF&point=ffffff&area=true" width="95%"/>
</p>

---

## 🎓 Certifications

`Meta` Front-End Developer &nbsp;·&nbsp;
`Meta` Version Control &nbsp;·&nbsp;
`IBM` Journey to Cloud &nbsp;·&nbsp;
`IBM` Getting Started with AI &nbsp;·&nbsp;
`Deloitte` Technology Job Simulation &nbsp;·&nbsp;
`Udemy` Complete SQL Bootcamp

---

## 📬 Contact

<p>
  <a href="mailto:debarundas237@gmail.com"><img src="https://img.shields.io/badge/Email-0078D4?style=flat-square&logo=microsoft-outlook&logoColor=white"/></a>&nbsp;
  <a href="https://linkedin.com/in/debarun-das"><img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white"/></a>&nbsp;
  <a href="https://github.com/debarun23"><img src="https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white"/></a>&nbsp;
  <a href="https://debarun-s-portfolio.vercel.app/"><img src="https://img.shields.io/badge/Portfolio-000000?style=flat-square&logo=vercel&logoColor=white"/></a>
</p>

---

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer&animation=twinkling" width="100%"/>

<p align="center">
  <sub>Building intelligence from first principles · Not <code>from_pretrained()</code></sub>
</p>
