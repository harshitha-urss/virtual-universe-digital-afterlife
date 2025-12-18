# Virtual Universe Construction for Digital Afterlife using Generative Neural Networks

> A large-scale multimodal AI research and engineering project focused on building a **persistent, interactive digital persona** capable of generating **text, voice, and facial
video responses** using generative neural networks.

This project represents **over 6 months of continuous development**, experimentation, and system integration, and forms the **core technical and research pillar** of my MS (Artificial 
Intelligence) profile.

---

## ⚠️ Important Note (Please Read)

This repository contains a **curated and representative subset** of the complete system.
The full project involved **hundreds of files, experiments, internal modules, and iterations** developed over several months in a local VS Code environment.

Only **academically appropriate and non-sensitive components** are included here.

---

## 🔐 Source Code Availability & Copyright Notice

The complete implementation, trained models, and full project assets are **intentionally not made public**.

### Reason for Limited Public Release

The system was developed using a **public business icon (Ratan Tata)** as a *personality reference* strictly for academic and experimental purposes.

Due to:
- copyright considerations,
- ethical responsibility,
- and the risk of identity misuse,

the full source code, datasets, trained voice models, and facial generation pipelines **cannot be publicly released**.

This repository therefore presents a **minimalized, carefully curated representation** of the original work to demonstrate **system design, research depth, and technical understanding**
while strictly adhering to responsible AI principles.

Detailed implementation details can be discussed in an **academic or institutional context** if required.

---

## Project Overview

The **Virtual Universe** system explores how generative AI can be used to construct a *digital afterlife–style interactive presence* by combining:

- Retrieval-Augmented Generation (RAG)
- Large Language Models (LLMs)
- Neural voice synthesis
- Audio-driven facial animation

Given a **natural language query**, the system produces:
1.  Context-aware **text response**
2.  Natural-sounding **voice output**
3.  Lip-synced **facial video response**

Each modality is implemented as an **independent module**, enabling scalability, experimentation, and future research extensions.

---

## Research Motivation

Human knowledge, communication style, and experiential intelligence are often lost over time.
This project investigates how **Generative Neural Networks** can be used to:

- preserve interaction patterns,
- enable realistic multimodal responses,
- study long-term digital interaction systems,
- and design responsible AI architectures for identity-sensitive use cases.

The goal is **not imitation**, but **research into multimodal system design, retrieval accuracy, response grounding, and ethical AI deployment**.

---

## High-Level System Architecture

    User Query
        ↓
    Query Preprocessing
        ↓
    MiniLM Embedding Model
        ↓
    FAISS Vector Retrieval
        ↓
    Context-Aware LLM (GPT4All)
        ↓
    Text Response
        ↓
    XTTS Voice Synthesis
        ↓
    SadTalker Video Generation
        ↓
    Multimodal Digital Avatar Output


Each stage is implemented as an **independent research module**.

---

## Core Technologies & Methodologies

### Retrieval-Augmented Generation (RAG)
- MiniLM used for **384-dimensional semantic embeddings**
- FAISS for **fast top-k similarity search**
- Prevents LLM hallucinations by grounding responses in retrieved context

### Large Language Model
- GPT4All used for **offline, controllable generation**
- Prompt builder dynamically injects retrieved context
- Ensures stylistic and semantic consistency

### Voice Synthesis
- XTTS v2 for **reference-based neural voice cloning**
- Produces natural intonation and speaker similarity

### Video Generation
- SadTalker for **audio-driven facial animation**
- Generates realistic lip-sync using a single reference image

---

## 🎥 Live System Demonstrations (Actual Output)

-  **Voice Bot Demo**  
  https://www.youtube.com/watch?v=CnrOs3BQoz8

-  **Video Bot Demo**  
  https://www.youtube.com/watch?v=01Jgs0BZJ-c

These links show the **real working system**, not mockups:
These outputs were generated using the same pipeline described in this repository.

---

## 📁 Repository Structure (Curated Representation)

```text
virtual-universe-digital-afterlife/
│
├── index.html           # Project website (GitHub Pages)
├── style.css            # Frontend styling
├── script.js            # UI interaction logic
│
├── assets/
│   ├── architecture.png
│   ├── flow_diagrams.png
│   └── system_outputs/
│
├── core_modules/        # Representative AI modules
│   ├── text_bot.py
│   ├── voice_bot.py
│   └── video_bot.py
│
├── docs/
│   ├── Final_PPT.pptx
│   └── Final_Project_Report.pdf
│
└── README.md
```

## Development Depth & Effort

The complete project involved:
- transcript preprocessing & chunking
- Embedding consistency analysis
- FAISS index tuning & evaluation
- Retrieval error analysis
- Latency benchmarking (CPU vs GPU)
- End-to-end multimodal testing
- UI–backend synchronization
- Report writing, diagrams, metrics, and evaluations

Only non-sensitive, academically appropriate components are shared here.

## Ethical & Legal Responsibility (Critical):

Certain implementation details are intentionally not open-sourced.

Reasons:
- A public business icon was used as a personality reference
- Releasing full voice, face, or trained models could enable misuse
- Ethical AI principles and academic integrity were strictly followed

Therefore:
- No identity-specific datasets are released
- No trained voice or face models are shared
- No deployable end-to-end executable is provided
- 
This project prioritizes responsible AI over exposure.

## Future Research Directions

- Persistent vector-memory integration
- Emotion-aware voice & facial synthesis
- Multi-agent virtual universes
- Privacy-preserving local inference
- Reinforcement learning for adaptive responses

## Author

**HARSHITHA M V**

AI & ML Engineer done this project as a part of final year project

Research Interests:
- Artificial Intelligence
- Generative AI
- Multimodal Systems
- Deep Learning
- Machine Learning
- Retrieval-Augmented Models
- Responsible AI
