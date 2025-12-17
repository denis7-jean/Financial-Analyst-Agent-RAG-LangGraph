# 📈 Financial Analyst Agent: RAG + LangGraph 

![Status](https://img.shields.io/badge/Status-Active_Development-green)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Tech](https://img.shields.io/badge/GenAI-LangChain_|_LangGraph-orange)

## 🎥 Project Demo

### 1. Core Capability: RAG + Precision Math
**Scenario:** The user asks about specific risk factors (Retrieval) and then requests a projection based on 2024 Net Sales (Reasoning + Calculation).
*Notice how the agent explicitly shows the calculation formula to ensure accuracy.*

https://github.com/denis7-jean/assets/demo_rag_calculation.mp4
*(Note: See instructions below on how to get this link)*

### 2. Advanced Reasoning: Multi-turn Context
**Scenario:** The user asks to compare the calculated projection against historical 2023 data.
*The agent recalls the previous calculation result and performs a new difference calculation without needing to re-fetch data.*

https://github.com//denis7-jean/assets/demo_multiturn_comparison.mp4


## 📖 Project Overview
This project is an advanced **Agentic RAG System** designed to perform autonomous analysis of financial documents (SEC 10-K filings). 

Unlike traditional RAG pipelines that simply retrieve text, this system uses **LangGraph** to orchestrate a multi-step reasoning workflow. It employs specific tools for different modes of analysis—vector retrieval for semantic search, a Python calculator for precise quantitative analysis, and a compliance engine for rule-based risk assessment.

### 🎯 Objective
To solve the "hallucination" and "math" problems in financial LLM applications by decoupling **retrieval**, **reasoning**, and **calculation**.

## 🏗️ High-Level Architecture

The system follows a graph-based orchestration pattern:

```mermaid
graph LR
    A[User Query] --> B(Router / Intent Classifier)
    B --> C{Decision Node}
    C -- "Need Info" --> D[RAG Retriever Tool]
    C -- "Need Math" --> E[Python Calculator Tool]
    C -- "Check Risk" --> F[Compliance Logic Tool]
    D & E & F --> G[State Update]
    G --> H{Is Answer Ready?}
    H -- No --> B
    H -- Yes --> I[Final Answer Generator]
````

## ✨ Key Features & Technical Capabilities

### 1\. Advanced RAG Engineering

  * **Hybrid Search:** Combines semantic search (vector embeddings) with keyword search to handle specific financial terminology.
  * **Smart Chunking:** Implements context-aware chunking strategies to keep financial tables and footnotes intact.
  * **Citation-Backed Answers:** Every claim in the final output is referenced back to the specific source document page.

### 2\. LangGraph Agent Workflow

  * **Multi-Tool Orchestration:** The model isn't just answering; it's *acting*. It autonomously decides when to use a calculator versus when to read text.
  * **Cyclic Graph:** Allows the agent to "self-correct" (e.g., if a retrieval comes back empty, it can rewrite the query and try again).
  * **State Management:** Maintains conversation history and intermediate reasoning steps across the workflow.

### 3\. Domain-Specific Tools

  * **📄 10-K Retriever:** Accesses indexed vector stores of Apple, Microsoft, and Tesla 10-K filings.
  * **🧮 Financial Calculator:** A Python REPL sandbox that executes code for precise YoY growth and margin calculations (solving the LLM math deficiency).
  * **⚖️ Compliance Checker:** A rule-based tool that flags specific risk factors (e.g., "Does this mention pending litigation?").

## 🛠️ Tech Stack

  * **Orchestration:** LangChain, LangGraph
  * **LLM:** GPT-4o / Claude 3.5 Sonnet (Configurable)
  * **Vector Database:** ChromaDB / FAISS
  * **Embeddings:** OpenAI text-embedding-3-small / HuggingFace
  * **Serving:** FastAPI (Backend) + Streamlit (Frontend UI)

## 📂 Project Structure (Planned)

```bash
├── data/                   # Raw 10-K PDFs and processed chunks
├── src/
│   ├── ingestion/          # PDF loading, cleaning, and embedding pipelines
│   ├── retrieval/          # Vector DB logic and custom retrievers
│   ├── graph/              # LangGraph nodes, edges, and state definitions
│   ├── tools/              # Custom tools (Calculator, Compliance, Search)
│   └── utils/              # Helper functions and config
├── app.py                  # Streamlit UI entry point
├── server.py               # FastAPI backend
└── notebooks/              # Prototyping and experimentation
```

## 🚀 Getting Started

*(Instructions to be added as development proceeds)*

1.  Clone the repo
2.  Install dependencies: `pip install -r requirements.txt`
3.  Set up `.env` with API keys
4.  Run the ingestion pipeline: `python src/ingestion/ingest.py`
5.  Launch the agent: `streamlit run app.py`

-----

*Author: [Huiyao Lan]*
*This project is part of a portfolio demonstrating end-to-end AI engineering skills, complementing my work on [LoRA Fine-Tuning Pipelines]([https://www.google.com/search?q=link-to-your-other-repo](https://github.com/denis7-jean/financial-nlp-lora-pipeline.git)).*
