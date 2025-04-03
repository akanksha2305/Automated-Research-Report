# 📄 Automated Research Report Generation System

This project is an **AI-powered application** that automates the process of generating structured research reports using **Retrieval-Augmented Generation (RAG)**. It integrates:
- A **vector-based semantic search system**
- **Groq-accelerated language models** for synthesis
- A clean **Streamlit UI** for end-to-end interaction

---

## 🧠 Problem Statement

In research-intensive domains, generating detailed reports is time-consuming and error-prone. Researchers manually sift through large volumes of data, documents, and literature scattered across formats and locations.

This project solves that by automating the process through:
- Intelligent document ingestion
- Semantic search using embeddings
- Research report generation via Groq-powered LLMs

---

## 🎯 Features

- 📚 **Multi-source Document Ingestion** (PDF, DOCX, TXT)
- 🔍 **Semantic Search with FAISS**
- 🤖 **Groq-hosted LLMs** for fast and cost-effective synthesis
- 📝 **Structured Report Output**: Abstract, Literature Review, Methodology, Findings, Conclusion
- ⚡ **Streamlit-based UI**
- ✅ Modular and easy-to-extend

---

## 🛠️ Tech Stack

| Component        | Technology                           |
|------------------|--------------------------------------|
| Embeddings       | `text-embedding-3-small` via Groq    |
| Vector DB        | FAISS                                |
| LLM              | `llama3-8b-8192` via Groq             |
| UI               | Streamlit                            |
| File Handling    | PyPDF2, python-docx                  |
| Environment      | Python 3.10+                          |
| Deployment (opt) | Hugging Face, Docker, Azure          |

---


## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/research-report-generator.git
cd research-report-generator
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your .env file
```bash
GROQ_API_KEY=your_groq_key
GROQ_API_BASE=https://api.groq.com/openai/v1
EMBEDDING_MODEL=text-embedding-3-small
```

### 4. Run the embedding pipeline
```bash
python build_embeddings.py
```

### 5. Launch the Streamlit app
```bash
streamlit run streamlit_app.py
```

## Example Query

"The role of LLMs in automating academic research workflows"

Output:

✅ Abstract

✅ Literature Review

✅ Methodology

✅ Findings

✅ Conclusion


