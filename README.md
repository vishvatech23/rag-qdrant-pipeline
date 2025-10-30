# Retrieval-Augmented Generation (RAG) Pipeline

A modular and scalable **Retrieval-Augmented Generation (RAG)** pipeline built using **FastAPI**, **Sentence Transformers**, and **Qdrant** for high-performance semantic document retrieval and question answering.

---

## Overview

This project implements a complete RAG workflow — from document ingestion to semantic search and response generation — using modern open-source tools and vector databases.  
It allows users to upload documents in various formats (PDF, TXT, DOCX, HTML) and query them using natural language to retrieve contextually relevant information.

---

## Architecture

      ┌────────────┐
      │  Documents │
      └──────┬─────┘
             │
             ▼
     ┌────────────────────┐
     │ Text Extraction &  │
     │   Preprocessing    │
     └────────────────────┘
             │
             ▼
    ┌─────────────────────┐
    │ SentenceTransformer │
    │  (Embeddings)       │
    └─────────────────────┘
             │
             ▼
      ┌────────────┐
      │  Qdrant DB │
      │ (Vectors)  │
      └────────────┘
             │
             ▼
      ┌──────────────┐
      │ FastAPI App  │
      │ (Query API)  │
      └──────────────┘

---

## Tech Stack

| Category | Technologies |
|-----------|--------------|
| **Backend Framework** | FastAPI |
| **Vector Database** | Qdrant |
| **Embeddings** | Sentence Transformers (`all-MiniLM-L6-v2`) |
| **Text Processing** | NLTK, BeautifulSoup4, PyMuPDF |
| **Deployment** | Docker, Uvicorn |
| **Testing & Evaluation** | Python Unittest, Precision@K, Recall@K |
| **Version Control** | Git, GitHub |

---

## Features

✅ Multi-format document ingestion (**PDF, DOCX, HTML, TXT**)  
✅ Automatic text cleaning, chunking, and embedding generation  
✅ RESTful API for document upload and semantic query retrieval  
✅ Real-time response generation via context-aware search  
✅ Modular, production-ready architecture  
✅ Dockerized for easy deployment  

---

## API Endpoints

| Method | Endpoint | Description |
|--------|-----------|-------------|
| `POST` | `/ingest` | Uploads and indexes a new document |
| `POST` | `/query` | Retrieves relevant context from Qdrant |
| `GET` | `/collections` | Lists available vector collections |
| `GET` | `/health` | Health check endpoint |

---

## 🧰 Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/vishvatech23/rag-qdrant-pipeline.git
cd rag-qdrant-pipeline

2️⃣ Create a Virtual Environment
python -m venv venv
venv\Scripts\activate    # On Windows
source venv/bin/activate # On Mac/Linux

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the Application
uvicorn main:app --reload
Then visit → http://127.0.0.1:8000/docs to explore the interactive Swagger UI.

🐳 Run with Docker
docker-compose up --build

🧪 Testing
Run basic unit tests:
pytest

Evaluate retrieval performance:
python evaluate_rag.py

📂 Project Structure
rag-qdrant-pipeline/
│
├── main.py                 # FastAPI app (entry point)
├── pipeline_qdrant.py      # Core RAG pipeline logic
├── evaluate_rag.py         # Retrieval evaluation metrics
├── demo.py                 # Sample demonstration script
├── requirements.txt        # Dependencies
├── Dockerfile              # Container setup
├── docker-compose.yml      # Optional multi-service setup
├── uploads/                # Uploaded sample docs
└── docs/                   # Documentation and usage notes

👨‍💻 Author
Vishvajit Menon
📧 Email: vishvajitmenon75@gmail.com
🔗 LinkedIn
💻 GitHub

⭐ If you found this project helpful, consider giving it a star on GitHub!
