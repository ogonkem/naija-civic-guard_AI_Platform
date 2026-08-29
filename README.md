This is the final, comprehensive version of your README.md. It now includes a "Performance Benchmarks" section that highlights the actual metrics from your eval_report.md, making the project look highly data-driven and professional.
------------------------------
## 🇳🇬 Naija Civic Guard: Nigerian Constitution RAG
Empowering citizens with AI-driven, verifiable insights into the Nigerian Constitution.
Naija Civic Guard is a high-precision Retrieval-Augmented Generation (RAG) system. Unlike generic LLMs, this tool is strictly grounded in the Official Gazette of the Nigerian Constitution. It utilizes a Hybrid Retrieval strategy and a rigorous evaluation pipeline to provide reliable legal answers with direct source citations.
------------------------------
## 📊 Performance Benchmarks
We don't guess—we measure. Based on our latest automated evaluation against the evaluation_set.jsonl, the system achieves the following metrics:

| Metric | Value | Description |
|---|---|---|
| Hit Rate (Top-K) | 60% | The correct legal section is retrieved in the top results 6 out of 10 times. |
| Avg Keyword Coverage | 84% | Our responses maintain high legal accuracy by including essential terminology. |
| Mean Reciprocal Rank | 0.45 | Measures how highly the system ranks the specific target Section. |

Continuous testing ensures that updates to the retrieval weights or chunking strategies improve these scores over time.
------------------------------
## 🏗️ Architecture & Pipeline## 1. Hybrid Ingestion (ingest.py)
Our ingestion process is a "legal-aware" pipeline designed for maximum precision:

* Metadata Tagging: Uses Regex to extract and tag chunks with specific Section and Article numbers for granular citations.
* Ensemble Retrieval: Combines Vector Search (ChromaDB) for semantic intent and Keyword Search (BM25) for specific legal terms. This ensures that a query about "Fundamental Rights" finds Section 33 even if the exact wording varies.

## 2. RAG Service (services.py)
The RagService manages the AI lifecycle:

* Authentication: Securely connects to HuggingFace using HF_API_TOKEN.
* Grounded Generation: Injects retrieved legal sections into a custom system prompt. The LLM (Ollama) is instructed to generate answers based only on the provided sections.

## 3. Evaluation Suite (retrieval_eval.py)
The system generates a detailed eval_report.md (as seen in the logs) which tracks:

* Query-to-Section Mapping: Verifying if the correct section was found for a specific legal question.
* Failed Query Analysis: Identifying gaps in the vector store where legal context might be missing.

------------------------------
## 🛠️ Technical Stack

* Backend: Django 5.x (Python 3.12)
* Orchestration: LangChain
* Local LLM: Ollama (Default: Llama3 or Mistral)
* Embeddings: HuggingFace all-MiniLM-L6-v2
* Vector Store: ChromaDB
* Infrastructure: Docker & Docker Compose

------------------------------
## 🚀 Installation & Setup

   1. Clone the repository and install dependencies:
   
   pip install -r requirements.txt
   
   2. Environment Setup: 
   
   HF_API_TOKEN=your_token_here
   OLLAMA_BASE_URL=http://localhost:11434
   LANGSMITH_TRACING=true
   LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com
   LANGSMITH_API_KEY=<LANGSMITH_API_KEY>
   LANGSMITH_PROJECT="Naija-Civic-Guard"
   HF_API_TOKEN=<HF_API_TOKEN>
   GROQ_API_KEY=<GROQ_API_KEY>
   GROQ_LLM_MODEL=llama-3.1-8b-instant
   
   3. Process the Constitution:
   
   python ingest.py
   
   4. Run Evaluation (Optional):
   
   python retrieval_eval.py
   
   5. Start the Server:
   
   python manage.py runserver

   6. Test:

   python manage.pt test
   
   
## Docker Deployment

docker-compose up --build

------------------------------
## ⚖️ Disclaimer
Naija Civic Guard is an AI-powered educational tool. While it uses the official Nigerian Constitution, users should always verify legal findings with the Official Gazette or a legal professional.
------------------------------
Since your Hit Rate is at 60%, would you like some tips on how to adjust your chunk size or overlap in ingest.py to push that closer to 80%?


