"""
Ingest and index the Nigerian Constitution PDF using a hybrid retrieval approach.
This script performs the following steps:   
1. Loads the PDF document.
2. Extracts section metadata using regex to tag each chunk with its corresponding section.
3. Splits the text into chunks while respecting legal formatting.   
4. Generates embeddings for the chunks using a local HuggingFace model.
5. Stores the embeddings in a local ChromaDB vector store.
6. Sets up a BM25 keyword retriever for precise section-based queries.
7. Combines both retrievers into an EnsembleRetriever for flexible querying.
Make sure to install the required libraries before running this script:
pip install langchain langchain-community langchain-chroma langchain-text-splitters huggingface_hub
Note: Adjust the chunk size, overlap, and retriever weights as needed based on your specific use case and document structure.
"""


import os
import re
import logging
from typing import List
from dotenv import load_dotenv
from huggingface_hub import login

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

# Set up logging for professional tracking
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConstitutionIngestor:
    """Handles the ingestion and hybrid indexing of legal documents."""

    def __init__(self, file_path: str, db_dir: str = "./chroma_db"):
        self.file_path = file_path
        self.db_dir = db_dir
        self.section_pattern = re.compile(r"(?:Section\s+)?(\d+)\.", re.IGNORECASE)
        
        load_dotenv(override=True)
        hf_api_token = os.getenv('HF_API_TOKEN')
        if hf_api_token:
            login(token=hf_api_token)
        else:
            logger.warning("HF_API_TOKEN not found in environment variables. Embeddings may fail.")
        
        # this will now use authenticated session
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

    def load_and_tag_metadata(self) -> List:
        """Loads PDF and injects Section numbers into metadata."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"PDF not found at {self.file_path}")

        logger.info("Loading PDF and extracting metadata...")
        loader = PyPDFLoader(self.file_path)
        docs = loader.load()

        current_section = "Preamble"
        for doc in docs:
            match = self.section_pattern.search(doc.page_content)
            if match:
                current_section = f"Section {match.group(1)}"
            doc.metadata["section"] = current_section
        
        return docs

    def process(self):
        """Executes the full pipeline with noise filtering."""
        try:
            # 1. Load & Metadata tagging
            raw_docs = self.load_and_tag_metadata()

            # 2. Chunking
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800, # Smaller chunks for higher precision
                chunk_overlap=150,
                separators=["\nSection ", "\nPART ", "\n\n", "\n", " "]
            )
            all_splits = text_splitter.split_documents(raw_docs)

            # 3. FILTERING: Remove short Preamble chunks and empty metadata
            splits = [
                doc for doc in all_splits 
                if not (doc.metadata.get("section") == "Preamble" and len(doc.page_content) < 250)
            ]
            
            logger.info(f"Cleaned {len(all_splits) - len(splits)} noisy chunks.")

            # 4. Vector Store (Semantic)
            logger.info("Rebuilding Vector Store...")
            # YOU MUST DELETE ./chroma_db FOLDER MANUALLY BEFORE RUNNING THIS
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                persist_directory=self.db_dir
            )

            # 5. Hybrid Ensemble - Optimized for Legal Search
            bm25_retriever = BM25Retriever.from_documents(splits)
            bm25_retriever.k = 4 # Retrieve more candidates
            
            vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
            
            # 0.7 Weight on BM25 is perfect for "Section X" queries
            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, vector_retriever],
                weights=[0.7, 0.3] 
            )

            logger.info("Ingestion complete. Hybrid index is ready.")
            return ensemble_retriever

        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            raise


if __name__ == "__main__":
    # Update this filename to match your actual file
    PDF_FILE = "constitution-of-the-federal-republic-of-nigeria.pdf"
    
    ingestor = ConstitutionIngestor(PDF_FILE)
    ingestor.process()
