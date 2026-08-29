"""Retrieval Evaluator for JSONL Test Set
This script evaluates the performance of the RAG system using a JSONL test set. Each entry in the test set should contain a query, the expected target section, and relevant keywords. The evaluation metrics include Mean Reciprocal Rank (MRR), Hit Rate, and Keyword Coverage.
The script performs the following steps:
1. Loads the test set from a JSONL file.
2. For each query, it retrieves relevant sections using the RagService.
3. Calculates MRR based on the rank of the correct section in the retrieved results.
4. Computes Keyword Coverage by checking how many of the expected keywords are present in the retrieved text.
5. Summarizes the results with average MRR, Hit Rate, and Keyword Coverage.
Make sure to have the RAG system running and the ChromaDB vector store set up before executing this evaluation script.
"""


import json
import logging
from datetime import datetime
from typing import List, Dict
from rag_engine.services import RagService

class JSONLRetrievalEvaluator:
    """Evaluator for RAG retrieval performance with Markdown export."""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.rag_service = RagService()
        self.detailed_logs = [] # To store data for the report

    def load_test_set(self) -> List[Dict]:
        test_set = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                test_set.append(json.loads(line))
        return test_set

    def run_eval(self):
        test_set = self.load_test_set()
        results = []
        self.detailed_logs = [] 

        print(f"📊 Evaluating {len(test_set)} cases from {self.data_path}...")

        for test in test_set:
            response = self.rag_service.query(test["query"])
            retrieved_docs = response.get("source_documents", [])
            retrieved_sections = [doc.metadata.get("section") for doc in retrieved_docs]
            
            print(f"DEBUG: Query: {test['query'][:30]}... | Target: '{test['target']}' | Found: {retrieved_sections}")
            
            retrieved_text = " ".join([doc.page_content.lower() for doc in retrieved_docs])

            # MRR Calculation
            mrr = 0.0
            for i, section in enumerate(retrieved_sections):
                if section == test["target"]:
                    mrr = 1 / (i + 1)
                    break

            # Keyword Coverage
            found = [k for k in test["keywords"] if k.lower() in retrieved_text]
            coverage = len(found) / len(test["keywords"])

            res_entry = {
                "mrr": mrr,
                "coverage": coverage,
                "hit": 1.0 if mrr > 0 else 0.0
            }
            results.append(res_entry)
            
            # Save info for Markdown table
            self.detailed_logs.append({
                "query": test["query"],
                "target": test["target"],
                "found": ", ".join(retrieved_sections) if retrieved_sections else "None",
                "mrr": mrr,
                "success": "✅" if mrr > 0 else "❌"
            })

        summary = self._summary(results)
        self._export_markdown(summary)

    def _summary(self, results):
        avg_mrr = sum(r['mrr'] for r in results) / len(results)
        avg_hit = sum(r['hit'] for r in results) / len(results)
        avg_cov = sum(r['coverage'] for r in results) / len(results)

        print("\n" + "="*40)
        print(f"FINAL DETERMINISTIC EVALUATION")
        print("="*40)
        print(f"Mean Reciprocal Rank (MRR): {avg_mrr:.2f}")
        print(f"Hit Rate (Top-K): {avg_hit:.0%}")
        print(f"Avg Keyword Coverage: {avg_cov:.0%}")
        print("="*40)
        
        return {"mrr": avg_mrr, "hit": avg_hit, "cov": avg_cov}

    def _export_markdown(self, summary):
        """Generates a clean Markdown report file."""
        filename = "eval_report.md"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(filename, "w", encoding='utf-8') as f:
            f.write(f"# 🇳🇬 RAG Evaluation Report\n")
            f.write(f"*Generated on: {timestamp}*\n\n")
            
            f.write("## 📊 Summary Metrics\n")
            f.write(f"| Metric | Value |\n")
            f.write(f"| :--- | :--- |\n")
            f.write(f"| **Mean Reciprocal Rank (MRR)** | {summary['mrr']:.2f} |\n")
            f.write(f"| **Hit Rate (Top-K)** | {summary['hit']:.0%} |\n")
            f.write(f"| **Avg Keyword Coverage** | {summary['cov']:.0%} |\n\n")
            
            f.write("## 📝 Detailed Logs\n")
            f.write(f"| Status | Query | Target | Found Sections | MRR |\n")
            f.write(f"| :--- | :--- | :--- | :--- | :--- |\n")
            
            for log in self.detailed_logs:
                f.write(f"| {log['success']} | {log['query']} | **{log['target']}** | {log['found']} | {log['mrr']:.2f} |\n")
        
        print(f"✅ Markdown report generated: {filename}")

if __name__ == "__main__":
    evaluator = JSONLRetrievalEvaluator("evaluation_set.jsonl")
    evaluator.run_eval()

