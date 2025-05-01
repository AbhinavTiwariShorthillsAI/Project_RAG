import os
import requests
import weaviate
import numpy as np
import csv
from openpyxl import Workbook, load_workbook
from sentence_transformers import SentenceTransformer
from typing import Generator, Dict, List

# Constants
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"
DATASET_PATH = "data/qa_dataset_1000_part2.csv"
OUTPUT_XLSX = "data/qa_with_predictions_part2.xlsx"

class RAGQuestionAnswering:
    """
    A class to perform retrieval-augmented generation (RAG) on a QA dataset,
    fetching context from Weaviate and generating answers using a local LLaMA model.
    """

    def __init__(self):
        self.client = weaviate.Client("http://localhost:8080")
        self.embed_model = SentenceTransformer("intfloat/e5-base-v2")

    def search_weaviate(self, query: str, top_k: int = 3) -> List[str]:
        """
        Retrieves top_k relevant text chunks from Weaviate for a given query.
        """
        query_embedding = self.embed_model.encode([query])
        query_embedding = np.array(query_embedding).astype(np.float32)

        response = self.client.query.get("WorldWarChunk", ["text"])\
            .with_near_vector({"vector": query_embedding[0]})\
            .with_limit(top_k).do()

        return [item["text"] for item in response["data"]["Get"]["WorldWarChunk"]]

    def query_llama(self, prompt: str) -> str:
        """
        Sends a prompt to the local LLaMA model via Ollama API and returns the response.
        """
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        }
        try:
            response = requests.post(OLLAMA_URL, json=payload)
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except Exception as e:
            return f"Error querying {OLLAMA_MODEL}: {e}"

    def question_fetch(self, csv_file_path: str) -> Generator[Dict[str, str], None, None]:
        """
        Yields question-answer pairs from a CSV file with two columns: question and answer.
        """
        with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            next(reader, None)  # Skip header
            for row in reader:
                if row and len(row) >= 2:
                    yield {"question": row[0], "answer": row[1]}

    def run(self):
        """
        Executes the full RAG pipeline: fetch context, generate answer, write to Excel.
        """
        print(f"\n🤖 World War History Q&A (Powered by {OLLAMA_MODEL})")

        if os.path.exists(OUTPUT_XLSX):
            wb = load_workbook(OUTPUT_XLSX)
            ws = wb.active
        else:
            wb = Workbook()
            ws = wb.active
            ws.append(["Question", "Answer", "predicted_answer", "context"])

        for item in self.question_fetch(DATASET_PATH):
            question = item["question"]
            ground_truth = item["answer"]

            if not question.strip():
                continue

            print(f"\n❓ Question: {question}")

            context_chunks = self.search_weaviate(question)
            context = "\n".join(context_chunks)[:1500]

            prompt = f"""
You are a factual Q&A assistant based on historical documents.

Instructions:
Give answers accurately and concisely based on the provided context.
Answers should be short and to the point.
Do not include any disclaimers or unnecessary information.
Do not make up any information.
Do not include any references to the source of the information.
Do not include any additional context or information outside of the provided context.

Context:
{context}

Question: {question}
Answer:
"""

            predicted_answer = self.query_llama(prompt)
            print(f"\n💬 Answer: {predicted_answer}\n")

            ws.append([question, ground_truth, predicted_answer, context])
            wb.save(OUTPUT_XLSX)

        print(f"\n✅ All predictions saved to: {OUTPUT_XLSX}")


if __name__ == "__main__":
    rag_qa = RAGQuestionAnswering()
    rag_qa.run()
