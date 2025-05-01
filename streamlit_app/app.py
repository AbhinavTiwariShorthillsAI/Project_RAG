import os
import json
import numpy as np
import requests
import weaviate
from typing import List
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
import streamlit as st
from io import StringIO

load_dotenv()

HISTORY_FILE = os.getenv("HISTORY_FILE", "data/conversation_history.json")
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

class WeaviateRetriever:
    """Retrieves relevant documents from Weaviate vector store based on semantic similarity."""

    def __init__(self, weaviate_url: str = "http://localhost:8080", embedding_model: str = "intfloat/e5-base-v2"):
        """
        Initializes the Weaviate client and embedding model.

        Args:
            weaviate_url (str): URL to the Weaviate instance.
            embedding_model (str): HuggingFace sentence transformer model name.
        """
        self.client = weaviate.Client(url=weaviate_url)
        self.embed_model = SentenceTransformer(embedding_model)

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """
        Retrieves top_k text chunks most relevant to the query.

        Args:
            query (str): User's question.
            top_k (int): Number of top results to retrieve.

        Returns:
            List[str]: List of relevant text chunks.
        """
        query_embedding = self.embed_model.encode([query])
        query_embedding = np.array(query_embedding).astype(np.float32)
        response = self.client.query.get("WorldWarChunk", ["text"])
        response = response.with_near_vector({"vector": query_embedding[0]}).with_limit(top_k).do()
        return [item["text"] for item in response["data"]["Get"]["WorldWarChunk"]]

class Reranker:
    """Reranks retrieved chunks using a cross-encoder model."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initializes the reranker model.

        Args:
            model_name (str): HuggingFace cross-encoder model name.
        """
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, chunks: List[str], top_k: int = 3) -> List[str]:
        """
        Scores and reranks the text chunks based on the input query.

        Args:
            query (str): User's question.
            chunks (List[str]): List of text chunks.
            top_k (int): Number of top chunks to return.

        Returns:
            List[str]: Reranked list of top_k text chunks.
        """
        if not chunks:
            return []
        pairs = [(query, chunk) for chunk in chunks]
        scores = self.model.predict(pairs)
        reranked = [chunk for _, chunk in sorted(zip(scores, chunks), reverse=True)]
        return reranked[:top_k]

class LlamaQuerier:
    """Handles querying the local LLaMA model via Ollama API."""

    def __init__(self, model: str = OLLAMA_MODEL, url: str = OLLAMA_URL):
        """
        Initializes the query interface to Ollama LLaMA model.

        Args:
            model (str): Name of the model registered in Ollama.
            url (str): API endpoint URL for Ollama.
        """
        self.model = model
        self.url = url

    def query(self, prompt: str) -> str:
        """
        Sends a prompt to the LLaMA model and retrieves a response.

        Args:
            prompt (str): Prompt text to send to the LLM.

        Returns:
            str: Model-generated answer.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        try:
            response = requests.post(self.url, json=payload)
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except Exception as e:
            return f"Error querying {self.model}: {e}"

class QAApp:
    """Streamlit-based front-end for RAG-style question answering."""

    def __init__(self):
        self.retriever = WeaviateRetriever()
        self.reranker = Reranker()
        self.llm = LlamaQuerier()
        self.history = self.load_history()

    def load_history(self):
        """Loads conversation history from the local history file."""
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def save_history(self):
        """Saves the conversation history to disk."""
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

    def run(self):
        """Launches the Streamlit UI and handles user interactions."""
        st.markdown("""
            <style>
                .stTextInput > div > div > input {
                    font-size: 18px;
                    padding: 0.5em;
                    font-family: 'Segoe UI', sans-serif;
                }
                .stMarkdown h1 {
                    text-align: center;
                    font-size: 2em;
                    font-family: 'Georgia', serif;
                }
                .expander-header p {
                    margin: 0;
                    font-weight: bold;
                    font-family: 'Segoe UI', sans-serif;
                }
            </style>
        """, unsafe_allow_html=True)

        st.title(f"World War History Q&A (Powered by Local {OLLAMA_MODEL.capitalize()} 🦙)")

        if "history" not in st.session_state:
            st.session_state.history = self.history

        st.sidebar.title("📜 Conversation History")
        if st.sidebar.button("🧹 Clear History"):
            st.session_state.history = []
            self.history = []
            self.save_history()
            st.experimental_rerun()

        if st.sidebar.button("⬇️ Download History"):
            history_str = json.dumps(st.session_state.history, indent=2)
            st.sidebar.download_button("Download", data=history_str, file_name="conversation_history.json")

        question = st.text_area("Ask your question about World War History:", height=100)
        if question:
            retrieved_chunks = self.retriever.retrieve(question, top_k=10)
            reranked_context = self.reranker.rerank(question, retrieved_chunks, top_k=5)
            full_context = "\n".join(reranked_context)

            prompt = f"""
You are a factual Q&A assistant based on historical documents.

Instructions:
- Answer questions accurately and concisely using only the provided context.
- If the answer is a date, name, or simple fact, respond with just that, avoiding full sentences unless necessary.
- If the question contains a wrong assumption, correct it politely and provide the correct answer.
- If the answer is not found in the text, respond with: "Answer not found in the text."
- Do not make up information beyond what is given.

Context:
{full_context}

Question: {question}
Answer:
"""
            answer = self.llm.query(prompt)
            st.session_state.history.append({"question": question, "answer": answer})
            self.save_history()

            st.markdown("**Answer:**")
            st.markdown(f"<div style='padding: 0.75em; background-color: #f4f4f4; border-radius: 6px;'>{answer}</div>", unsafe_allow_html=True)

        if st.session_state.history:
            for idx, entry in reversed(list(enumerate(st.session_state.history))):
                with st.sidebar.expander(f"Q{idx+1}: {entry['question'][:60]}..."):
                    st.markdown(f"**Q:** {entry['question']}")
                    st.markdown(f"**A:** {entry['answer']}")

if __name__ == "__main__":
    app = QAApp()
    app.run()
