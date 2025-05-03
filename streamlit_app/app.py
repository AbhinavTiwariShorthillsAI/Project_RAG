import os
import json
import numpy as np
import requests
import weaviate
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer, CrossEncoder
import streamlit as st

HISTORY_FILE = "data/conversation_history.json"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"

class WeaviateRetriever:
    """
    Retrieves relevant documents from Weaviate vector store based on semantic similarity.
    
    Attributes:
        client (weaviate.Client): Weaviate client for database operations.
        embed_model (SentenceTransformer): Model for creating text embeddings.
    """

    def __init__(self, weaviate_url: str = "http://localhost:8080", embedding_model: str = "intfloat/e5-base-v2") -> None:
        """
        Initialize the retriever with Weaviate connection and embedding model.
        
        Args:
            weaviate_url (str, optional): URL to the Weaviate instance. Defaults to "http://localhost:8080".
            embedding_model (str, optional): Name of the SentenceTransformer model. Defaults to "intfloat/e5-base-v2".
            
        Raises:
            ConnectionError: If connection to Weaviate fails.
            RuntimeError: If loading the embedding model fails.
        """
        try:
            self.client = weaviate.Client(url=weaviate_url)
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Weaviate at {weaviate_url}: {e}")
        try:
            self.embed_model = SentenceTransformer(embedding_model)
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model '{embedding_model}': {e}")

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """
        Retrieve the most semantically similar text chunks for a given query.
        
        Args:
            query (str): The search query or question.
            top_k (int, optional): Number of chunks to retrieve. Defaults to 5.
            
        Returns:
            List[str]: List of text chunks retrieved from Weaviate.
            
        Note:
            Returns error message as a single list item if retrieval fails.
        """
        try:
            query_embedding = self.embed_model.encode([query])
            query_embedding = np.array(query_embedding).astype(np.float32)
            response = self.client.query.get("WorldWarChunk", ["text"])
            response = response.with_near_vector({"vector": query_embedding[0]}).with_limit(top_k).do()
            return [item["text"] for item in response["data"]["Get"]["WorldWarChunk"]]
        except Exception as e:
            return [f"Error retrieving from Weaviate: {e}"]

class Reranker:
    """
    Reranks retrieved documents using a cross-encoder model to improve relevance.
    
    Attributes:
        model (CrossEncoder): Cross-encoder model for document reranking.
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        """
        Initialize the reranker with a cross-encoder model.
        
        Args:
            model_name (str, optional): Name of the cross-encoder model. 
                Defaults to "cross-encoder/ms-marco-MiniLM-L-6-v2".
                
        Raises:
            RuntimeError: If loading the reranker model fails.
        """
        try:
            self.model = CrossEncoder(model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load reranker model '{model_name}': {e}")

    def rerank(self, query: str, chunks: List[str], top_k: int = 3) -> List[str]:
        """
        Reranks text chunks based on their relevance to the query.
        
        Args:
            query (str): The search query or question.
            chunks (List[str]): List of text chunks to rerank.
            top_k (int, optional): Number of top chunks to return. Defaults to 3.
            
        Returns:
            List[str]: Reranked list of text chunks, limited to top_k.
            
        Note:
            Returns empty list if chunks is empty or error message as a single list item if reranking fails.
        """
        try:
            if not chunks:
                return []
            pairs = [(query, chunk) for chunk in chunks]
            scores = self.model.predict(pairs)
            reranked = [chunk for _, chunk in sorted(zip(scores, chunks), reverse=True)]
            return reranked[:top_k]
        except Exception as e:
            return [f"Error during reranking: {e}"]

class LlamaQuerier:
    """
    Sends queries to a local LLaMA model via Ollama API.
    
    Attributes:
        model (str): Name of the LLaMA model.
        url (str): URL endpoint for the Ollama API.
    """
    
    def __init__(self, model: str = OLLAMA_MODEL, url: str = OLLAMA_URL) -> None:
        """
        Initialize the LLaMA querier with model name and API endpoint.
        
        Args:
            model (str, optional): Name of the LLaMA model. Defaults to OLLAMA_MODEL.
            url (str, optional): Ollama API endpoint. Defaults to OLLAMA_URL.
        """
        self.model = model
        self.url = url

    def query(self, prompt: str) -> str:
        """
        Send a prompt to the LLaMA model via Ollama API and get the response.
        
        Args:
            prompt (str): The formatted prompt to send to the model.
            
        Returns:
            str: The model's response or error message if the query fails.
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
    """
    Streamlit application for question answering with conversation history.
    
    Attributes:
        retriever (WeaviateRetriever): For retrieving relevant documents from Weaviate.
        reranker (Reranker): For reranking retrieved documents by relevance.
        llm (LlamaQuerier): For querying the LLaMA model.
        history (List[Dict[str, str]]): Conversation history with questions and answers.
    """
    
    def __init__(self) -> None:
        """
        Initialize the QA application with retriever, reranker, LLM, and conversation history.
        """
        self.retriever = WeaviateRetriever()
        self.reranker = Reranker()
        self.llm = LlamaQuerier()
        self.history = self.load_history()

    def load_history(self) -> List[Dict[str, str]]:
        """
        Load conversation history from a JSON file.
        
        Returns:
            List[Dict[str, str]]: List of conversation entries, each with 'question' and 'answer' keys.
            
        Note:
            Returns empty list if file doesn't exist or loading fails.
        """
        try:
            if os.path.exists(HISTORY_FILE):
                with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            st.error(f"Failed to load history: {e}")
        return []

    def save_history(self) -> None:
        """
        Save the current conversation history to a JSON file.
        
        Note:
            Displays error in Streamlit UI if saving fails.
        """
        try:
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            st.error(f"Failed to save history: {e}")

    def run(self) -> None:
        """
        Run the Streamlit application with the RAG Q&A interface.
        
        Builds and displays:
            - Custom CSS styling
            - Application title
            - Question input area
            - Answer display
            - Conversation history sidebar
        """
        st.markdown("""
            <style>
                textarea {
                overflow-y: auto !important;
                overflow-x: hidden !important;
                resize: none !important;
                height: 30px !important;
                font-size: 16px;
                font-family: 'Monolisa', monospace;
                }
            </style>
        """, unsafe_allow_html=True)

        st.title(f"World War History Q&A (Powered by Local {OLLAMA_MODEL.capitalize()} 🦙)")

        if "history" not in st.session_state:
            st.session_state.history = self.history

        if "question" not in st.session_state:
            st.session_state.question = ""

        st.sidebar.title("📜 Conversation History")

        if st.sidebar.button("⬇️ Download History"):
            history_str = json.dumps(st.session_state.history, indent=2)
            st.sidebar.download_button("Download", data=history_str, file_name="conversation_history.json")

        question = st.text_area("Ask your question about World War History:", height=100)
        if question:
            try:
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
            except Exception as e:
                answer = f"Unexpected error: {e}"

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
