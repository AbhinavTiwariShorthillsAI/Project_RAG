import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import weaviate
import json
import os
import requests

from dotenv import load_dotenv
load_dotenv()

# Constants (reading from .env)
HISTORY_FILE = os.getenv("HISTORY_FILE", "conversation_history.json")
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# Streamlit title
st.title(f"World War History Q&A (Powered by Local {OLLAMA_MODEL.capitalize()} 🦙)")

# Connect to Weaviate
client = weaviate.Client(url="http://localhost:8080")

# Load embedding model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to search inside Weaviate
def search_weaviate(query, top_k=5):
    query_embedding = embed_model.encode([query])
    query_embedding = np.array(query_embedding).astype(np.float32)

    response = client.query.get(
        "WorldWarChunk", ["text"]
    ).with_near_vector({
        "vector": query_embedding[0]
    }).with_limit(top_k).do()

    results = []
    for item in response["data"]["Get"]["WorldWarChunk"]:
        results.append(item["text"])

    return results

# Function to load conversation history
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        return []

# Function to save conversation history
def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

# Function to query local LLM via Ollama
def query_llama(prompt):
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

# Initialize conversation history
if "history" not in st.session_state:
    st.session_state.history = load_history()

# Streamlit input
question = st.text_input("Ask your question about World War History:")

if question:
    relevant_chunks = search_weaviate(question, top_k=3)
    full_context = "\n".join(relevant_chunks)[:1500]

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

    answer = query_llama(prompt)

    # Save the Q&A into session state
    st.session_state.history.append({"question": question, "answer": answer})
    save_history(st.session_state.history)

    # Display latest answer
    st.subheader("Answer:")
    st.write(answer)

# Show conversation history
if st.session_state.history:
    st.sidebar.title("Conversation History")
    for idx, entry in enumerate(st.session_state.history):
        st.sidebar.markdown(f"**Q{idx+1}:** {entry['question']}")
        st.sidebar.markdown(f"**A{idx+1}:** {entry['answer']}")
        st.sidebar.markdown("---")
