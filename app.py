import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import google.generativeai as genai
import weaviate
import json
from dotenv import load_dotenv
import os

load_dotenv()  # take environment variables from .env.

API_KEY = os.getenv("API_KEY")

# Constants
HISTORY_FILE = "/home/shtlp_0012/codes/RAG/conversation_history.json"

# Streamlit title
st.title("World War History Q&A")

# Initialize Gemini API
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

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

# Function to load conversation history from JSON file
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        return []

# Function to save conversation history to JSON file
def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

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
- Answer questions **accurately and concisely** based only on the provided text.
- If the answer is a **date, name, event**, or **simple fact**, respond with **just that** unless a full sentence is absolutely necessary.
- If a question contains a **wrong assumption**, **correct it politely** first, then provide the correct information.
- If the answer is **not found in the provided text**, respond clearly: **"Answer not found in the text."**
- Do not invent or assume any information beyond the given text.

Context:
{full_context}

Question:
{question}

Answer:
"""

    response = model.generate_content(prompt)
    answer = response.text

    # Save the Q&A into session state 
    st.session_state.history.append({"question": question, "answer": answer})
    save_history(st.session_state.history)  # Also save to local file immediately

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
