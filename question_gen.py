import os
import requests
import pandas as pd
from dotenv import load_dotenv
from time import sleep

load_dotenv()

# Constants
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# Read your document
with open("modern_history_combined.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Split into chunks
def split_text(text, chunk_size=800):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

chunks = split_text(text)

questions = []
answers = []

# Query LLaMA via Ollama
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
        return f"Error: {e}"

# Total for progress
total = min(500, len(chunks))  # Adjustable for 1000 Q&A
print("🚀 Generating Q&A...")

for i, chunk in enumerate(chunks[:total]):
    prompt = f"""
Based only on the following historical text, generate 2 simple factual questions and their short, correct answers. 
Output format strictly like:
Question: <your question>
Answer: <your answer>

Historical text:
{chunk}
"""

    response_text = query_llama(prompt)
    output_lines = response_text.strip().split("\n")

    current_question = ""
    current_answer = ""

    for line in output_lines:
        if "Question:" in line:
            if current_question and current_answer:
                questions.append(current_question)
                answers.append(current_answer)
                current_question, current_answer = "", ""
            current_question = line.split(":", 1)[1].strip()

        elif "Answer:" in line:
            current_answer = line.split(":", 1)[1].strip()

    # Capture last pair
    if current_question and current_answer:
        questions.append(current_question)
        answers.append(current_answer)

    # ✅ Show progress
    percent = int(((i + 1) / total) * 100)
    print(f"Progress: {percent}% ({i+1}/{total})", end='\r')  # overwrite last line

    # Optional sleep (to avoid overload)
    sleep(0.2)

print("\n✅ Q&A generation complete!")

# Save to CSV
qa_df = pd.DataFrame({"Question": questions, "Answer": answers})
qa_df.to_csv("qa_dataset_1000.csv", index=False)

print(f"✅ Saved {len(qa_df)} Q&A pairs to qa_dataset_1000.csv")
