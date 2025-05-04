import pandas as pd
import json
from tqdm import tqdm
import google.generativeai as genai
from dotenv import load_dotenv
import re
import os
import logging
from typing import List, Tuple, Dict, Optional, Union, Any

# Set up logging
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, 'output')
os.makedirs(output_dir, exist_ok=True)
log_file = os.path.join(output_dir, 'raga_test.log')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Get the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# === CONFIG ===
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EXCEL_PATH = os.path.join(project_root, "data", "processed", "qa_with_predictions.xlsx")
OUTPUT_XLSX = os.path.join(project_root, "data", "evaluation", "evaluated_results_partial_mistral.xlsx")
OUTPUT_JSON = os.path.join(project_root, "data", "evaluation", "rag_evaluation_scores_partial_mistral.json")
API_KEY = GEMINI_API_KEY 

# Index range for partial evaluation
start_index = 649  # <-- CHANGE THIS
end_index = start_index + 15

# === Setup Gemini ===
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("models/gemini-2.0-flash")

# === Load Input Data ===
df = pd.read_excel(EXCEL_PATH)
df_slice = df.iloc[start_index:end_index].copy()

def build_prompt(question: str, answer: str, predicted_answer: str, context: str) -> str:
    """
    Build an evaluation prompt for the Gemini model to evaluate RAG performance.
    
    Args:
        question (str): The original question.
        answer (str): The ground truth answer.
        predicted_answer (str): The RAG system's predicted answer.
        context (str): The context retrieved and used for generating the answer.
        
    Returns:
        str: A formatted prompt for the Gemini model to evaluate the RAG system.
    """
    return f"""
Evaluate the following Q&A pair using the provided context and return a list of 4 scores only, no explanation.

Each score must be a float between 0 and 1, rounded to 5 decimal places. The 4 scores are:
1. Faithfulness: Is the predicted answer supported by the context?
2. Answer Relevancy: Does the predicted answer directly answer the question?
3. Context Precision: How much of the context is relevant to the answer?
4. Context Recall: How much of the ground truth answer is covered by the context?

Return ONLY a Python-style list of 4 floats like this: [0.95521, 0.85312, 0.91230, 0.85428]

Question: {question}
Ground Truth Answer: {answer}
Predicted Answer: {predicted_answer}
Context: {context}
"""

def extract_scores(response_text: str) -> List[Optional[float]]:
    """
    Extract the four evaluation scores from the Gemini model's response.
    
    Args:
        response_text (str): The raw text response from the Gemini model.
        
    Returns:
        List[Optional[float]]: A list of 4 floats representing the evaluation scores,
                              or a list with None values if extraction fails.
    """
    match = re.search(r"\[\s*(\d\.\d+)\s*,\s*(\d\.\d+)\s*,\s*(\d\.\d+)\s*,\s*(\d\.\d+)\s*\]", response_text)
    if match:
        return [float(match.group(i)) for i in range(1, 5)]
    return [None, None, None, None]

# === Evaluation Loop ===
all_scores: List[List[Optional[float]]] = []
for idx, row in tqdm(df_slice.iterrows(), total=len(df_slice), desc="Evaluating"):
    q, a, p, c = row["Question"], row["Answer"], row["predicted_answer"], row["context"]
    prompt = build_prompt(q, a, p, c)
    try:
        response = model.generate_content(prompt)
        scores = extract_scores(response.text)
    except Exception as e:
        logger.error(f"Error on row {idx + start_index}: {e}")
        scores = [None, None, None, None]

    logger.info(f"Row {idx + start_index} scores: {scores}")
    all_scores.append(scores)

# === Add Scores to Slice ===
df_slice["evaluation_scores"] = all_scores

# === Append to Excel File ===
if os.path.exists(OUTPUT_XLSX):
    existing_df = pd.read_excel(OUTPUT_XLSX)
    combined_df = pd.concat([existing_df, df_slice], ignore_index=True)
else:
    combined_df = df_slice
combined_df.to_excel(OUTPUT_XLSX, index=False)

# === Update JSON File ===
filtered_scores = [s for s in all_scores if None not in s]

if filtered_scores:
    # New scores
    new_count = len(filtered_scores)
    new_sums = [sum(s[i] for s in filtered_scores) for i in range(4)]

    # Existing scores (if any)
    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON, "r") as f:
            existing_avg = json.load(f)
        old_scores = [
            existing_avg["faithfulness"],
            existing_avg["answer_relevancy"],
            existing_avg["context_precision"],
            existing_avg["context_recall"],
        ]
        old_count = len(pd.read_excel(OUTPUT_XLSX)) - new_count
        old_sums = [old_scores[i] * old_count for i in range(4)]
    else:
        old_sums = [0.0] * 4
        old_count = 0

    # Combine
    total_count = new_count + old_count
    final_avg = {
        "faithfulness": round((old_sums[0] + new_sums[0]) / total_count, 4),
        "answer_relevancy": round((old_sums[1] + new_sums[1]) / total_count, 4),
        "context_precision": round((old_sums[2] + new_sums[2]) / total_count, 4),
        "context_recall": round((old_sums[3] + new_sums[3]) / total_count, 4),
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(final_avg, f, indent=2)

    logger.info("\n✅ Updated RAG Evaluation Scores:")
    logger.info(json.dumps(final_avg, indent=2))
else:
    logger.warning("⚠️ No valid scores to average.")


