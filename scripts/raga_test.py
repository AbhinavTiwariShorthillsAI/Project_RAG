import pandas as pd
import json
from tqdm import tqdm
import google.generativeai as genai
from dotenv import load_dotenv
import re
import os

# === CONFIG ===
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EXCEL_PATH = "data/qa_with_predictions_part2.xlsx"
OUTPUT_XLSX = "data/evaluated_results_partial.xlsx"
OUTPUT_JSON = "data/rag_evaluation_scores_partial.json"
API_KEY = GEMINI_API_KEY  # <-- Replace this

# Index range for partial evaluation
start_index = 977  # <-- CHANGE THIS
end_index = 985    # <-- CHANGE THIS

# === Setup Gemini ===
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("models/gemini-2.0-flash")

# === Load Input Data ===
df = pd.read_excel(EXCEL_PATH)
df_slice = df.iloc[start_index:end_index].copy()

# === Prompt Template ===
def build_prompt(question: str, answer: str, predicted_answer: str, context: str) -> str:
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

# === Score Extraction ===
def extract_scores(response_text: str) -> list[float]:
    match = re.search(r"\[\s*(\d\.\d+)\s*,\s*(\d\.\d+)\s*,\s*(\d\.\d+)\s*,\s*(\d\.\d+)\s*\]", response_text)
    if match:
        return [float(match.group(i)) for i in range(1, 5)]
    return [None, None, None, None]

# === Evaluation Loop ===
all_scores = []
for idx, row in tqdm(df_slice.iterrows(), total=len(df_slice), desc="Evaluating"):
    q, a, p, c = row["Question"], row["Answer"], row["predicted_answer"], row["context"]
    prompt = build_prompt(q, a, p, c)
    try:
        response = model.generate_content(prompt)
        scores = extract_scores(response.text)
    except Exception as e:
        print(f"Error on row {idx + start_index}: {e}")
        scores = [None, None, None, None]

    print(f"Row {idx + start_index} scores: {scores}")
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

    print("\n✅ Updated RAG Evaluation Scores:")
    print(json.dumps(final_avg, indent=2))
else:
    print("⚠️ No valid scores to average.")



