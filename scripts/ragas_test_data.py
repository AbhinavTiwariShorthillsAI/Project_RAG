import csv
import json

# === CONFIG ===
INPUT_CSV_FILE = "data/evaluation_scores_split_mistral.csv"  # Path to your CSV
OUTPUT_JSON_FILE = "data/evaluation_summary_ragas_mistral.json"    # Output summary JSON

# === Initialize ===
scores = {
    "faithfulness": [],
    "answer_relevancy": [],
    "context_precision": [],
    "context_recall": []
}

# === Read CSV and collect scores ===
with open(INPUT_CSV_FILE, mode="r") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        try:
            scores["faithfulness"].append(float(row["faithfulness"]))
            scores["answer_relevancy"].append(float(row["answer_relevancy"]))
            scores["context_precision"].append(float(row["context_precision"]))
            scores["context_recall"].append(float(row["context_recall"]))
        except ValueError as e:
            print(f"Skipping row due to error: {e} | Row: {row}")

# === Compute averages ===
summary = {
    "faithfulness": round(sum(scores["faithfulness"]) / len(scores["faithfulness"]), 4),
    "answer_relevancy": round(sum(scores["answer_relevancy"]) / len(scores["answer_relevancy"]), 4),
    "context_precision": round(sum(scores["context_precision"]) / len(scores["context_precision"]), 4),
    "context_recall": round(sum(scores["context_recall"]) / len(scores["context_recall"]), 4)
}

# === Dump to JSON file ===
with open(OUTPUT_JSON_FILE, "w") as jsonfile:
    json.dump(summary, jsonfile, indent=2)

print(f"✅ Evaluation summary saved to: {OUTPUT_JSON_FILE}")
print(json.dumps(summary, indent=2))
