import csv
import json
import os
import logging
from typing import Dict, List, Any

# Set up logging
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, 'output')
os.makedirs(output_dir, exist_ok=True)
log_file = os.path.join(output_dir, 'ragas_test_data.log')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def read_scores_from_csv(csv_file: str) -> Dict[str, List[float]]:
    """
    Read evaluation scores from a CSV file and organize them by metric.
    
    Args:
        csv_file (str): Path to the CSV file containing evaluation scores
        
    Returns:
        Dict[str, List[float]]: Dictionary containing lists of scores for each metric
            Keys are metric names (faithfulness, answer_relevancy, etc.)
            Values are lists of float scores
            
    Raises:
        FileNotFoundError: If the specified CSV file does not exist
        ValueError: If any score cannot be converted to float
    """
    scores = {
        "faithfulness": [],
        "answer_relevancy": [],
        "context_precision": [],
        "context_recall": []
    }
    
    with open(csv_file, mode="r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                scores["faithfulness"].append(float(row["faithfulness"]))
                scores["answer_relevancy"].append(float(row["answer_relevancy"]))
                scores["context_precision"].append(float(row["context_precision"]))
                scores["context_recall"].append(float(row["context_recall"]))
            except ValueError as e:
                logger.warning(f"Skipping row due to error: {e} | Row: {row}")
    
    return scores

def compute_averages(scores: Dict[str, List[float]]) -> Dict[str, float]:
    """
    Calculate the average score for each metric.
    
    Args:
        scores (Dict[str, List[float]]): Dictionary containing lists of scores for each metric
        
    Returns:
        Dict[str, float]: Dictionary containing average scores for each metric
            Keys are metric names
            Values are average scores rounded to 4 decimal places
            
    Raises:
        ZeroDivisionError: If any metric has no scores
    """
    return {
        metric: round(sum(score_list) / len(score_list), 4)
        for metric, score_list in scores.items()
    }

def save_summary_to_json(summary: Dict[str, float], output_file: str) -> None:
    """
    Save the evaluation summary to a JSON file.
    
    Args:
        summary (Dict[str, float]): Dictionary containing average scores for each metric
        output_file (str): Path where the JSON file should be saved
        
    Returns:
        None
        
    Raises:
        IOError: If the file cannot be written
    """
    with open(output_file, "w") as jsonfile:
        json.dump(summary, jsonfile, indent=2)
    logger.info(f"✅ Evaluation summary saved to: {output_file}")
    logger.info(json.dumps(summary, indent=2))

def main():
    """
    Main function to process evaluation scores and generate a summary.
    
    This function:
    1. Reads scores from a CSV file
    2. Computes averages for each metric
    3. Saves the summary to a JSON file
    """
    # Get the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Configuration
    INPUT_CSV_FILE = os.path.join(project_root, "data", "evaluation", "evaluation_scores_split_mistral.csv")
    OUTPUT_JSON_FILE = os.path.join(project_root, "data", "evaluation", "evaluation_summary_ragas_mistral.json")
    
    # Process scores
    scores = read_scores_from_csv(INPUT_CSV_FILE)
    summary = compute_averages(scores)
    save_summary_to_json(summary, OUTPUT_JSON_FILE)

if __name__ == "__main__":
    main()
