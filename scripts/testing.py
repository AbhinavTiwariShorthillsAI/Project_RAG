import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
from typing import Dict, Any, Optional, List, Union
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score
from difflib import SequenceMatcher

# Set up logging
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, 'output')
os.makedirs(output_dir, exist_ok=True)
log_file = os.path.join(output_dir, 'testing.log')

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

# Config paths
INPUT_FILE = os.path.join(project_root, "data", "processed", "qa_with_predictions.xlsx")
OUTPUT_FILE = os.path.join(project_root, "data", "processed", "qa_evaluated_scores.xlsx")
SUMMARY_FILE = os.path.join(project_root, "data", "evaluation", "evaluation_summary_mistral.json")

# Metric Weights (can be adjusted to tune scoring priorities)
METRIC_WEIGHTS = {
    "rouge_score": 0.15,  # Good for overlap in long texts
    "cosine_similarity": 0.35,  # High priority for retrieval accuracy
    "bert_score_f1": 0.30,  # Essential for semantic accuracy
    "bleu": 0.05,  # Less important for factual Q&A tasks
    "meteor": 0.10,  # Useful for paraphrasing and synonym matching
    "levenshtein": 0.05  # Low importance for historical fact accuracy
}

class QAEvaluator:
    """
    A comprehensive evaluator for comparing predicted and reference answers using both
    traditional NLP metrics and embedding-based similarity scores.

    Metrics used:
    - Cosine Similarity (from sentence embeddings)
    - ROUGE-L Score
    - BERTScore F1
    - BLEU Score
    - METEOR Score
    - Levenshtein Ratio

    Output includes both detailed per-row metrics and an aggregate summary.
    """

    def __init__(self):
        """
        Initializes the sentence embedding model and other scorers.
        
        Raises:
            ImportError: If required libraries are not installed
        """
        try:
            self.similarity_model = SentenceTransformer("intfloat/e5-base-v2")
            self.rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            self.smooth_fn = SmoothingFunction().method1
        except Exception as e:
            logger.error(f"Error initializing QAEvaluator: {e}")
            raise

    def calculate_metrics(self, generated: str, reference: str) -> Dict[str, float]:
        """
        Compute all supported similarity and evaluation metrics for one Q-A pair.

        Args:
            generated (str): The model-generated/predicted answer.
            reference (str): The correct/expected answer.

        Returns:
            Dict[str, float]: A dictionary of all individual metric scores and the
                              overall weighted final score.
                              
        Raises:
            ValueError: If inputs are empty or invalid
        """
        if not generated or not reference:
            raise ValueError("Generated and reference texts cannot be empty")
            
        try:
            # Embedding-based cosine similarity
            emb_gen = self.similarity_model.encode(generated)
            emb_ref = self.similarity_model.encode(reference)
            cosine_sim = np.dot(emb_gen, emb_ref) / (np.linalg.norm(emb_gen) * np.linalg.norm(emb_ref))

            # ROUGE-L score
            rouge_score = self.rouge.score(reference, generated)['rougeL'].fmeasure

            # BERTScore F1
            _, _, bert_f1 = bert_score([generated], [reference], lang="en", model_type="bert-base-uncased")

            # BLEU score (token-level n-gram precision)
            bleu_score = sentence_bleu([reference.split()], generated.split(), smoothing_function=self.smooth_fn)

            # METEOR score (considers synonyms and word alignment)
            meteor_score = single_meteor_score(reference.split(), generated.split())

            # Levenshtein ratio (edit similarity)
            levenshtein_ratio = SequenceMatcher(None, reference, generated).ratio()

            # Weighted aggregation of all metrics
            final_score = sum([
                METRIC_WEIGHTS["rouge_score"] * rouge_score,
                METRIC_WEIGHTS["cosine_similarity"] * cosine_sim,
                METRIC_WEIGHTS["bert_score_f1"] * bert_f1.mean().item(),
                METRIC_WEIGHTS["bleu"] * bleu_score,
                METRIC_WEIGHTS["meteor"] * meteor_score,
                METRIC_WEIGHTS["levenshtein"] * levenshtein_ratio
            ])

            return {
                "rouge_score": rouge_score,
                "cosine_similarity": float(cosine_sim),
                "bert_score_f1": bert_f1.mean().item(),
                "bleu": bleu_score,
                "meteor": meteor_score,
                "levenshtein": levenshtein_ratio,
                "final_score": final_score
            }
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            # Return a set of default metrics to allow testing to continue
            return {
                "rouge_score": 0.0,
                "cosine_similarity": 0.0,
                "bert_score_f1": 0.0,
                "bleu": 0.0, 
                "meteor": 0.0,
                "levenshtein": 0.0,
                "final_score": 0.0
            }

    def calculate_grade(self, score: float) -> str:
        """
        Convert a final numeric score to a qualitative letter grade.

        Args:
            score (float): Final weighted score (between 0 and 1).

        Returns:
            str: Qualitative grade from A to F.
        """
        if score >= 0.90: return "A (Excellent)"
        elif score >= 0.80: return "B (Good)"
        elif score >= 0.70: return "C (Average)"
        elif score >= 0.60: return "D (Below Average)"
        else: return "F (Poor)"

    def evaluate_excel(self, input_path: str, output_path: str, summary_path: str) -> None:
        """
        Process all Q&A pairs in the Excel file, compute evaluation metrics,
        write the result sheet, and save an aggregated summary as JSON.

        Args:
            input_path (str): Path to the Excel file with 'Question', 'Answer', 'predicted_answer'.
            output_path (str): Destination Excel file with metrics per row.
            summary_path (str): Path to JSON file for storing average scores and grade.
            
        Raises:
            FileNotFoundError: If input file doesn't exist
            ValueError: If required columns are missing
        """
        try:
            df = pd.read_excel(input_path)

            required_cols = ["Question", "Answer", "predicted_answer"]
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"Excel must contain columns: {required_cols}")

            # Initialize containers
            metrics_dict = {
                "rouge_score": [],
                "cosine_similarity": [],
                "bert_score_f1": [],
                "bleu": [],
                "meteor": [],
                "levenshtein": [],
                "final_score": []
            }

            # Evaluate each row
            for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
                ref = str(row["Answer"])
                pred = str(row["predicted_answer"])
                metrics = self.calculate_metrics(pred, ref)

                for key in metrics_dict:
                    metrics_dict[key].append(metrics[key])

            # Add results to dataframe
            for metric_name, values in metrics_dict.items():
                df[metric_name] = values

            # Save detailed results
            df.to_excel(output_path, index=False)
            logger.info(f"✅ Scores saved to {output_path}")

            # Save summary JSON
            summary = {k: float(np.mean(v)) for k, v in metrics_dict.items()}
            summary["grade"] = self.calculate_grade(summary["final_score"])

            self.save_summary(summary, summary_path)
            
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            raise

    def save_summary(self, summary: Dict[str, Union[float, str]], summary_path: str) -> None:
        """
        Save evaluation summary to a JSON file.
        
        Args:
            summary: Dictionary containing metric averages and letter grade
            summary_path: Path where to write the JSON file
            
        Raises:
            IOError: If file cannot be written
        """
        try:
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=4)
            logger.info(f"📊 Summary saved to {summary_path}")
            logger.info(json.dumps(summary, indent=2))
        except Exception as e:
            logger.error(f"Error saving summary to {summary_path}: {e}")
            raise


if __name__ == "__main__":
    evaluator = QAEvaluator()
    evaluator.evaluate_excel(INPUT_FILE, OUTPUT_FILE, SUMMARY_FILE)
