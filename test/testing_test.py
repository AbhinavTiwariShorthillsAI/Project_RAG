import unittest
from unittest.mock import patch, MagicMock, mock_open
import pandas as pd
import os
import sys
import json
import logging
from typing import Dict, Any, Union, Optional
import numpy as np

# Add the project root to the Python path to properly resolve imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Set up logging
log_folder = os.path.join(os.path.dirname(__file__), 'logs')
if not os.path.exists(log_folder):
    os.makedirs(log_folder)

log_file_name = os.path.splitext(os.path.basename(__file__))[0] + "_logs.log"
log_file_path = os.path.join(log_folder, log_file_name)

logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)

logger = logging.getLogger(__name__)
logger.info("Starting QA Evaluator test suite.")

# Now import from scripts module
from scripts.testing import QAEvaluator, INPUT_FILE, OUTPUT_FILE, SUMMARY_FILE

class TestQAEvaluator(unittest.TestCase):
    """
    Test suite for the QAEvaluator class.
    
    Tests the various functions of the QAEvaluator class, including Excel file
    processing, metric calculations, and result saving.
    """

    def setUp(self) -> None:
        """
        Set up the test environment.
        """
        logger.info("Setting up test case.")
    
    def tearDown(self) -> None:
        """
        Clean up after each test case.
        """
        logger.info("Test case completed.")

    @patch("scripts.testing.SentenceTransformer")
    @patch("pandas.read_excel")
    def test_evaluate_excel_invalid_file_path(self, mock_read_excel: MagicMock, mock_transformer: MagicMock) -> None:
        """
        Test behavior when evaluating an Excel file that does not exist.
        
        Args:
            mock_read_excel: Mocked pandas.read_excel function
            mock_transformer: Mocked SentenceTransformer class
            
        Asserts:
            FileNotFoundError is raised when the input file doesn't exist
        """
        logger.info("Testing evaluate_excel with invalid file path.")
        # Set up the SentenceTransformer mock
        mock_model = MagicMock()
        mock_transformer.return_value = mock_model
        
        # Mocking the scenario when the file path doesn't exist
        mock_read_excel.side_effect = FileNotFoundError("File not found!")
        
        evaluator = QAEvaluator()

        with self.assertRaises(FileNotFoundError):
            evaluator.evaluate_excel(INPUT_FILE, OUTPUT_FILE, SUMMARY_FILE)
        logger.info("evaluate_excel invalid file path test passed.")

    @patch("scripts.testing.SentenceTransformer")
    @patch("pandas.read_excel")
    def test_evaluate_excel_missing_columns(self, mock_read_excel: MagicMock, mock_transformer: MagicMock) -> None:
        """
        Test behavior when the Excel file is missing required columns.
        
        Args:
            mock_read_excel: Mocked pandas.read_excel function
            mock_transformer: Mocked SentenceTransformer class
            
        Asserts:
            ValueError is raised when required columns are missing
        """
        logger.info("Testing evaluate_excel with missing columns.")
        # Set up the SentenceTransformer mock
        mock_model = MagicMock()
        mock_transformer.return_value = mock_model
        
        # Mocking invalid Excel file structure (missing columns)
        mock_df = MagicMock()
        mock_df.columns = ["Question", "predicted_answer"]  # Missing 'Answer' column
        mock_read_excel.return_value = mock_df

        evaluator = QAEvaluator()

        with self.assertRaises(ValueError):
            evaluator.evaluate_excel(INPUT_FILE, OUTPUT_FILE, SUMMARY_FILE)
        logger.info("evaluate_excel missing columns test passed.")

    @patch("scripts.testing.SentenceTransformer")
    @patch("scripts.testing.QAEvaluator.calculate_metrics")
    @patch("pandas.DataFrame.to_excel")
    @patch("pandas.read_excel")
    def test_evaluate_excel_success(self, 
                                   mock_read_excel: MagicMock, 
                                   mock_to_excel: MagicMock,
                                   mock_metrics: MagicMock,
                                   mock_transformer: MagicMock) -> None:
        """
        Test successful evaluation of an Excel file with valid data.
        
        Args:
            mock_read_excel: Mocked pandas.read_excel function
            mock_to_excel: Mocked DataFrame.to_excel method
            mock_metrics: Mocked calculate_metrics method
            mock_transformer: Mocked SentenceTransformer class
            
        Asserts:
            to_excel method is called with correct parameters
        """
        logger.info("Testing evaluate_excel with successful case.")
        # Set up the SentenceTransformer mock
        mock_model = MagicMock()
        mock_transformer.return_value = mock_model
        
        # Mocking a valid DataFrame and metrics calculation
        mock_df = pd.DataFrame({
            "Question": ["What is war?", "What is peace?"],
            "Answer": ["Conflict", "Tranquility"],
            "predicted_answer": ["War", "Peace"]
        })
        mock_read_excel.return_value = mock_df
        
        # Include all the metrics keys that are expected
        mock_metrics.return_value = {
            "rouge_score": 0.8,
            "cosine_similarity": 0.9,
            "bert_score_f1": 0.85,
            "bleu": 0.7,
            "meteor": 0.75,
            "levenshtein": 0.65,
            "final_score": 0.85
        }

        evaluator = QAEvaluator()
        evaluator.evaluate_excel(INPUT_FILE, OUTPUT_FILE, SUMMARY_FILE)

        # Check if DataFrame's `to_excel` was called
        mock_to_excel.assert_called_once_with(OUTPUT_FILE, index=False)
        logger.info("evaluate_excel successful case test passed.")

    @patch("scripts.testing.SentenceTransformer")
    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump")
    def test_save_summary(self, 
                         mock_json_dump: MagicMock, 
                         mock_open_file: MagicMock,
                         mock_transformer: MagicMock) -> None:
        """
        Test saving evaluation summary to a JSON file.
        
        Args:
            mock_json_dump: Mocked json.dump function
            mock_open_file: Mocked open function
            mock_transformer: Mocked SentenceTransformer class
            
        Asserts:
            json.dump is called with correct parameters
        """
        logger.info("Testing save_summary function.")
        # Set up the SentenceTransformer mock
        mock_model = MagicMock()
        mock_transformer.return_value = mock_model
        
        # Mocking the summary file saving
        evaluator = QAEvaluator()
        summary: Dict[str, Union[float, str]] = {
            "rouge_score": 0.8,
            "cosine_similarity": 0.9,
            "bert_score_f1": 0.85,
            "bleu": 0.7,
            "meteor": 0.75,
            "levenshtein": 0.65,
            "final_score": 0.85,
            "grade": "A (Excellent)"
        }

        evaluator.save_summary(summary, SUMMARY_FILE)
        # Ensure that summary was written correctly
        mock_json_dump.assert_called_once_with(summary, mock_open_file.return_value, indent=4)
        logger.info("save_summary function test passed.")

    @patch("scripts.testing.SentenceTransformer")
    @patch("pandas.read_excel")
    @patch("pandas.DataFrame.to_excel")
    def test_evaluate_excel_with_invalid_input(self, 
                                             mock_to_excel: MagicMock, 
                                             mock_read_excel: MagicMock,
                                             mock_transformer: MagicMock) -> None:
        """
        Test behavior when the input Excel file is malformed.
        
        Args:
            mock_to_excel: Mocked DataFrame.to_excel method
            mock_read_excel: Mocked pandas.read_excel function
            mock_transformer: Mocked SentenceTransformer class
            
        Asserts:
            ValueError is raised for malformed input file
        """
        logger.info("Testing evaluate_excel with malformed input.")
        # Set up the SentenceTransformer mock
        mock_model = MagicMock()
        mock_transformer.return_value = mock_model
        
        # Mocking a scenario where input Excel file is malformed
        mock_df = MagicMock()
        mock_df.columns = []  # No columns
        mock_read_excel.return_value = mock_df

        evaluator = QAEvaluator()
        
        # Expecting ValueError for malformed input file
        with self.assertRaises(ValueError):
            evaluator.evaluate_excel(INPUT_FILE, OUTPUT_FILE, SUMMARY_FILE)
        logger.info("evaluate_excel malformed input test passed.")

    @patch("scripts.testing.SentenceTransformer")
    @patch("scripts.testing.rouge_scorer.RougeScorer")
    @patch("scripts.testing.bert_score")
    @patch("scripts.testing.sentence_bleu")
    @patch("scripts.testing.single_meteor_score")
    @patch("scripts.testing.SequenceMatcher")
    def test_calculate_metrics_success(self, mock_seq_matcher: MagicMock, 
                                   mock_meteor: MagicMock, mock_bleu: MagicMock, 
                                   mock_bert_score: MagicMock, mock_rouge_scorer: MagicMock, mock_st: MagicMock):
        """
        Test the calculate_metrics function under successful conditions.
        
        Args:
            mock_seq_matcher: Mocked SequenceMatcher class
            mock_meteor: Mocked single_meteor_score function
            mock_bleu: Mocked sentence_bleu function
            mock_bert_score: Mocked bert_score function
            mock_rouge_scorer: Mocked RougeScorer class
            mock_st: Mocked SentenceTransformer class
            
        Asserts:
            All metrics are calculated correctly
            The final score is the weighted sum of all metrics
        """
        logger.info("Testing calculate_metrics with success case.")
        
        # Initialize the evaluator
        with patch("scripts.testing.SentenceTransformer") as mock_st:
            with patch("scripts.testing.rouge_scorer.RougeScorer") as mock_rouge_scorer:
                # Set up mocks
                mock_model = MagicMock()
                mock_st.return_value = mock_model
                mock_rouge = MagicMock()
                mock_rouge_scorer.return_value = mock_rouge
                
                # Create evaluator
                evaluator = QAEvaluator()
                
                # Configure mocks
                mock_model.encode.side_effect = [
                    np.array([0.1, 0.2, 0.3]),  # Generated text embedding
                    np.array([0.2, 0.3, 0.4])   # Reference text embedding
                ]
                
                # Rouge score mock
                mock_rouge.score.return_value = {'rougeL': MagicMock(fmeasure=0.75)}
                
                # BERT score mock
                bert_f1_mock = MagicMock()
                bert_f1_mock.mean.return_value.item.return_value = 0.85
                mock_bert_score.return_value = (None, None, bert_f1_mock)
                
                # BLEU score mock
                mock_bleu.return_value = 0.65
                
                # METEOR score mock
                mock_meteor.return_value = 0.70
                
                # Levenshtein ratio mock
                seq_matcher_instance = MagicMock()
                seq_matcher_instance.ratio.return_value = 0.80
                mock_seq_matcher.return_value = seq_matcher_instance
                
                # Call the function
                generated_text = "World War II ended in 1945 with victory for the Allies."
                reference_text = "The Second World War concluded in 1945 with the Allied powers victorious."
                metrics = evaluator.calculate_metrics(generated_text, reference_text)
                
                # Verify the metrics
                self.assertIn("rouge_score", metrics)
                self.assertIn("cosine_similarity", metrics)
                self.assertIn("bert_score_f1", metrics)
                self.assertIn("bleu", metrics)
                self.assertIn("meteor", metrics)
                self.assertIn("levenshtein", metrics)
                self.assertIn("final_score", metrics)
                
                # Check all metrics are between 0 and 1
                for metric_name, value in metrics.items():
                    self.assertGreaterEqual(value, 0.0)
                    self.assertLessEqual(value, 1.0)
                
                # Verify the individual metrics
                self.assertEqual(metrics["rouge_score"], 0.75)
                self.assertEqual(metrics["bert_score_f1"], 0.85)
                self.assertEqual(metrics["bleu"], 0.65)
                self.assertEqual(metrics["meteor"], 0.70)
                self.assertEqual(metrics["levenshtein"], 0.80)
                
                # Calculate the expected final score directly from the weights and metrics
                expected_final_score = (
                    0.15 * 0.75 +   # rouge_score
                    0.35 * metrics["cosine_similarity"] +  # Use the actual value from metrics
                    0.30 * 0.85 +   # bert_score_f1
                    0.05 * 0.65 +   # bleu
                    0.10 * 0.70 +   # meteor
                    0.05 * 0.80     # levenshtein
                )
                
                # Verify the final score
                self.assertAlmostEqual(metrics["final_score"], expected_final_score, places=5)
                
                logger.info("calculate_metrics success test passed.")

    def test_calculate_metrics_error(self):
        """
        Test the calculate_metrics function with empty inputs.
        
        Asserts:
            ValueError is raised when inputs are empty
        """
        logger.info("Testing calculate_metrics with error case.")
        
        # Initialize the evaluator
        with patch("scripts.testing.SentenceTransformer") as mock_st:
            with patch("scripts.testing.rouge_scorer.RougeScorer") as mock_rouge_scorer:
                # Create evaluator
                evaluator = QAEvaluator()
                
                # Test with empty inputs
                with self.assertRaises(ValueError):
                    evaluator.calculate_metrics("", "reference text")
                    
                with self.assertRaises(ValueError):
                    evaluator.calculate_metrics("generated text", "")
                    
                with self.assertRaises(ValueError):
                    evaluator.calculate_metrics("", "")
                    
                logger.info("calculate_metrics error test passed.")
    
    def test_calculate_grade(self):
        """
        Test the calculate_grade function.
        
        Asserts:
            Correct grade letter is returned for different score ranges
        """
        logger.info("Testing calculate_grade function.")
        
        # Initialize the evaluator
        with patch("scripts.testing.SentenceTransformer") as mock_st:
            with patch("scripts.testing.rouge_scorer.RougeScorer") as mock_rouge_scorer:
                # Create evaluator
                evaluator = QAEvaluator()
                
                # Test various score ranges
                self.assertEqual(evaluator.calculate_grade(0.95), "A (Excellent)")
                self.assertEqual(evaluator.calculate_grade(0.90), "A (Excellent)")
                self.assertEqual(evaluator.calculate_grade(0.85), "B (Good)")
                self.assertEqual(evaluator.calculate_grade(0.80), "B (Good)")
                self.assertEqual(evaluator.calculate_grade(0.75), "C (Average)")
                self.assertEqual(evaluator.calculate_grade(0.70), "C (Average)")
                self.assertEqual(evaluator.calculate_grade(0.65), "D (Below Average)")
                self.assertEqual(evaluator.calculate_grade(0.60), "D (Below Average)")
                self.assertEqual(evaluator.calculate_grade(0.55), "F (Poor)")
                self.assertEqual(evaluator.calculate_grade(0.0), "F (Poor)")
                
                logger.info("calculate_grade test passed.")


if __name__ == '__main__':
    unittest.main()
    logger.info("QA Evaluator test suite completed.")
