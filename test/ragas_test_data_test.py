import unittest
from unittest.mock import patch, mock_open, MagicMock
import json
import os
import sys
import csv
import logging
from io import StringIO
from typing import Dict, List, Any
import re

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
logger.info("Starting RAGAS test data processing test suite.")

# Import the functions to be tested
from scripts.ragas_test_data import read_scores_from_csv, compute_averages, save_summary_to_json, main
from scripts.raga_test import build_prompt, extract_scores

class TestRagasTestData(unittest.TestCase):
    """
    Test suite for the RAGAS test data processing functions.
    
    Tests the various functions for reading, computing, and saving
    evaluation metrics for RAG systems.
    """
    
    def setUp(self) -> None:
        """
        Set up test environment before each test case.
        """
        logger.info("Setting up test case.")
        # Sample CSV data for testing
        self.csv_data = """faithfulness,answer_relevancy,context_precision,context_recall
0.8,0.9,0.7,0.6
0.9,0.7,0.8,0.5
0.7,0.8,0.6,0.7
"""
        # Expected scores after parsing the CSV
        self.expected_scores = {
            "faithfulness": [0.8, 0.9, 0.7],
            "answer_relevancy": [0.9, 0.7, 0.8],
            "context_precision": [0.7, 0.8, 0.6],
            "context_recall": [0.6, 0.5, 0.7]
        }
        
        # Expected averages
        self.expected_averages = {
            "faithfulness": 0.8,
            "answer_relevancy": 0.8,
            "context_precision": 0.7,
            "context_recall": 0.6
        }
    
    def tearDown(self) -> None:
        """
        Clean up after each test case.
        """
        logger.info("Test case completed.")
    
    @patch("builtins.open", new_callable=mock_open)
    def test_read_scores_from_csv(self, mock_file: MagicMock) -> None:
        """
        Test reading scores from CSV file.
        
        Args:
            mock_file: Mocked open function
            
        Asserts:
            Scores are correctly parsed from CSV data
        """
        logger.info("Testing read_scores_from_csv with valid CSV data.")
        # Set up the mock to return our sample CSV data
        mock_file.return_value = StringIO(self.csv_data)
        
        # Call the function
        scores = read_scores_from_csv("dummy.csv")
        
        # Verify the scores match our expected values
        for metric, values in self.expected_scores.items():
            self.assertEqual(len(scores[metric]), len(values))
            for i, value in enumerate(values):
                self.assertEqual(scores[metric][i], value)
        
        # Check that the file was opened correctly
        mock_file.assert_called_once_with("dummy.csv", mode="r")
        logger.info("read_scores_from_csv valid data test passed.")
    
    @patch("builtins.open", new_callable=mock_open)
    @patch("builtins.print")
    def test_read_scores_with_invalid_data(self, mock_print: MagicMock, mock_file: MagicMock) -> None:
        """
        Test reading CSV with invalid (non-numeric) data.
        
        Args:
            mock_print: Mocked print function
            mock_file: Mocked open function
            
        Asserts:
            Invalid rows are skipped and warning is printed
        """
        logger.info("Testing read_scores_from_csv with invalid CSV data.")
        # CSV with invalid data in the second row
        invalid_csv = """faithfulness,answer_relevancy,context_precision,context_recall
0.8,0.9,0.7,0.6
N/A,invalid,error,bad
0.7,0.8,0.6,0.7
"""
        mock_file.return_value = StringIO(invalid_csv)
        
        # Call the function
        scores = read_scores_from_csv("invalid.csv")
        
        # Check that only valid rows were parsed
        self.assertEqual(len(scores["faithfulness"]), 2)
        self.assertEqual(scores["faithfulness"], [0.8, 0.7])
        
        # Verify that print was called (for the warning)
        mock_print.assert_called_once()
        logger.info("read_scores_from_csv invalid data test passed.")
    
    def test_compute_averages(self) -> None:
        """
        Test computation of average scores.
        
        Asserts:
            Averages are correctly calculated for each metric
        """
        logger.info("Testing compute_averages with valid scores.")
        averages = compute_averages(self.expected_scores)
        
        # Check that each average matches the expected value
        for metric, expected in self.expected_averages.items():
            self.assertEqual(averages[metric], expected)
        logger.info("compute_averages valid scores test passed.")
    
    def test_compute_averages_empty_list(self) -> None:
        """
        Test computation of averages with empty data.
        
        Asserts:
            ZeroDivisionError is raised when a metric has no scores
        """
        logger.info("Testing compute_averages with empty list.")
        empty_scores = {
            "faithfulness": [],
            "answer_relevancy": [0.5],
            "context_precision": [0.7],
            "context_recall": [0.6]
        }
        
        with self.assertRaises(ZeroDivisionError):
            compute_averages(empty_scores)
        logger.info("compute_averages empty list test passed.")
    
    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump")
    @patch("builtins.print")
    def test_save_summary_to_json(self, mock_print: MagicMock, mock_json_dump: MagicMock, mock_file: MagicMock) -> None:
        """
        Test saving summary to JSON file.
        
        Args:
            mock_print: Mocked print function
            mock_json_dump: Mocked json.dump function
            mock_file: Mocked open function
            
        Asserts:
            Summary is correctly saved to JSON
        """
        logger.info("Testing save_summary_to_json.")
        # Call the function
        save_summary_to_json(self.expected_averages, "output.json")
        
        # Check that the file was opened correctly
        mock_file.assert_called_once_with("output.json", "w")
        
        # Check that json.dump was called with the right arguments
        mock_json_dump.assert_called_once_with(self.expected_averages, mock_file(), indent=2)
        
        # Check that the success message was printed
        self.assertEqual(mock_print.call_count, 2)  # Two print calls
        logger.info("save_summary_to_json test passed.")
    
    @patch("scripts.ragas_test_data.read_scores_from_csv")
    @patch("scripts.ragas_test_data.compute_averages")
    @patch("scripts.ragas_test_data.save_summary_to_json")
    def test_main(self, mock_save: MagicMock, mock_compute: MagicMock, mock_read: MagicMock) -> None:
        """
        Test the main function that orchestrates the whole process.
        
        Args:
            mock_save: Mocked save_summary_to_json function
            mock_compute: Mocked compute_averages function
            mock_read: Mocked read_scores_from_csv function
            
        Asserts:
            All component functions are called with correct arguments
        """
        logger.info("Testing main function orchestration.")
        # Set up the mocks
        mock_read.return_value = self.expected_scores
        mock_compute.return_value = self.expected_averages
        
        # Call the main function
        main()
        
        # Check that each function was called with the right arguments
        mock_read.assert_called_once_with("data/evaluation_scores_split_mistral.csv")
        mock_compute.assert_called_once_with(self.expected_scores)
        mock_save.assert_called_once_with(self.expected_averages, "data/evaluation_summary_ragas_mistral.json")
        logger.info("main function test passed.")

class TestRagasTestFunctions(unittest.TestCase):
    """
    Test suite for RAGAS test script functions.
    
    Tests the functionality of `build_prompt` and `extract_scores` functions.
    """
    
    def test_build_prompt(self):
        """
        Test the build_prompt function to ensure it correctly formats the evaluation prompt.
        
        Asserts:
            The prompt contains all required components in the expected format
        """
        logger.info("Testing build_prompt function.")
        
        # Test data
        question = "When did World War II end?"
        answer = "World War II ended in 1945."
        predicted_answer = "The Second World War ended in 1945 with victory for the Allies."
        context = "World War II was a global conflict that lasted from 1939 to 1945. The war ended with the victory of the Allied Powers."
        
        # Get the prompt
        prompt = build_prompt(question, answer, predicted_answer, context)
        
        # Check that the prompt contains the key elements
        self.assertIn(question, prompt)
        self.assertIn(answer, prompt)
        self.assertIn(predicted_answer, prompt)
        self.assertIn(context, prompt)
        
        # Check that it contains the evaluation criteria
        self.assertIn("Faithfulness", prompt)
        self.assertIn("Answer Relevancy", prompt)
        self.assertIn("Context Precision", prompt)
        self.assertIn("Context Recall", prompt)
        
        # Check that it specifies the format for response
        self.assertIn("Return ONLY a Python-style list of 4 floats", prompt)
        
        logger.info("build_prompt test passed.")
        
    def test_extract_scores_valid_format(self):
        """
        Test the extract_scores function with properly formatted response text.
        
        Asserts:
            The function correctly extracts the 4 float scores from the response
        """
        logger.info("Testing extract_scores with valid format.")
        
        # Test cases with various valid formats
        test_cases = [
            # Standard format
            "[0.95521, 0.85312, 0.91230, 0.85428]",
            # Extra spaces
            "[  0.95521  ,  0.85312  ,  0.91230  ,  0.85428  ]",
            # Different values
            "[0.12345, 0.67890, 0.11111, 0.99999]",
            # With text before and after (should still work)
            "Some text before [0.95521, 0.85312, 0.91230, 0.85428] and after",
        ]
        
        expected_results = [
            [0.95521, 0.85312, 0.91230, 0.85428],
            [0.95521, 0.85312, 0.91230, 0.85428],
            [0.12345, 0.67890, 0.11111, 0.99999],
            [0.95521, 0.85312, 0.91230, 0.85428],
        ]
        
        for i, test_case in enumerate(test_cases):
            scores = extract_scores(test_case)
            self.assertEqual(len(scores), 4)
            for j in range(4):
                self.assertAlmostEqual(scores[j], expected_results[i][j])
        
        logger.info("extract_scores valid format test passed.")
        
    def test_extract_scores_invalid_format(self):
        """
        Test the extract_scores function with improperly formatted response text.
        
        Asserts:
            The function returns a list of None values when format is invalid
        """
        logger.info("Testing extract_scores with invalid format.")
        
        # Test cases with invalid formats
        test_cases = [
            # Empty string
            "",
            # No brackets
            "0.95521, 0.85312, 0.91230, 0.85428",
            # Not enough values
            "[0.95521, 0.85312, 0.91230]",
            # Wrong type of values
            "[abc, def, ghi, jkl]",
            # Not even close
            "This is not a valid response",
        ]
        
        for test_case in test_cases:
            scores = extract_scores(test_case)
            self.assertEqual(scores, [None, None, None, None])
        
        logger.info("extract_scores invalid format test passed.")

if __name__ == "__main__":
    unittest.main()
    logger.info("RAGAS test data processing test suite completed.") 