import unittest
from unittest.mock import patch, MagicMock, mock_open
import pandas as pd
import os
import sys
import json
from typing import Dict, Any, Union, Optional
from scripts.testing import QAEvaluator, INPUT_FILE, OUTPUT_FILE, SUMMARY_FILE

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

class TestQAEvaluator(unittest.TestCase):
    """
    Test suite for the QAEvaluator class.
    
    Tests the various functions of the QAEvaluator class, including Excel file
    processing, metric calculations, and result saving.
    """

    @patch("pandas.read_excel")
    def test_evaluate_excel_invalid_file_path(self, mock_read_excel: MagicMock) -> None:
        """
        Test behavior when evaluating an Excel file that does not exist.
        
        Args:
            mock_read_excel: Mocked pandas.read_excel function
            
        Asserts:
            FileNotFoundError is raised when the input file doesn't exist
        """
        # Mocking the scenario when the file path doesn't exist
        mock_read_excel.side_effect = FileNotFoundError("File not found!")
        
        evaluator = QAEvaluator()

        with self.assertRaises(FileNotFoundError):
            evaluator.evaluate_excel(INPUT_FILE, OUTPUT_FILE, SUMMARY_FILE)

    @patch("pandas.read_excel")
    def test_evaluate_excel_missing_columns(self, mock_read_excel: MagicMock) -> None:
        """
        Test behavior when the Excel file is missing required columns.
        
        Args:
            mock_read_excel: Mocked pandas.read_excel function
            
        Asserts:
            ValueError is raised when required columns are missing
        """
        # Mocking invalid Excel file structure (missing columns)
        mock_df = MagicMock()
        mock_df.columns = ["Question", "predicted_answer"]  # Missing 'Answer' column
        mock_read_excel.return_value = mock_df

        evaluator = QAEvaluator()

        with self.assertRaises(ValueError):
            evaluator.evaluate_excel(INPUT_FILE, OUTPUT_FILE, SUMMARY_FILE)

    @patch("pandas.read_excel")
    @patch("pandas.DataFrame.to_excel")
    @patch("scripts.qa_evaluator.QAEvaluator.calculate_metrics")
    def test_evaluate_excel_success(self, 
                                   mock_metrics: MagicMock, 
                                   mock_to_excel: MagicMock, 
                                   mock_read_excel: MagicMock) -> None:
        """
        Test successful evaluation of an Excel file with valid data.
        
        Args:
            mock_metrics: Mocked calculate_metrics method
            mock_to_excel: Mocked DataFrame.to_excel method
            mock_read_excel: Mocked pandas.read_excel function
            
        Asserts:
            to_excel method is called with correct parameters
        """
        # Mocking a valid DataFrame and metrics calculation
        mock_df = pd.DataFrame({
            "Question": ["What is war?", "What is peace?"],
            "Answer": ["Conflict", "Tranquility"],
            "predicted_answer": ["War", "Peace"]
        })
        mock_read_excel.return_value = mock_df
        mock_metrics.return_value = {
            "rouge_score": 0.8,
            "cosine_similarity": 0.9,
            "bert_score_f1": 0.85,
            "final_score": 0.85
        }

        evaluator = QAEvaluator()
        evaluator.evaluate_excel(INPUT_FILE, OUTPUT_FILE, SUMMARY_FILE)

        # Check if DataFrame's `to_excel` was called
        mock_to_excel.assert_called_once_with(OUTPUT_FILE, index=False)

    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump")
    def test_save_summary(self, 
                         mock_json_dump: MagicMock, 
                         mock_open_file: MagicMock) -> None:
        """
        Test saving evaluation summary to a JSON file.
        
        Args:
            mock_json_dump: Mocked json.dump function
            mock_open_file: Mocked open function
            
        Asserts:
            json.dump is called with correct parameters
        """
        # Mocking the summary file saving
        evaluator = QAEvaluator()
        summary: Dict[str, Union[float, str]] = {
            "rouge_score": 0.8,
            "cosine_similarity": 0.9,
            "bert_score_f1": 0.85,
            "final_score": 0.85,
            "grade": "A (Excellent)"
        }

        evaluator.save_summary(summary, SUMMARY_FILE)
        # Ensure that summary was written correctly
        mock_json_dump.assert_called_once_with(summary, mock_open_file.return_value, indent=4)

    @patch("pandas.read_excel")
    @patch("pandas.DataFrame.to_excel")
    def test_evaluate_excel_with_invalid_input(self, 
                                             mock_to_excel: MagicMock, 
                                             mock_read_excel: MagicMock) -> None:
        """
        Test behavior when the input Excel file is malformed.
        
        Args:
            mock_to_excel: Mocked DataFrame.to_excel method
            mock_read_excel: Mocked pandas.read_excel function
            
        Asserts:
            ValueError is raised for malformed input file
        """
        # Mocking a scenario where input Excel file is malformed
        mock_df = MagicMock()
        mock_df.columns = []  # No columns
        mock_read_excel.return_value = mock_df

        evaluator = QAEvaluator()
        
        # Expecting ValueError for malformed input file
        with self.assertRaises(ValueError):
            evaluator.evaluate_excel(INPUT_FILE, OUTPUT_FILE, SUMMARY_FILE)


if __name__ == '__main__':
    unittest.main()
