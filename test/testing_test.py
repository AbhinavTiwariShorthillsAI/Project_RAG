import unittest
from unittest.mock import patch, MagicMock, mock_open
import pandas as pd
import os
import sys
import json
from scripts.testing import QAEvaluator, INPUT_FILE, OUTPUT_FILE, SUMMARY_FILE

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

class TestQAEvaluator(unittest.TestCase):

    @patch("pandas.read_excel")
    def test_evaluate_excel_invalid_file_path(self, mock_read_excel):
        # Mocking the scenario when the file path doesn't exist
        mock_read_excel.side_effect = FileNotFoundError("File not found!")
        
        evaluator = QAEvaluator()

        with self.assertRaises(FileNotFoundError):
            evaluator.evaluate_excel(INPUT_FILE, OUTPUT_FILE, SUMMARY_FILE)

    @patch("pandas.read_excel")
    def test_evaluate_excel_missing_columns(self, mock_read_excel):
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
    def test_evaluate_excel_success(self, mock_metrics, mock_to_excel, mock_read_excel):
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
    def test_save_summary(self, mock_json_dump, mock_open_file):
        # Mocking the summary file saving
        evaluator = QAEvaluator()
        summary = {
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
    def test_evaluate_excel_with_invalid_input(self, mock_to_excel, mock_read_excel):
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
