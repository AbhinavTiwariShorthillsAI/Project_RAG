import unittest
from unittest.mock import patch, MagicMock, mock_open
import pandas as pd
import os
import sys
import requests
import logging
from typing import List, Dict, Any

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
logger.info("Starting question generator test suite.")

# Import the class to be tested
from scripts.question_gen import LLaMAQuestionGenerator

class TestLLaMAQuestionGenerator(unittest.TestCase):
    """
    Test suite for the LLaMAQuestionGenerator class.
    
    Tests the various methods of the LLaMAQuestionGenerator class, including text
    loading, splitting, LLaMA querying, QA pair generation, and file operations.
    """
    
    def setUp(self) -> None:
        """
        Set up test environment before each test case.
        """
        logger.info("Setting up test case.")
        # Create an instance of the generator for testing
        self.generator = LLaMAQuestionGenerator(model="test-model", url="http://test-url")
    
    def tearDown(self) -> None:
        """
        Clean up after each test case.
        """
        logger.info("Test case completed.")
    
    def test_init(self) -> None:
        """
        Test initialization of LLaMAQuestionGenerator class.
        
        Asserts:
            Generator is correctly initialized with specified parameters
        """
        logger.info("Testing generator initialization.")
        # Test with custom parameters
        generator = LLaMAQuestionGenerator(model="custom-model", url="http://custom-url")
        self.assertEqual(generator.model, "custom-model")
        self.assertEqual(generator.url, "http://custom-url")
        
        # Test with default parameters
        generator = LLaMAQuestionGenerator()
        self.assertEqual(generator.model, "llama3")
        self.assertEqual(generator.url, "http://localhost:11434/api/generate")
        logger.info("Generator initialization test passed.")
    
    @patch("builtins.open", new_callable=mock_open, read_data="Test document content")
    def test_load_text_success(self, mock_file: MagicMock) -> None:
        """
        Test successful loading of text from a file.
        
        Args:
            mock_file: Mocked open function
            
        Asserts:
            Correct text content is loaded from the file
        """
        logger.info("Testing load_text with successful file read.")
        result = self.generator.load_text("test_file.txt")
        self.assertEqual(result, "Test document content")
        mock_file.assert_called_once_with("test_file.txt", "r", encoding="utf-8")
        logger.info("load_text success test passed.")
    
    @patch("builtins.open", side_effect=FileNotFoundError)
    @patch("builtins.print")
    def test_load_text_file_not_found(self, mock_print: MagicMock, mock_file: MagicMock) -> None:
        """
        Test handling of FileNotFoundError when loading text.
        
        Args:
            mock_print: Mocked print function
            mock_file: Mocked open function that raises FileNotFoundError
            
        Asserts:
            Empty string is returned when file is not found
        """
        logger.info("Testing load_text with FileNotFoundError.")
        result = self.generator.load_text("nonexistent_file.txt")
        self.assertEqual(result, "")
        mock_file.assert_called_once_with("nonexistent_file.txt", "r", encoding="utf-8")
        mock_print.assert_called_once()
        logger.info("load_text FileNotFoundError test passed.")
    
    @patch("builtins.open", side_effect=Exception("General error"))
    @patch("builtins.print")
    def test_load_text_general_error(self, mock_print: MagicMock, mock_file: MagicMock) -> None:
        """
        Test handling of general exceptions when loading text.
        
        Args:
            mock_print: Mocked print function
            mock_file: Mocked open function that raises a general Exception
            
        Asserts:
            Empty string is returned when an error occurs
        """
        logger.info("Testing load_text with general Exception.")
        result = self.generator.load_text("problem_file.txt")
        self.assertEqual(result, "")
        mock_file.assert_called_once_with("problem_file.txt", "r", encoding="utf-8")
        mock_print.assert_called_once()
        logger.info("load_text general Exception test passed.")
    
    @patch("scripts.question_gen.RecursiveCharacterTextSplitter")
    def test_split_text_semantic_success(self, mock_splitter_class: MagicMock) -> None:
        """
        Test successful splitting of text.
        
        Args:
            mock_splitter_class: Mocked RecursiveCharacterTextSplitter class
            
        Asserts:
            Text is split correctly into chunks
        """
        logger.info("Testing split_text_semantic with successful case.")
        # Set up the mock splitter
        mock_splitter = MagicMock()
        mock_splitter_class.return_value = mock_splitter
        mock_splitter.split_text.return_value = ["Chunk 1", "Chunk 2", "Chunk 3"]
        
        # Call the method
        result = self.generator.split_text_semantic("Test document content", chunk_size=500, chunk_overlap=50)
        
        # Verify the result
        self.assertEqual(result, ["Chunk 1", "Chunk 2", "Chunk 3"])
        mock_splitter_class.assert_called_once_with(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", ".", "!", "?", "\n", " "],
            length_function=len
        )
        logger.info("split_text_semantic success test passed.")
    
    @patch("scripts.question_gen.RecursiveCharacterTextSplitter")
    @patch("builtins.print")
    def test_split_text_semantic_error(self, mock_print: MagicMock, mock_splitter_class: MagicMock) -> None:
        """
        Test handling of errors during text splitting.
        
        Args:
            mock_print: Mocked print function
            mock_splitter_class: Mocked RecursiveCharacterTextSplitter class that raises an exception
            
        Asserts:
            Empty list is returned when an error occurs
        """
        logger.info("Testing split_text_semantic with Exception.")
        # Set up the mock to raise an exception
        mock_splitter_class.side_effect = Exception("Splitting error")
        
        # Call the method
        result = self.generator.split_text_semantic("Test document content")
        
        # Verify the result
        self.assertEqual(result, [])
        mock_print.assert_called_once()
        logger.info("split_text_semantic Exception test passed.")
    
    @patch("requests.post")
    def test_query_llama_success(self, mock_post: MagicMock) -> None:
        """
        Test successful query to LLaMA model.
        
        Args:
            mock_post: Mocked requests.post function
            
        Asserts:
            Correct response is returned from the model
        """
        logger.info("Testing query_llama with successful API response.")
        # Set up the mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "Model response"}
        mock_post.return_value = mock_response
        
        # Call the method
        result = self.generator.query_llama("Test prompt")
        
        # Verify the result
        self.assertEqual(result, "Model response")
        mock_post.assert_called_once_with(
            "http://test-url",
            json={"model": "test-model", "prompt": "Test prompt", "stream": False}
        )
        logger.info("query_llama success test passed.")
    
    @patch("requests.post", side_effect=requests.exceptions.ConnectionError)
    def test_query_llama_connection_error(self, mock_post: MagicMock) -> None:
        """
        Test handling of connection errors when querying LLaMA.
        
        Args:
            mock_post: Mocked requests.post function that raises ConnectionError
            
        Asserts:
            Error message is returned when connection fails
        """
        logger.info("Testing query_llama with ConnectionError.")
        result = self.generator.query_llama("Test prompt")
        self.assertEqual(result, "Error: Unable to connect to Ollama API. Is it running?")
        logger.info("query_llama ConnectionError test passed.")
    
    @patch("requests.post", side_effect=Exception("General error"))
    def test_query_llama_general_error(self, mock_post: MagicMock) -> None:
        """
        Test handling of general errors when querying LLaMA.
        
        Args:
            mock_post: Mocked requests.post function that raises a general Exception
            
        Asserts:
            Error message with exception details is returned
        """
        logger.info("Testing query_llama with general Exception.")
        result = self.generator.query_llama("Test prompt")
        self.assertEqual(result, "Error: General error")
        logger.info("query_llama general Exception test passed.")
    
    @patch.object(LLaMAQuestionGenerator, "query_llama")
    @patch("builtins.print")
    def test_generate_qa_pairs(self, mock_print: MagicMock, mock_query: MagicMock) -> None:
        """
        Test generation of QA pairs from text chunks.
        
        Args:
            mock_print: Mocked print function
            mock_query: Mocked query_llama method
            
        Asserts:
            DataFrame with questions and answers is correctly generated
        """
        logger.info("Testing generate_qa_pairs with successful LLM response.")
        # Set up the mock to return a structured response
        mock_query.return_value = """
Question: What is the capital of France?
Answer: Paris
Question: What year did World War II end?
Answer: 1945
"""
        
        # Call the method with test chunks
        chunks = ["Chunk 1", "Chunk 2"]
        result = self.generator.generate_qa_pairs(chunks, total=2)
        
        # Verify the result
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 4)  # 2 chunks with 2 QA pairs each
        self.assertEqual(list(result.columns), ["Question", "Answer"])
        
        # Check the content
        self.assertEqual(result["Question"].iloc[0], "What is the capital of France?")
        self.assertEqual(result["Answer"].iloc[0], "Paris")
        self.assertEqual(result["Question"].iloc[1], "What year did World War II end?")
        self.assertEqual(result["Answer"].iloc[1], "1945")
        logger.info("generate_qa_pairs success test passed.")
    
    @patch.object(LLaMAQuestionGenerator, "query_llama", side_effect=Exception("Generation error"))
    @patch("builtins.print")
    def test_generate_qa_pairs_error(self, mock_print: MagicMock, mock_query: MagicMock) -> None:
        """
        Test handling of errors during QA pair generation.
        
        Args:
            mock_print: Mocked print function
            mock_query: Mocked query_llama method that raises an exception
            
        Asserts:
            Empty DataFrame is returned when an error occurs
        """
        logger.info("Testing generate_qa_pairs with Exception.")
        chunks = ["Chunk 1"]
        result = self.generator.generate_qa_pairs(chunks, total=1)
        
        # Verify the result is an empty DataFrame with the correct columns
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 0)
        self.assertEqual(list(result.columns), ["Question", "Answer"])
        logger.info("generate_qa_pairs Exception test passed.")
    
    @patch("pandas.DataFrame.to_csv")
    @patch("builtins.print")
    def test_save_to_csv_success(self, mock_print: MagicMock, mock_to_csv: MagicMock) -> None:
        """
        Test successful saving of DataFrame to CSV.
        
        Args:
            mock_print: Mocked print function
            mock_to_csv: Mocked DataFrame.to_csv method
            
        Asserts:
            to_csv method is called with correct parameters
        """
        logger.info("Testing save_to_csv with successful case.")
        # Create a test DataFrame
        df = pd.DataFrame({
            "Question": ["Q1", "Q2"],
            "Answer": ["A1", "A2"]
        })
        
        # Call the method
        self.generator.save_to_csv(df, "output.csv")
        
        # Verify the result
        mock_to_csv.assert_called_once_with("output.csv", index=False)
        mock_print.assert_called_once()
        logger.info("save_to_csv success test passed.")
    
    @patch("pandas.DataFrame.to_csv", side_effect=Exception("CSV error"))
    @patch("builtins.print")
    def test_save_to_csv_error(self, mock_print: MagicMock, mock_to_csv: MagicMock) -> None:
        """
        Test handling of errors when saving to CSV.
        
        Args:
            mock_print: Mocked print function
            mock_to_csv: Mocked DataFrame.to_csv method that raises an exception
            
        Asserts:
            Error is caught and error message is printed
        """
        logger.info("Testing save_to_csv with Exception.")
        # Create a test DataFrame
        df = pd.DataFrame({
            "Question": ["Q1", "Q2"],
            "Answer": ["A1", "A2"]
        })
        
        # Call the method
        self.generator.save_to_csv(df, "problem_output.csv")
        
        # Verify error handling
        mock_to_csv.assert_called_once_with("problem_output.csv", index=False)
        mock_print.assert_called_once()
        self.assertIn("❌ Error saving CSV", mock_print.call_args[0][0])
        logger.info("save_to_csv Exception test passed.")


if __name__ == "__main__":
    unittest.main()
    logger.info("Question generator test suite completed.") 