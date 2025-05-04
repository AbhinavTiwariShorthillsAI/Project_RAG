import unittest
from unittest.mock import patch, MagicMock, mock_open
import os
import sys
import json
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Union
import requests

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
logger.info("Starting RAG Question Answering test suite.")

# Import the class to be tested
from scripts.testing_data import RAGQuestionAnswering

class TestRAGQuestionAnswering(unittest.TestCase):
    """
    Test suite for the RAGQuestionAnswering class.
    
    Tests the various methods of the RAGQuestionAnswering class, including
    Weaviate searching, LLaMA querying, and file operations.
    """
    
    @patch("scripts.testing_data.weaviate.Client")
    @patch("scripts.testing_data.SentenceTransformer")
    def setUp(self, mock_transformer: MagicMock, mock_weaviate: MagicMock) -> None:
        """
        Set up test environment before each test case.
        
        Args:
            mock_transformer: Mocked SentenceTransformer class
            mock_weaviate: Mocked weaviate.Client class
        """
        logger.info("Setting up test case.")
        # Mock the embedding model
        self.mock_model = MagicMock()
        mock_transformer.return_value = self.mock_model
        
        # Mock Weaviate client
        self.mock_client = MagicMock()
        mock_weaviate.return_value = self.mock_client
        
        # Initialize RAGQuestionAnswering with mocked dependencies
        self.rag_qa = RAGQuestionAnswering()
    
    def tearDown(self) -> None:
        """
        Clean up after each test case.
        """
        logger.info("Test case completed.")
    
    def test_init_success(self) -> None:
        """
        Test successful initialization of RAGQuestionAnswering class.
        
        Asserts:
            Correct client and embedding model are assigned
        """
        logger.info("Testing initialization success.")
        self.assertEqual(self.rag_qa.client, self.mock_client)
        self.assertEqual(self.rag_qa.embed_model, self.mock_model)
        logger.info("Initialization success test passed.")
    
    @patch("scripts.testing_data.weaviate.Client")
    @patch("scripts.testing_data.SentenceTransformer")
    def test_init_weaviate_failure(self, mock_transformer: MagicMock, mock_weaviate: MagicMock) -> None:
        """
        Test initialization when Weaviate connection fails.
        
        Args:
            mock_transformer: Mocked SentenceTransformer class
            mock_weaviate: Mocked weaviate.Client class
            
        Asserts:
            Exception is raised when Weaviate connection fails
        """
        logger.info("Testing initialization with Weaviate failure.")
        # Make Weaviate client raise an exception
        mock_weaviate.side_effect = Exception("Connection failed")
        
        # Expect the exception to be propagated
        with self.assertRaises(Exception):
            RAGQuestionAnswering()
        logger.info("Initialization Weaviate failure test passed.")
    
    @patch("scripts.testing_data.weaviate.Client")
    @patch("scripts.testing_data.SentenceTransformer")
    def test_init_transformer_failure(self, mock_transformer: MagicMock, mock_weaviate: MagicMock) -> None:
        """
        Test initialization when SentenceTransformer model loading fails.
        
        Args:
            mock_transformer: Mocked SentenceTransformer class
            mock_weaviate: Mocked weaviate.Client class
            
        Asserts:
            Exception is raised when model loading fails
        """
        logger.info("Testing initialization with transformer failure.")
        # Weaviate client works fine
        mock_weaviate.return_value = MagicMock()
        
        # But SentenceTransformer raises an exception
        mock_transformer.side_effect = Exception("Model loading failed")
        
        # Expect the exception to be propagated
        with self.assertRaises(Exception):
            RAGQuestionAnswering()
        logger.info("Initialization transformer failure test passed.")
    
    def test_search_weaviate_success(self) -> None:
        """
        Test successful retrieval of chunks from Weaviate.
        
        Asserts:
            Correct text chunks are returned from the search
        """
        logger.info("Testing search_weaviate with successful retrieval.")
        # Mock embedding generation
        self.mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        
        # Setup the mock query chain
        mock_get = MagicMock()
        mock_near_vector = MagicMock()
        mock_with_limit = MagicMock()
        mock_do = MagicMock()
        
        self.mock_client.query.get.return_value = mock_get
        mock_get.with_near_vector.return_value = mock_near_vector
        mock_near_vector.with_limit.return_value = mock_with_limit
        mock_with_limit.do.return_value = {
            "data": {
                "Get": {
                    "WorldWarChunk": [
                        {"text": "Chunk 1"},
                        {"text": "Chunk 2"},
                        {"text": "Chunk 3"}
                    ]
                }
            }
        }
        
        # Call the method
        result = self.rag_qa.search_weaviate("test query")
        
        # Verify the result
        self.assertEqual(result, ["Chunk 1", "Chunk 2", "Chunk 3"])
        self.mock_model.encode.assert_called_once_with(["test query"])
        logger.info("search_weaviate successful retrieval test passed.")
    
    def test_search_weaviate_failure(self) -> None:
        """
        Test Weaviate search when query fails.
        
        Asserts:
            Empty list is returned when search fails
        """
        logger.info("Testing search_weaviate with query failure.")
        # Mock embedding generation
        self.mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        
        # Make the query chain raise an exception
        self.mock_client.query.get.side_effect = Exception("Query failed")
        
        # Call the method
        result = self.rag_qa.search_weaviate("test query")
        
        # Verify empty list is returned on failure
        self.assertEqual(result, [])
        logger.info("search_weaviate query failure test passed.")
    
    @patch("scripts.testing_data.requests.post")
    def test_query_llama_success(self, mock_post: MagicMock) -> None:
        """
        Test successful query to LLaMA model via Ollama API.
        
        Args:
            mock_post: Mocked requests.post function
            
        Asserts:
            Correct response is extracted from the API response
        """
        logger.info("Testing query_llama with successful API response.")
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "Model's answer"}
        mock_post.return_value = mock_response
        
        # Call the method
        result = self.rag_qa.query_llama("Test prompt")
        
        # Verify the result
        self.assertEqual(result, "Model's answer")
        mock_post.assert_called_once()
        logger.info("query_llama successful API response test passed.")
    
    @patch("scripts.testing_data.requests.post")
    def test_query_llama_failure(self, mock_post: MagicMock) -> None:
        """
        Test LLaMA query when API request fails.
        
        Args:
            mock_post: Mocked requests.post function
            
        Asserts:
            Error message is returned when API request fails
        """
        logger.info("Testing query_llama with API request failure.")
        # Make the API request raise an exception
        mock_post.side_effect = requests.exceptions.RequestException("API request failed")
        
        # Call the method
        result = self.rag_qa.query_llama("Test prompt")
        
        # Verify error message is returned
        self.assertTrue("Error querying" in result)
        logger.info("query_llama API request failure test passed.")
    
    @patch("builtins.open", new_callable=mock_open, read_data="question,answer\nq1,a1\nq2,a2\n")
    def test_question_fetch(self, mock_file: MagicMock) -> None:
        """
        Test fetching question-answer pairs from CSV file.
        
        Args:
            mock_file: Mocked open function with CSV content
            
        Asserts:
            Correct question-answer pairs are yielded
        """
        logger.info("Testing question_fetch with CSV file.")
        # Call the method and collect results
        results = list(self.rag_qa.question_fetch("dummy.csv"))
        
        # Verify the results
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0], {"question": "q1", "answer": "a1"})
        self.assertEqual(results[1], {"question": "q2", "answer": "a2"})
        logger.info("question_fetch CSV file test passed.")
    
    @patch("scripts.testing_data.os.path.exists")
    @patch("scripts.testing_data.load_workbook")
    @patch("scripts.testing_data.Workbook")
    def test_run_existing_file(self, 
                              mock_workbook: MagicMock, 
                              mock_load_workbook: MagicMock, 
                              mock_exists: MagicMock) -> None:
        """
        Test run method when output Excel file already exists.
        
        Args:
            mock_workbook: Mocked Workbook class
            mock_load_workbook: Mocked load_workbook function
            mock_exists: Mocked os.path.exists function
            
        Asserts:
            Existing workbook is loaded instead of creating new one
        """
        logger.info("Testing run method with existing Excel file.")
        # Mock that file exists
        mock_exists.return_value = True
        
        # Mock workbook loading
        mock_wb = MagicMock()
        mock_ws = MagicMock()
        mock_wb.active = mock_ws
        mock_load_workbook.return_value = mock_wb
        
        # Mock other methods to prevent full execution
        self.rag_qa.question_fetch = MagicMock(return_value=[])
        
        # Call the method
        self.rag_qa.run()
        
        # Verify existing workbook was loaded
        mock_load_workbook.assert_called_once()
        mock_workbook.assert_not_called()
        logger.info("run method with existing Excel file test passed.")


if __name__ == "__main__":
    unittest.main()
    logger.info("RAG Question Answering test suite completed.") 