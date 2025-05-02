import unittest
from unittest.mock import patch, MagicMock
import os
import logging
import sys

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from app.embedding import ChunkUploader

# Set up logging
logging.basicConfig(
    filename='test/logs/test_chunk_uploader.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger()
logger.info("Starting the test suite.")

class TestChunkUploader(unittest.TestCase):

    @patch("app.embedding.SentenceTransformer")
    @patch("app.embedding.weaviate.Client")
    def setUp(self, mock_weaviate_client, mock_sentence_transformer):
        """
        Test setup for ChunkUploader.
        Mocks external dependencies (Weaviate client and SentenceTransformer).
        """
        # Mocking the Weaviate client
        self.mock_client = MagicMock()
        mock_weaviate_client.return_value = self.mock_client

        # Mocking the SentenceTransformer
        self.mock_embed_model = MagicMock()
        mock_sentence_transformer.return_value = self.mock_embed_model

        # Setting up the test file and the uploader
        self.test_file = "test_data.txt"
        self.uploader = ChunkUploader(
            text_file=self.test_file,
            embedding_model_name="test-model",
            weaviate_url="http://mock-url:8080",
            class_name="TestClass"
        )

        # Create a temporary text file for testing
        with open(self.test_file, "w") as f:
            f.write("This is a test document for chunk uploader testing.")
        logger.info("Test file created successfully.")

    def tearDown(self):
        """Cleanup after each test."""
        os.remove(self.test_file)
        logger.info(f"Test file {self.test_file} removed.")

    def test_load_text(self):
        """Test the load_text function."""
        logger.info("Testing load_text function...")
        text = self.uploader.load_text()
        self.assertEqual(text, "This is a test document for chunk uploader testing.")
        logger.info("load_text test passed.")

def test_split_text_semantic(self):
    """Test the split_text_semantic function."""
    logger.info("Testing split_text_semantic function...")
    text = "This is a test document. It has multiple sentences."
    chunks = self.uploader.split_text_semantic(text, chunk_size=50, chunk_overlap=25)
    self.assertGreater(len(chunks), 1)  # It should split into more than one chunk
    logger.info("split_text_semantic test passed.")


    def test_create_weaviate_schema(self):
        """Test the create_weaviate_schema function."""
        logger.info("Testing create_weaviate_schema function...")
        self.uploader.create_weaviate_schema()
        self.mock_client.schema.create_class.assert_called_once()
        logger.info("create_weaviate_schema test passed.")

    def test_insert_chunks(self):
        """Test the insert_chunks function."""
        logger.info("Testing insert_chunks function...")
        chunks = ["Chunk 1", "Chunk 2"]
        self.mock_embed_model.encode.return_value = [[0.1, 0.2], [0.3, 0.4]]  # Mock embedding
        self.uploader.insert_chunks(chunks)
        self.mock_client.data_object.create.assert_called()
        logger.info("insert_chunks test passed.")

    def test_run(self):
        """Test the full run function."""
        logger.info("Testing full run function...")
        with patch("app.embedding.ChunkUploader.load_text", return_value="This is a test document for chunk uploader testing.") as mock_load_text, \
             patch("app.embedding.ChunkUploader.split_text_semantic", return_value=["Chunk 1", "Chunk 2"]) as mock_split_text:
            self.uploader.run()
            mock_load_text.assert_called_once()
            mock_split_text.assert_called_once()
            logger.info("run function test passed.")

if __name__ == "__main__":
    unittest.main()
    logger.info("Test suite completed.")
