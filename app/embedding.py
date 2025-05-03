import numpy as np
import weaviate
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

class ChunkUploader:
    """
    A utility class to process large text documents, chunk them semantically, 
    embed them using a SentenceTransformer, and upload the data to Weaviate.

    Attributes:
        text_file (str): Path to the text file to be processed.
        embedding_model_name (str): Name of the SentenceTransformer model to use for embedding.
        weaviate_url (str): URL of the Weaviate instance.
        class_name (str): Weaviate class name to insert the data into.
        embed_model (SentenceTransformer): The sentence transformer model for creating embeddings.
        client (weaviate.Client): The Weaviate client instance for database operations.
    """

    def __init__(self, 
                 text_file: str,
                 embedding_model_name: str = "intfloat/e5-base-v2",
                 weaviate_url: str = "http://localhost:8080",
                 class_name: str = "WorldWarChunk") -> None:
        """
        Initializes the ChunkUploader with file paths, model, and DB config.

        Args:
            text_file (str): Path to the text file to be processed.
            embedding_model_name (str, optional): Name of the SentenceTransformer model. 
                Defaults to "intfloat/e5-base-v2".
            weaviate_url (str, optional): URL of the Weaviate instance. 
                Defaults to "http://localhost:8080".
            class_name (str, optional): Weaviate class name to insert the data into. 
                Defaults to "WorldWarChunk".
                
        Raises:
            Exception: If loading the embedding model or connecting to Weaviate fails.
        """
        self.text_file = text_file
        self.class_name = class_name
        try:
            self.embed_model = SentenceTransformer(embedding_model_name)
        except Exception as e:
            print(f"Error loading embedding model {embedding_model_name}: {e}")
            raise

        try:
            self.client = weaviate.Client(url=weaviate_url)
        except Exception as e:
            print(f"Error connecting to Weaviate instance at {weaviate_url}: {e}")
            raise

    def load_text(self) -> str:
        """
        Reads and returns the full contents of the text file.

        Returns:
            str: The raw text from the file.
            
        Raises:
            Exception: If the file cannot be read or processed.
        """
        try:
            with open(self.text_file, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"Error reading file {self.text_file}: {e}")
            raise

    def split_text_semantic(self, text: str, chunk_size: int = 800, chunk_overlap: int = 100) -> List[str]:
        """
        Splits the input text into semantic chunks for better retrieval quality.

        Args:
            text (str): The full document text.
            chunk_size (int, optional): Maximum length of each chunk. Defaults to 800.
            chunk_overlap (int, optional): Overlap between consecutive chunks. Defaults to 100.

        Returns:
            List[str]: List of text chunks.
            
        Raises:
            Exception: If the text splitting process fails.
        """
        try:
            # Ensure chunk_overlap is always smaller than chunk_size
            valid_overlap = min(chunk_overlap, chunk_size - 1)

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=valid_overlap,
                separators=["\n\n", ".", "!", "?", "\n", " "],
                length_function=len
            )

            chunks = splitter.split_text(text)

            # Fallback: if splitter fails to produce any chunk (e.g., empty string), return raw text
            return chunks if chunks else [text]

        except Exception as e:
            print(f"Error splitting text into chunks: {e}")
            raise

    def create_weaviate_schema(self) -> None:
        """
        Clears existing schema and sets up a new Weaviate class with manual vectorization.
        
        Creates a class in Weaviate with a 'text' property and configures it for
        manual vector management (no automatic vectorization).
        
        Raises:
            Exception: If schema creation in Weaviate fails.
        """
        try:
            self.client.schema.delete_all()
            class_obj = {
                "class": self.class_name,
                "vectorizer": "none",
                "properties": [
                    {
                        "name": "text",
                        "dataType": ["text"]
                    }
                ]
            }
            self.client.schema.create_class(class_obj)
        except Exception as e:
            print(f"Error creating Weaviate schema: {e}")
            raise

    def insert_chunks(self, chunks: List[str]) -> None:
        """
        Generates vector embeddings for each chunk and inserts them into Weaviate.

        Args:
            chunks (List[str]): List of preprocessed text chunks.
            
        Raises:
            Exception: If embedding generation or Weaviate insertion fails.
        """
        try:
            embeddings = self.embed_model.encode(chunks, show_progress_bar=True)
            embeddings = np.array(embeddings).astype(np.float32)

            for chunk, embedding in zip(chunks, embeddings):
                self.client.data_object.create(
                    data_object={"text": chunk},
                    class_name=self.class_name,
                    vector=embedding.tolist()
                )
        except Exception as e:
            print(f"Error inserting chunks into Weaviate: {e}")
            raise

    def run(self) -> None:
        """
        Executes the full processing pipeline.
        
        Steps:
        1. Loads text from file
        2. Splits into semantic chunks
        3. Creates schema in Weaviate
        4. Inserts data and embeddings into the vector store
        
        Raises:
            Exception: If any step in the processing pipeline fails.
        """
        try:
            print("📄 Reading and processing text file...")
            text = self.load_text()
            chunks = self.split_text_semantic(text)

            print("🧱 Setting up Weaviate schema...")
            self.create_weaviate_schema()

            print("📤 Uploading chunks to Weaviate...")
            self.insert_chunks(chunks)
            print("✅ All chunks inserted into Weaviate!")
        except Exception as e:
            print(f"Error during the processing pipeline: {e}")
            raise

if __name__ == "__main__":
    try:
        uploader = ChunkUploader("/home/shtlp_0012/codes/RAG/modern_history_combined.txt")
        uploader.run()
    except Exception as e:
        print(f"Error during script execution: {e}")