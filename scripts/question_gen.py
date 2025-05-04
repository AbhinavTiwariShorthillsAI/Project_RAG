import os
import requests
import pandas as pd
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from time import sleep
from typing import List

# Set up logging
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, 'output')
os.makedirs(output_dir, exist_ok=True)
log_file = os.path.join(output_dir, 'question_gen.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"

class LLaMAQuestionGenerator:
    """
    Class for generating Q&A pairs from historical text using a local LLaMA model via Ollama.
    
    Methods:
        load_text(path): Load text content from a file.
        split_text_semantic(text): Split text into semantically coherent chunks.
        query_llama(prompt): Query the LLaMA model with a prompt.
        generate_qa_pairs(chunks): Generate questions and answers from chunks.
        save_to_csv(df, output_path): Save generated Q&A pairs to a CSV file.
    """

    def __init__(self, model: str = OLLAMA_MODEL, url: str = OLLAMA_URL):
        """
        Initialize the generator with model and Ollama API URL.
        
        Args:
            model (str): Name of the model to query.
            url (str): URL endpoint for Ollama.
        """
        self.model = model
        self.url = url

    def load_text(self, path: str) -> str:
        """
        Load text from a specified file path.

        Args:
            path (str): File path to the text document.

        Returns:
            str: The content of the text file.
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"❌ File not found: {path}")
            return ""
        except Exception as e:
            logger.error(f"❌ Error loading text: {e}")
            return ""

    def split_text_semantic(self, text: str, chunk_size: int = 800, chunk_overlap: int = 100) -> List[str]:
        """
        Split text into semantically meaningful chunks using common punctuation.

        Args:
            text (str): Full text to split.
            chunk_size (int): Max size of each chunk.
            chunk_overlap (int): Number of overlapping characters between chunks.

        Returns:
            List[str]: List of text chunks.
        """
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", ".", "!", "?", "\n", " "],
                length_function=len
            )
            return splitter.split_text(text)
        except Exception as e:
            logger.error(f"❌ Error splitting text: {e}")
            return []

    def query_llama(self, prompt: str) -> str:
        """
        Query the local LLaMA model via Ollama.

        Args:
            prompt (str): Input prompt to send to the model.

        Returns:
            str: Response text from the model.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        try:
            response = requests.post(self.url, json=payload)
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except requests.exceptions.ConnectionError:
            return "Error: Unable to connect to Ollama API. Is it running?"
        except Exception as e:
            return f"Error: {e}"

    def generate_qa_pairs(self, chunks: List[str], total: int = 500) -> pd.DataFrame:
        """
        Generate Q&A pairs from chunks of text.

        Args:
            chunks (List[str]): List of text chunks to process.
            total (int): Number of chunks to process (default 500).

        Returns:
            pd.DataFrame: DataFrame with "Question" and "Answer" columns.
        """
        questions, answers = [], []
        total = min(total, len(chunks))
        logger.info("\U0001F680 Generating Q&A...")

        for i, chunk in enumerate(chunks[:total]):
            prompt = f"""
Based only on the following historical text, generate 2 simple factual questions and their short, correct answers. 
Output format strictly like:
Question: <your question>
Answer: <your answer>

Historical text:
{chunk}
"""
            try:
                response_text = self.query_llama(prompt)
                output_lines = response_text.strip().split("\n")

                current_question, current_answer = "", ""
                for line in output_lines:
                    if "Question:" in line:
                        if current_question and current_answer:
                            questions.append(current_question)
                            answers.append(current_answer)
                            current_question, current_answer = "", ""
                        current_question = line.split(":", 1)[1].strip()
                    elif "Answer:" in line:
                        current_answer = line.split(":", 1)[1].strip()

                if current_question and current_answer:
                    questions.append(current_question)
                    answers.append(current_answer)

                percent = int(((i + 1) / total) * 100)
                logger.info(f"Progress: {percent}% ({i+1}/{total})")
                sleep(0.2)
            except Exception as e:
                logger.error(f"\n❌ Error in chunk {i+1}: {e}")

        logger.info("\n✅ Q&A generation complete!")
        return pd.DataFrame({"Question": questions, "Answer": answers})

    def save_to_csv(self, df: pd.DataFrame, output_path: str) -> None:
        """
        Save the Q&A DataFrame to a CSV file.

        Args:
            df (pd.DataFrame): DataFrame containing the questions and answers.
            output_path (str): Path to save the CSV file.
        """
        try:
            df.to_csv(output_path, index=False)
            logger.info(f"✅ Saved {len(df)} Q&A pairs to {output_path}")
        except Exception as e:
            logger.error(f"❌ Error saving CSV: {e}")


if __name__ == "__main__":
    generator = LLaMAQuestionGenerator()
    
    # Get the correct paths using the project structure
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(project_root, "data", "raw", "modern_history_combined.txt")
    output_file = os.path.join(project_root, "data", "qa_pairs", "qa_dataset_1000_part2.csv")
    
    raw_text = generator.load_text(input_file)
    if raw_text:
        chunks = generator.split_text_semantic(raw_text)
        if chunks:
            qa_df = generator.generate_qa_pairs(chunks)
            generator.save_to_csv(qa_df, output_file)
