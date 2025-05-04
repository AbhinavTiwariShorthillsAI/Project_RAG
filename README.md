# 🧠 RAG Chatbot on World War History using Wikipedia

## 📚 Table of Contents
- [Project Overview](#-project-overview)
- [Why World War Data](#-why-world-war-data)
- [Project Architecture](#-project-architecture)
- [Project Structure](#-project-structure)
- [Logging System](#-logging-system)
- [Web Scraping](#-web-scraping)
- [Chunking Techniques](#-chunking-techniques-fixed-vs-semantic)
- [Docker Volume for Persistent Weaviate](#-docker-volume-for-persistent-weaviate)
- [Embedding Models](#-embedding-models)
- [LLMs Used and Comparison](#-llms-used-and-comparison)
- [Chatbot Pipeline](#-chatbot-pipeline)
- [Technical Requirements and Setup](#-technical-requirements-and-setup)
- [Evaluation Metrics](#-evaluation-metrics-custom)
- [Comparison Analysis of Improvements](#-comparison-analysis-of-improvements)
- [Summary of Enhancements](#-summary-of-enhancements)
- [Future Work](#-future-work)
- [Final Thoughts](#-final-thoughts)

## 📌 Project Overview

This project implements a Retrieval-Augmented Generation (RAG) pipeline tailored to answer queries related to **World War history** using data scraped from **Wikipedia**. It integrates:

- Web scraping to fetch historical content
- Semantic chunking and vector embeddings (using Sentence Transformers)
- Weaviate as a vector store
- Local LLMs (Mistral, LLaMA3 via Ollama)
- Streamlit interface for chatbot
- Evaluation pipeline with custom metrics and RAGAS-based scores
- Comprehensive logging system for all components


## 🔍 Why World War Data?

| Criteria              | Reason                                                                 |
|-----------------------|------------------------------------------------------------------------|
| 📚 Volume             | Massive structured and unstructured data on World War available online |
| ❓ Question Diversity  | Spans events, leaders, timelines, strategies, causes, and consequences |
| ✅ Relevance          | Suitable to test LLM's historical factual consistency                  |
| 🧪 Test Scope         | Enables meaningful evaluation across semantic and factual dimensions   |

## 🏗️ Project Architecture

![Architecture](assets/Project_architecture.png)

## 📂 Project Structure

The project follows a modular structure for better organization:

```
Project_RAG/
├── app/                  # Core application components
│   ├── embedding.py      # Handles text embeddings 
│   └── output/           # Log files for app components
│
├── data/                 # Data storage and organization
│   ├── raw/              # Raw scraped text (e.g., modern_history_combined.txt)
│   ├── qa_pairs/         # Generated question-answer pairs (CSV)
│   ├── processed/        # Processed data files (e.g., predictions)
│   └── evaluation/       # Evaluation results and metrics
│
├── scripts/              # Implementation scripts
│   ├── scraper.py        # Web scraping implementation
│   ├── question_gen.py   # Question generation using LLMs
│   ├── testing_data.py   # RAG question answering implementation
│   ├── testing.py        # Custom evaluation implementation
│   ├── raga_test.py      # RAGAS evaluation implementation 
│   ├── ragas_test_data.py# Process RAGAS evaluation data
│   └── output/           # Log files for scripts
│
├── test/                 # Test suites and test cases
│   ├── embedding_test.py # Tests for embedding functionality
│   ├── question_gen_test.py # Tests for question generation
│   ├── rag_qa_test.py    # Tests for RAG QA pipeline
│   ├── scraper_test.py   # Tests for web scraper
│   ├── ragas_test_data_test.py # Tests for RAGAS data handling
│   ├── testing_test.py   # Tests for evaluation metrics
│   └── logs/             # Log files for test execution
│
├── streamlit_app/        # Streamlit UI components
│
├── weaviate_data/        # Persistent storage for vector database
│
└── requirements.txt      # Project dependencies
```

## 📝 Logging System

The project implements a comprehensive logging system across all components:

### Key Features:

- **Consistent Format**: All logs follow a standardized format with timestamp, module name, log level, and message
- **Local Log Storage**: Each component stores logs in its local output directory:
  - Implementation files (scripts/app): Logs in `output/` subdirectory
  - Test files: Logs in `test/logs/` directory
- **Configurable Levels**: Default logging level is INFO, capturing normal operations and errors
- **Dual Output**: Logs are written to both files and console output
- **Error Tracking**: All exceptions are properly logged with traceback information
- **Component-Specific Logs**: Each component has its own log file for easier debugging and monitoring

### Example Log Configuration:

```python
import logging
import os

# Set up logging
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, 'output')
os.makedirs(output_dir, exist_ok=True)
log_file = os.path.join(output_dir, 'component_name.log')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
```

## 🕸️ Web Scraping

- Scraped the **Wikipedia page for World War II**, followed relevant internal links to extract sub-topic content.
- Used `requests`, `BeautifulSoup`, and `re` to clean data and store it in text format.


## 🧩 Chunking Techniques: Fixed vs Semantic

Initially, we used **fixed-size chunks** of 1000 characters with 200 character overlap. Later, we switched to **semantic chunking** using sentence boundaries.

| Feature                  | Fixed Chunking             | Semantic Chunking                |
|--------------------------|----------------------------|----------------------------------|
| 📏 Size Control          | Manual (chars)             | Automatic (sentence boundaries)  |
| 🧠 Context Preservation  | Often breaks sentences     | Retains complete ideas           |
| 🔍 Relevance             | Medium                     | High                             |
| ✅ Why Used?             | Baseline approach          | Final approach for better RAG    |

Implemented using `RecursiveCharacterTextSplitter` and later `SemanticChunker` from LangChain.


## 📦 Docker Volume for Persistent Weaviate

Used Docker volumes to persist Weaviate's vector store data.

```bash
docker run -d   -p 8080:8080   -v $(pwd)/weaviate_data:/var/lib/weaviate   semitechnologies/weaviate
```

**Why?** Ensures that embeddings don't need to be regenerated each run—crucial for experiments.


## 🧠 Embedding Models

| Model               | Size     | Speed     | Performance | Comment                           |
|---------------------|----------|-----------|-------------|-----------------------------------|
| `all-MiniLM-L6-v2`  | Small    | Fast      | Basic       | Used in baseline run              |
| `intfloat/e5-base-v2`| Medium  | Moderate  | High        | Final choice: high accuracy + speed|
| `bge-large-en-v1.5` | Large    | Slow      | SOTA        | Skipped due to resource constraints|

✅ **Used**: `e5-base-v2` for its balance between accuracy and performance.


## 🦙 LLMs Used and Comparison

| Model     | Tokens | Speed  | Quality | Comment                      |
|-----------|--------|--------|---------|------------------------------|
| Mistral   | 7B     | Fast   | Mid     | Good baseline performance    |
| LLaMA 3   | 8B     | Fast   | High    | Used in final RAG pipeline   |

✅ **Used**: LLaMA 3 via Ollama for generation and Gemini 2 Flash for RAGAS scoring.


## 🤖 Chatbot Pipeline

- Input question from Streamlit UI
- Embed query and retrieve relevant context via Weaviate
- Use retrieved text to form context-aware prompt
- Use LLM to generate final response

## 🛠️ Technical Requirements and Setup

### Prerequisites
- Python 3.8+
- Docker for Weaviate
- 16GB+ RAM recommended for optimal performance

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rag-worldwar-chatbot.git
cd rag-worldwar-chatbot
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Start Weaviate using Docker:
```bash
docker run -d -p 8080:8080 -v $(pwd)/weaviate_data:/var/lib/weaviate semitechnologies/weaviate
```

4. Install Ollama (for local LLM):
```bash
# For macOS/Linux
curl -fsSL https://ollama.com/install.sh | sh
# Then pull the LLaMA 3 model
ollama pull llama3
```

### Running the Application

1. Process the data pipeline:
```bash
# Scrape and embed data (if not already done)
python3 scripts/scraper.py
python3 app/embedding.py

# Generate QA pairs
python3 scripts/question_gen.py

# Run RAG processing to generate answers
python3 scripts/testing_data.py
```

2. Start the Streamlit application:
```bash
cd streamlit_app
streamlit run app.py
```

3. Access the chatbot UI at http://localhost:8501

### Running Tests
```bash
# Run individual test
python3 -m unittest test/embedding_test.py

# Run all tests
python3 -m unittest discover -s test
```

# 📏 Evaluation Metrics (Custom)

| Metric              | Type        | Description                                                             |
|---------------------|-------------|-------------------------------------------------------------------------|
| ROUGE-L             | Non-LLM     | Measures recall-based lexical overlap between generated and reference text |
| Cosine Similarity   | Non-LLM     | Assesses semantic similarity between embeddings of generated and reference text |
| BERT F1             | LLM-based   | Evaluates semantic similarity using F1 score from pre-trained BERT embeddings |
| BLEU                | Non-LLM     | Measures n-gram precision for lexical similarity, often used in translation tasks |
| METEOR              | Non-LLM     | Considers synonyms, stemming, and word order for lexical matching |
| Levenshtein         | Non-LLM     | Measures edit distance to quantify character-level differences |
| Final Score         | Weighted    | Composite score combining all metrics based on assigned weights |

**Weights Used:**

```python
METRIC_WEIGHTS = {
    "rouge_score": 0.15,
    "cosine_similarity": 0.35,
    "bert_score_f1": 0.30,
    "bleu": 0.05,
    "meteor": 0.10,
    "levenshtein": 0.05
}
```

**Rationale for Weightages:**
- **Cosine Similarity (0.35)**: Highest weight due to its focus on semantic similarity via embeddings, crucial for evaluating meaning preservation in context-rich tasks.
- **BERT Score F1 (0.30)**: Significant weight for its deep semantic evaluation using BERT embeddings, ensuring robust contextual alignment.
- **ROUGE-L (0.15)**: Moderate weight as it captures lexical overlap, useful for surface-level similarity but less effective for semantic nuances.
- **METEOR (0.10)**: Lower weight since it complements lexical evaluation with synonyms and stemming, but is less critical than semantic metrics.
- **BLEU (0.05)**: Minimal weight due to its strict n-gram precision, which may penalize valid variations in wording.
- **Levenshtein (0.05)**: Lowest weight as it focuses on character-level edits, less relevant for high-level text quality but useful for syntactic errors.

The weight distribution emphasizes semantic understanding (Cosine Similarity, BERT Score) over lexical or character-level metrics, aligning with the goal of evaluating meaningful text generation.

# 📈 Comparison Analysis of Improvements

The transition from the first run (Mistral) to the second run (LLaMA 3) demonstrates significant enhancements across both custom and RAGAS metrics, driven by strategic changes in methodology. The bar charts below, stored in the `assets` folder, visually depict these improvements, with percentage increases annotated above the LLaMA 3 bars for clarity.


## 1. Custom Metrics Improvement

![Custom Metrics Comparison](assets/custom_metrics.png)

| Metric              | Mistral (First Run) | LLaMA 3 (Second Run) | Improvement (%) |
|---------------------|---------------------|----------------------|-----------------|
| ROUGE Score         | 0.3768              | 0.6678               | 77.2%           |
| Cosine Similarity   | 0.8669              | 0.9153               | 5.6%            |
| BERT Score (F1)     | 0.6040              | 0.7409               | 22.7%           |
| BLEU                | 0.1378              | 0.2327               | 68.9%           |
| METEOR              | 0.3393              | 0.4555               | 34.2%           |
| Levenshtein         | 0.4030              | 0.6843               | 69.8%           |
| **Final Score**     | **0.6021**          | **0.7342**           | **21.9%**       |

**Insights**:
- **ROUGE-L**: The largest improvement (+77.2%) is evident in the significantly taller LLaMA 3 bar, indicating enhanced lexical overlap between generated and reference text.
- **Cosine Similarity**: A modest gain (+5.6%) is shown by a slight increase in bar height, reflecting a high baseline semantic alignment.
- **BERT Score (F1)**: A notable increase (+22.7%) is visible in the extended LLaMA 3 bar, demonstrating improved deep semantic similarity.
- **BLEU** and **Levenshtein**: Strong improvements (+68.9% and +69.8%) are highlighted by the taller LLaMA 3 bars, indicating better n-gram precision and syntactic alignment.
- **Final Score**: The overall improvement (+21.9%) is clear in the bar chart, with the LLaMA 3 bar surpassing Mistral's, reflecting a comprehensive performance boost.

## 2. RAGAS Metrics Improvement

![RAGAS Metrics Comparison](assets/ragas_metrics.png)

| Metric              | Mistral (First Run) | LLaMA 3 (Second Run) | Improvement (%) |
|---------------------|---------------------|----------------------|-----------------|
| Faithfulness        | 0.7381              | 0.9194               | 24.6%           |
| Answer Relevancy    | 0.7689              | 0.9432               | 22.7%           |
| Context Precision   | 0.4974              | 0.7484               | 50.5%           |
| Context Recall      | 0.6067              | 0.8970               | 47.8%           |

**Insights**:
- **Context Precision**: The largest improvement (+50.5%) is shown by the significantly taller LLaMA 3 bar, indicating more relevant retrieved context.
- **Context Recall**: A strong gain (+47.8%) is visible in the extended LLaMA 3 bar, reflecting better retrieval of relevant context.
- **Faithfulness**: An improvement of +24.6% is evident in the taller LLaMA 3 bar, indicating reduced hallucinations and better context adherence.
- **Answer Relevancy**: A +22.7% increase is shown by the higher LLaMA 3 bar, demonstrating improved alignment with query intent.


## 3. Analysis of Improvements

The bar charts, with percentage increases annotated above the LLaMA 3 bars, visually highlight the improvements from the first run (Mistral) to the second run (LLaMA 3), driven by strategic changes:

1. **Semantic Chunking**:
   - **First Run**: Fixed-size chunking led to fragmented context, resulting in low Context Precision (0.4974) and ROUGE-L (0.3768), as seen in the shorter Mistral bars.
   - **Second Run**: Semantic chunking improved coherence, boosting Context Recall to 0.8970 (+47.8%) and ROUGE-L to 0.6678 (+77.2%), as shown by the taller LLaMA 3 bars with prominent percentage annotations.
   - **Impact**: The significant height increase in ROUGE-L and Context Recall bars reflects enhanced context preservation and coherence.

2. **Embedding Model (MiniLM → e5-base-v2)**:
   - **First Run**: MiniLM's limited embeddings yielded moderate semantic similarity (Cosine Similarity: 0.8669, BERT Score: 0.6040), visible in the shorter Mistral bars.
   - **Second Run**: e5-base-v2 enhanced semantic alignment, increasing Cosine Similarity to 0.9153 (+5.6%) and BERT Score to 0.7409 (+22.7%), supporting higher Answer Relevancy (0.9432), as seen in the taller LLaMA 3 bars.
   - **Impact**: The steady growth in Cosine Similarity and BERT Score bars, with +5.6% and +22.7% annotations, indicates improved semantic understanding.

3. **LLM Upgrade (Mistral → LLaMA 3)**:
   - **First Run**: Mistral's limitations resulted in lower Faithfulness (0.7381) and lexical matching (BLEU: 0.1378, METEOR: 0.3393), reflected in the shorter Mistral bars.
   - **Second Run**: LLaMA 3's superior context handling improved Faithfulness to 0.9194 (+24.6%) and METEOR to 0.4555 (+34.2%), as shown by the taller LLaMA 3 bars with clear percentage annotations.
   - **Impact**: The significant increases in Faithfulness and METEOR bars highlight LLaMA 3's enhanced contextual accuracy and text quality.

4. **Consistent Evaluation Framework**:
   - Both runs were evaluated using the same custom metrics and RAGAS evaluation methodology
   - The second run showed significant improvements across all metrics, with Context Precision having the largest gain (+50.5%)
   - The consistent evaluation framework provides an objective basis for comparing the performance improvements between runs

**Quantitative Impact**:
- **Custom Metrics**: The Final Score improved from 0.6021 (D) to 0.7342 (C), with ROUGE-L (+77.2%) and Levenshtein (+69.8%) showing the largest gains, as seen in the significantly taller LLaMA 3 bars with percentage annotations in the custom metrics chart.
- **RAGAS Metrics**: Context Precision improved most significantly (+50.5%), followed by Context Recall (+47.8%), as visualized by the extended LLaMA 3 bars with percentage annotations in the RAGAS chart.
- The consistent upward trend across all metrics, depicted in the bar charts with clear percentage increases, highlights the effectiveness of the upgrades.

**Conclusion**:
The second run's enhancements in chunking, embeddings, and LLM choice resulted in more coherent, relevant, and contextually accurate outputs, as visually evident in the taller LLaMA 3 bars with annotated percentage increases in the charts. The expanded evaluation framework, with comparable RAGAS metrics, validated these improvements, providing a robust basis for future optimizations.


# ✅ Summary of Enhancements

| Area              | First Run              | Second Run (Final)      | Improvement                      |
|-------------------|------------------------|-------------------------|----------------------------------|
| Chunking          | Fixed size             | Semantic                | Better context + coherence       |
| Embedding Model   | MiniLM                 | e5-base-v2              | More accurate + faster than BGE  |
| LLM Used          | Mistral                | LLaMA 3 (Ollama)        | Higher quality, better context   |
| Evaluation        | Custom + RAGAS         | Custom + RAGAS          | Same methodology, better results |


# 🔮 Future Work

- **BERT-based Chunking**: Implement more sophisticated semantic chunking using BERT embeddings to better preserve contextual boundaries and improve retrieval precision
- **Query Caching**: Add caching mechanism for previously asked questions to significantly increase response speeds for common or similar queries
- **UI Improvements**: Enhance the Streamlit interface with source citations, confidence scores, and better visualization of retrieved context

---

## 💬 Final Thoughts

This RAG chatbot system effectively leverages semantic chunking, smart embedding models, and lightweight LLMs to answer complex questions on historical data with improved accuracy and context. The use of **semantic chunking** and **LLaMA 3** showed significant gains in answer quality, as seen from the jump in evaluation scores.

**Made with ❤️ by Abhinav**
