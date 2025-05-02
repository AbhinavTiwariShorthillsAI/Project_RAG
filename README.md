
# 🧠 RAG Chatbot on World War History using Wikipedia

## 📌 Project Overview

This project implements a Retrieval-Augmented Generation (RAG) pipeline tailored to answer queries related to **World War history** using data scraped from **Wikipedia**. It integrates:

- Web scraping to fetch historical content
- Semantic chunking and vector embeddings (using Sentence Transformers)
- Weaviate as a vector store
- Local LLMs (Mistral, LLaMA3 via Ollama)
- Streamlit interface for chatbot
- Evaluation pipeline with custom metrics and RAGAS-based scores

---

## 🔍 Why World War Data?

| Criteria              | Reason                                                                 |
|-----------------------|------------------------------------------------------------------------|
| 📚 Volume             | Massive structured and unstructured data on World War available online |
| ❓ Question Diversity  | Spans events, leaders, timelines, strategies, causes, and consequences |
| ✅ Relevance          | Suitable to test LLM's historical factual consistency                  |
| 🧪 Test Scope         | Enables meaningful evaluation across semantic and factual dimensions   |

---

## 🏗️ Project Architecture

![Architecture](assets/Project_architecture.png)

---

## 🕸️ Web Scraping

- Scraped the **Wikipedia page for World War II**, followed relevant internal links to extract sub-topic content.
- Used `requests`, `BeautifulSoup`, and `re` to clean data and store it in text format.

---

## 🧩 Chunking Techniques: Fixed vs Semantic

Initially, we used **fixed-size chunks** of 1000 characters with 200 character overlap. Later, we switched to **semantic chunking** using sentence boundaries.

| Feature                  | Fixed Chunking             | Semantic Chunking                |
|--------------------------|----------------------------|----------------------------------|
| 📏 Size Control          | Manual (chars)             | Automatic (sentence boundaries)  |
| 🧠 Context Preservation  | Often breaks sentences      | Retains complete ideas           |
| 🔍 Relevance             | Medium                     | High                             |
| ✅ Why Used?             | Baseline approach           | Final approach for better RAG    |

Implemented using `RecursiveCharacterTextSplitter` and later `SemanticChunker` from LangChain.

---

## 📦 Docker Volume for Persistent Weaviate

Used Docker volumes to persist Weaviate's vector store data.

```bash
docker run -d   -p 8080:8080   -v $(pwd)/weaviate_data:/var/lib/weaviate   semitechnologies/weaviate
```

**Why?** Ensures that embeddings don't need to be regenerated each run—crucial for experiments.

---

## 🧠 Embedding Models

| Model              | Size     | Speed     | Performance | Comment                                |
|--------------------|----------|-----------|-------------|----------------------------------------|
| `all-MiniLM-L6-v2` | Small    | Fast      | Basic       | Used in baseline run                   |
| `intfloat/e5-base-v2` | Medium | Moderate  | High        | Final choice: high accuracy + speed    |
| `bge-large-en-v1.5`| Large    | Slow      | SOTA        | Skipped due to resource constraints    |

✅ **Used**: `e5-base-v2` for its balance between accuracy and performance.

---

## 🦙 LLMs Used and Comparison

| Model     | Tokens | Speed  | Quality | Comment                    |
|-----------|--------|--------|---------|----------------------------|
| Mistral   | 7B     | Fast   | Mid     | Good baseline performance  |
| LLaMA 3   | 8B     | Fast   | High    | Used in final RAG pipeline |

✅ **Used**: LLaMA 3 via Ollama for generation and Gemini 2 Flash for RAGAS scoring.

---

## 🤖 Chatbot Pipeline

- Input question from Streamlit UI
- Embed query and retrieve relevant context via Weaviate
- Use retrieved text to form context-aware prompt
- Use LLM to generate final response

---

## 📏 Evaluation Metrics (Custom)

| Metric              | Type        | Description                                             |
|---------------------|-------------|---------------------------------------------------------|
| ROUGE-L             | Non-LLM     | Recall-based lexical overlap                           |
| Cosine Similarity   | Non-LLM     | Semantic similarity of embeddings                      |
| BERT F1             | LLM-based   | F1 score using pre-trained BERT embeddings             |
| Final Score         | Weighted    | Composite score from all three metrics                 |

**Weights Used:**
```python
METRIC_WEIGHTS = {
    "rouge_score": 0,
    "cosine_similarity": 0.4,
    "bert_score_f1": 0.6
}
```

---

## 📊 Evaluation Comparison: First Run vs Final Run

### 1. Custom Metric-Based Scores

| Metric           | Mistral (First Run) | LLaMA 3 (Second Run) |
|------------------|---------------------|-----------------------|
| ROUGE Score      | 0.3768              | 0.6758                |
| Cosine Similarity| 0.6119              | 0.9166                |
| BERT Score (F1)  | 0.6040              | 0.7443                |
| Final Score      | 0.5723              | 0.8132                |
| Grade            | F (Poor)            | B (Good)              |

🎯 **Improvement Observed in every metric** after:
- Switching to semantic chunking
- Switching to `e5-base-v2`
- Switching from Mistral to LLaMA 3

---

### 2. RAGAS Evaluation (Final RAG)

| Metric             | Score    |
|--------------------|----------|
| Faithfulness       | 0.9194   |
| Answer Relevancy   | 0.9432   |
| Context Precision  | 0.7484   |
| Context Recall     | 0.8970   |

✅ Computed using Gemini 2.0 Flash for better judgment and reproducibility

---

## ✅ Summary of Enhancements

| Area              | First Run              | Second Run (Final)     | Improvement                     |
|-------------------|------------------------|-------------------------|----------------------------------|
| Chunking          | Fixed size             | Semantic                | Better context + coherence       |
| Embedding Model   | MiniLM                 | e5-base-v2              | More accurate + faster than BGE  |
| LLM Used          | Mistral                | LLaMA 3 (Ollama)        | Higher quality, better context   |
| Evaluation        | Custom only            | Custom + RAGAS          | More reliable + diverse metrics  |

---

## 💬 Final Thoughts

This RAG chatbot system effectively leverages semantic chunking, smart embedding models, and lightweight LLMs to answer complex questions on historical data with improved accuracy and context. The use of **semantic chunking** and **LLaMA 3** showed significant gains in answer quality, as seen from the jump in evaluation scores.

---

**Made with ❤️ by Abhinav**
