# 🧠 PROJECT_RAG – Semantic Retrieval-Augmented Generation (RAG) System for Historical Q&A

This project is a **Retrieval-Augmented Generation (RAG)** system designed to answer questions from historical documents (like World War history) using local models and open-source tools.

---

## 🚀 Key Features

- 🔎 **Semantic Chunking**: Smart text splitting using `RecursiveCharacterTextSplitter`
- 🧠 **Embeddings**: Generated with `intfloat/e5-base-v2`
- 🔍 **Weaviate Vector DB**: Fast and scalable vector storage + search
- 🧮 **Cross-Encoder Re-ranking**: Improves retrieval accuracy
- 💬 **Local LLM (LLaMA3 via Ollama)**: Offline inference with high quality
- 📊 **Evaluation Metrics**: ROUGE-L, cosine similarity, BERTScore-F1
- 🧪 **Streamlit UI**: Ask questions interactively with real-time answers

---

## 🧱 Project Structure

```
PROJECT_RAG/
├── app/                    # Core logic
│   └── embedding.py        # Chunk creation + upload
│
├── streamlit_app/
│   └── app.py              # Streamlit chatbot
│
├── scripts/
│   ├── question_gen.py     # Auto-generate Q&A
│   ├── test.py             # Evaluation pipeline
│   └── testing_data.py     # Predict answers for questions
│
├── data/
│   ├── modern_history_combined.txt
│   ├── qa_dataset_1000.csv
│   ├── qa_with_predictions.xlsx
│   └── qa_evaluated_scores.xlsx
│
├── .env                    # Environment variables
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone and Setup

```bash
git clone https://github.com/AbhinavTiwariShorthillsAI/PROJECT_RAG.git
cd PROJECT_RAG
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure `.env`

Create a `.env` file with:

```
OLLAMA_MODEL=llama3
DATASET_PATH=data/qa_dataset_1000.csv
HISTORY_FILE=conversation_history.json
```

### 3. Start Ollama and Weaviate

```bash
# Start Ollama (with LLaMA3 model)
ollama run llama3

# Start Weaviate with Docker
docker run -d -p 8080:8080 semitechnologies/weaviate:latest
```

---

## 🧪 How to Use

### ✅ Index Text into Weaviate

```bash
python app/embedding.py
```

### ✅ Generate Questions + Answers

```bash
python scripts/question_gen.py
```

### ✅ Generate Predictions for Existing Qs

```bash
python scripts/testing_data.py
```

### ✅ Evaluate Model Accuracy

```bash
python scripts/test.py
```

### ✅ Launch Streamlit Chatbot

```bash
streamlit run streamlit_app/app.py
```

---

## 📊 Evaluation Metrics

| Metric           | Description                                 |
|------------------|---------------------------------------------|
| **ROUGE-L**      | Overlap in word sequences (surface match)   |
| **Cosine Similarity** | Vector-based semantic similarity       |
| **BERTScore F1** | Deep contextual alignment with reference    |
| **Final Score**  | Weighted composite of all above             |

Grades (A to F) are computed from final score thresholds.

---

## 📦 Requirements

Install all via:

```bash
pip install -r requirements.txt
```

---


## 🤝 Contributions

PRs and issues are welcome! Please open a discussion if you'd like to improve this project.

---

## 📄 License

MIT License © 2025 [Your Name]
