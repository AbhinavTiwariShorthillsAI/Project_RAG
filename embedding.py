
from sentence_transformers import SentenceTransformer
import numpy as np
import weaviate

client = weaviate.Client(url="http://localhost:8080")

# 1. Read text file
with open("/home/shtlp_0012/codes/RAG/modern_history_combined.txt", "r") as f:
    text = f.read()

# 2. Split text into chunks
def split_text(text, chunk_size=1000, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

chunks = split_text(text)

# 3. Create embeddings and insert
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embed_model.encode(chunks, show_progress_bar=True)
embeddings = np.array(embeddings).astype(np.float32)

# OPTIONAL: Clear old schema if needed
client.schema.delete_all()

# Create new class schema
class_obj = {
    "class": "WorldWarChunk",
    "vectorizer": "none",   # We provide vectors manually
    "properties": [
        {
            "name": "text",
            "dataType": ["text"]
        }
    ]
}
client.schema.create_class(class_obj)

# Insert chunks + vectors
for chunk, embedding in zip(chunks, embeddings):
    client.data_object.create(
        data_object={"text": chunk},
        class_name="WorldWarChunk",
        vector=embedding.tolist()
    )

print("✅ All chunks inserted into Weaviate!")
