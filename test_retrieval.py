import sys
sys.path.append("src")
from vector_store import Retriever
import numpy as np

retriever = Retriever("data/vector_store")

query = "LoRA reduces trainable parameters by 10000 times compared to full fine-tuning"
query_vec = retriever.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)

target_text = None
for doc in retriever.documents:
    if doc["source_id"] == "lora" and "10,000 times" in doc["text"]:
        target_text = doc["text"]
        break

target_vec = retriever.model.encode([target_text], convert_to_numpy=True, normalize_embeddings=True)

similarity = float(np.dot(query_vec[0], target_vec[0]))
print(f"Direct similarity between query and target chunk: {similarity:.4f}")
print(f"\nTarget chunk text:\n{target_text}")