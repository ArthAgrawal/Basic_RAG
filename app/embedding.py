import numpy as np
import os
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name="all-miniLM-L6-v2"):  # Using a smaller model for faster embedding generation, use 'all-mpnet-base-v2' for better quality if you have the resources
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, chunks):
        texts = [chunk["text"] for chunk in chunks]  # Extracting only the text field from each chunk's dictionary(metadata)
        return self.model.encode(texts, show_progress_bar=True)
    
    def embed_query(self, query):
        return self.model.encode([query])[0]  # Return the single embedding vector instead of a list
    
    def save_embeddings(self, embeddings, path="storage/embeddings.npy"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, embeddings)

    def load_embeddings(self, path="storage/embeddings.npy"):
        if os.path.exists(path):
            return np.load(path)
        return None