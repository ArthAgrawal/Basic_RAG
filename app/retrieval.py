import faiss
import numpy as np
import os


class VectorStore:
    def __init__(self, embeddings, chunks):
        self.chunks = chunks
        
        # Convert embeddings to numpy array
        self.embeddings = np.array(embeddings).astype("float32")
        
        # Get dimension
        dim = self.embeddings.shape[1]
        
        # Create FAISS index (L2 distance)
        self.index = faiss.IndexFlatL2(dim)
        
        # Add embeddings to index
        self.index.add(self.embeddings)

    def search(self, query_embedding, top_k=3):
        query_embedding = np.array([query_embedding]).astype("float32")
        
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for i in indices[0]:
            results.append(self.chunks[i]) # returns full chunk dictionary(metadata) now instead of just text
        
        return results
    
    def save_index(self, path="storage/index.faiss"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        faiss.write_index(self.index, path)

    def load_index(self, path="storage/index.faiss"):
        if os.path.exists(path):
            self.index = faiss.read_index(path)
            return True
        return False