from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, chunks, top_k=5):
        # Create (query, chunk) pairs
        pairs = [(query, chunk["text"]) for chunk in chunks]

        # Get relevance scores
        scores = self.model.predict(pairs)

        # Combine chunks with scores
        scored_chunks = list(zip(chunks, scores))

        # Sort by score (descending)
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        # Select top_k
        top_chunks = [chunk for chunk, _ in scored_chunks[:top_k]]

        return top_chunks