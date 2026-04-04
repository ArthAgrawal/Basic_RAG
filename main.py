from app.ingestion import get_chunks
from app.embedding import Embedder
from app.retrieval import VectorStore
from app.generation import Generator
from app.query_expansion import QueryExpander
from app.reranker import Reranker


if __name__ == "__main__":
    # Step 1: Load & chunk documents
    chunks = get_chunks("data/docs")

    # Step 2: Initialize embedder
    embedder = Embedder()

    # Step 3: Try loading embeddings
    embeddings = embedder.load_embeddings()

    if embeddings is None:
        print("\nGenerating embeddings...")
        embeddings = embedder.embed_documents(chunks)
        embedder.save_embeddings(embeddings)
    else:
        print("\nLoaded embeddings from disk.")

    # Step 4: Initialize vector store
    vector_store = VectorStore(embeddings, chunks)

    # Step 5: Try loading FAISS index
    if vector_store.load_index():
        print("Loaded FAISS index from disk.")
    else:
        print("Building FAISS index...")
        vector_store.save_index()

    # Step 6: Initialize components
    generator = Generator()
    expander = QueryExpander()
    reranker = Reranker()

    # Step 7: Query loop
    while True:
        query = input("\nAsk a question (or type 'exit'): ")

        if query.lower() == "exit":
            break

        # Step 1: Expand query (include original)
        expanded = expander.expand(query)
        expanded_queries = [query] + expanded if expanded else [query]

        print("\nExpanded Queries:")
        for q in expanded_queries:
            print("-", q)

        # Step 2: Retrieve for each query
        all_results = []

        for q in expanded_queries:       # Here, we basically go through each query and expanded query separately and get results for each of them 
            q_embedding = embedder.embed_query(q) 
            results = vector_store.search(q_embedding, top_k=5)
            all_results.extend(results)

        # Step 3: Deduplicate
        seen = set()
        retrieved_chunks = []

        for chunk in all_results:
            key = chunk["text"]  # use text as unique identifier
            
            if key not in seen:
                seen.add(key)
                retrieved_chunks.append(chunk)

        # Step 4: Re-rank (precision step)
        reranked_chunks = reranker.rerank(query, retrieved_chunks, top_k=5)

        print("\nRetrieved Context:\n")
        for i, chunk in enumerate(reranked_chunks):
            print(f"--- Chunk {i+1} ---")
            print(f"Source: {chunk['source']}")
            print(f"Text: {chunk['text']}")
            print()

        # Step 5: Generate answer (use ORIGINAL query)
        answer = generator.generate(query, reranked_chunks)

        print("\nFinal Answer:\n")
        print(answer)