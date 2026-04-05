# Basic RAG (PDF Question Answering)

A Retrieval-Augmented Generation (RAG) project that answers user questions from local PDF documents using semantic search, query expansion, reranking, and LLM-based generation.

## What This Project Does

- Loads PDF documents from `data/docs/`
- Cleans and chunks text into semantic chunks
- Creates and caches embeddings
- Builds a FAISS vector index for fast retrieval
- Expands user queries to improve recall
- Reranks retrieved chunks for precision
- Generates final answers grounded in retrieved context with source attribution


## Pipeline
![rag_pipeline](https://github.com/user-attachments/assets/3b022a26-b6bf-433d-8d69-ec7bf7596990)

## Tech Stack

- Python
- OpenAI API (`gpt-4.1-mini`) for:
	- query expansion
	- answer generation
- Sentence Transformers:
	- `all-miniLM-L6-v2` for embeddings
	- `cross-encoder/ms-marco-MiniLM-L-6-v2` for reranking
- FAISS for vector similarity search
- PyMuPDF (`fitz`) for PDF text extraction

## Repository Structure

```text
Basic_RAG/
	app/
		ingestion.py         # PDF loading, cleaning, chunking
		embedding.py         # Embedding model + embedding cache I/O
		retrieval.py         # FAISS index build/search + index cache I/O
		query_expansion.py   # LLM-based query rewrites
		reranker.py          # Cross-encoder relevance reranking
		generation.py        # LLM answer generation from retrieved context
	data/
		docs/                # Source PDFs for indexing
	storage/               # Generated artifacts: embeddings.npy, index.faiss
	main.py                # Interactive CLI entrypoint
	requirements.txt
```

## End-to-End Pipeline

1. Ingestion
- `app/ingestion.py` loads PDFs from `data/docs/*.pdf`
- Text is normalized and cleaned
- Text is split into paragraphs (with sentence-level fallback)
- Long paragraphs are semantically chunked with overlap
- Output format per chunk:
	- `{"text": <chunk_text>, "source": <file_stem>}`

2. Embedding + Caching
- `app/embedding.py` embeds all chunk texts
- Embeddings are cached to `storage/embeddings.npy`

3. Vector Index + Caching
- `app/retrieval.py` creates a FAISS `IndexFlatL2` index from embeddings
- Index is cached to `storage/index.faiss`

4. Query-Time Retrieval Flow
- User asks a question in `main.py`
- Query expander generates 3 rephrased queries (default)
- Original query is added to that set
- For each query variant, top-5 chunks are retrieved from FAISS
- Retrieved chunks are deduplicated by chunk text
- Top-5 chunks are reranked by cross-encoder
- Note, there are two ways I have shown for retrieval. One is using OpenAI for which an API Key is needed. The second, is using Ollama and using a locally downloaded model.

5. Generation
- Final top reranked chunks are provided to the LLM
- Prompt asks model to answer only from retrieved context and cite source names

## Current Retrieval/Generation Counts

With current defaults in code:

- Query expansion target: 3 rewrites
- Effective query count: original + rewrites
- Retrieval per query variant: top-5
- Reranked chunks passed to generation: top-5 (or fewer if deduped set is smaller)

## How to Clone and Run

### 1. Clone

```bash
git clone <your-repo-url>
cd Basic_RAG
```

### 2. Create and Activate a Virtual Environment

Windows (PowerShell):

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 5. Add Documents

Place your PDF files inside:

```text
data/docs/
```

### 6. Run the App

```bash
python main.py
```

Type questions in the terminal. Type `exit` to quit.

## Example Session

```text
Ask a question (or type 'exit'): What is the core idea in Attention Is All You Need?

Expanded Queries:
- What is the core idea in Attention Is All You Need?
- ...

Retrieved Context:
--- Chunk 1 ---
Source: Attention_Is_All_You_Need
Text: ...

Final Answer:
...
```


## Important Operational Notes

- If you change documents in `data/docs/`, clear cached artifacts to reindex fresh content:
	- `storage/embeddings.npy`
	- `storage/index.faiss`
- The current ingestion path loads PDFs (`*.pdf`) only.


## License

Add your preferred license here (for example: MIT).
