from pathlib import Path
import fitz  # PyMuPDF
import re


def load_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""

    for page in doc:
        text += page.get_text()  # Storing all the text in a single string for now

    return text


def load_documents(folder_path):
    docs = []

    for file in Path(folder_path).glob("*.pdf"):
        text = load_pdf(file)
        docs.append({
            "text": text,
            "source": file.stem  # Storing metadata to help LLM cite sources later
        })

    return docs


def clean_text(text):
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    # Fix broken hyphenation (common in PDFs)
    text = re.sub(r'-\s+', '', text)

    return text.strip()


def split_into_paragraphs(text):
    # Try splitting by paragraph breaks
    paragraphs = re.split(r'\n{2,}|\r\n{2,}', text)

    # If PDF destroyed structure → fallback to sentence grouping
    if len(paragraphs) < 5:
        paragraphs = re.split(r'(?<=\.) ', text)

    return [p.strip() for p in paragraphs if len(p.strip()) > 50]


def smart_chunk(paragraph, chunk_size=400, overlap=80):
    sentences = re.split(r'(?<=[.!?]) +', paragraph)

    chunks = []
    current = ""

    for s in sentences:
        if len(current) + len(s) <= chunk_size:
            current += " " + s
        else:
            chunks.append(current.strip())
            current = s

    if current:
        chunks.append(current.strip())

    # Add overlap
    final_chunks = []
    for i, c in enumerate(chunks):
        if i > 0:
            prev = chunks[i - 1]
            c = prev[-overlap:] + " " + c
        final_chunks.append(c.strip())

    return final_chunks


def chunk_text(text):
    text = clean_text(text)

    paragraphs = split_into_paragraphs(text)

    all_chunks = []

    for para in paragraphs:
        if len(para) < 200:
            all_chunks.append(para)
        else:
            all_chunks.extend(smart_chunk(para))

    return all_chunks


def get_chunks(folder_path):
    documents = load_documents(folder_path)

    all_chunks = []

    for doc in documents:
        chunks = chunk_text(doc["text"])

        for chunk in chunks:
            all_chunks.append({
                "text": chunk,
                "source": doc["source"]
            })

    return all_chunks