# Two ways to run this code, uncommented version uses OpenAI's API, while the commented version uses Ollama for local inference with Gemma4. Both are designed to work with the same prompt structure and context formatting.

from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")


class Generator:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate(self, query, context_chunks):
        context = "\n\n".join(
                [f"[Source: {chunk['source']}]\n{chunk['text']}" for chunk in context_chunks]
            ) 

        prompt = f"""
You are a helpful assistant.

Use the provided context to answer the question.

Instructions:
- Carefully analyze the context
- Combine information if needed
- Do NOT assume anything not present
- Cite only [Source: name]
- If unsure, say "I don't know"

Context:
{context}

Question:
{query}

Answer:
"""
        response = self.client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content


# import ollama  # Replaces OpenAI

# class Generator:
#     def __init__(self):
#         # No API key needed! Ollama connects to the local service automatically.
#         # We ensure the model is pulled; if not, you can run 'ollama pull gemma4:e2b'
#         self.model = "gemma4:e2b"

#     def generate(self, query, context_chunks):
#         # Keep your existing context formatting logic
#         context = "\n\n".join(
#             [f"[Source: {chunk['source']}]\n{chunk['text']}" for chunk in context_chunks]
#         ) 

#         prompt = f"""
# You are a helpful assistant.

# Use the provided context to answer the question.

# Instructions:
# - Carefully analyze the context
# - Combine information if needed
# - Do NOT assume anything not present
# - Cite only [Source: name]
# - If unsure, say "I don't know"

# Context:
# {context}

# Question:
# {query}

# Answer:
# """
#         # Use the local Ollama chat method
#         response = ollama.chat(
#             model=self.model,
#             messages=[
#                 {"role": "user", "content": prompt}
#             ],
#             options={
#                 "temperature": 0.2,  # Low temperature is better for RAG/Fact-checking
#                 "num_ctx": 8192      # Gives the model more "memory" for your context
#             }
#         )

#         return response['message']['content']