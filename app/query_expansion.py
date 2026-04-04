# Two ways to run this code, uncommented version uses OpenAI's API, while the commented version uses Ollama for local inference with Gemma4. 

from openai import OpenAI
import os
from dotenv import load_dotenv
import re

load_dotenv(dotenv_path=".env")


class QueryExpander:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def expand(self, query, n=3):
        prompt = f"""
You are helping improve search queries for a retrieval system.

Given a user query, generate {n} different rephrased versions that capture the same intent.
Keep them concise and semantically similar.

Original Query:
{query}

Expanded Queries:
"""

        response = self.client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        text = response.choices[0].message.content

        # Split into lines and clean
        queries = []
        for q in text.split("\n"):
            q = q.strip()
            q = re.sub(r"^\d+\.\s*", "", q)  # remove "1. ", "2. "
            q = q.lstrip("- ").strip()
            if q:
                queries.append(q)

        return queries

# import ollama  # Replaces OpenAI
# import re

# class QueryExpander:
#     def __init__(self):
#         # No API key or .env loading needed for the local model
#         self.model = "gemma4:e2b"

#     def expand(self, query, n=3):
#         prompt = f"""
# You are helping improve search queries for a retrieval system.

# Given a user query, generate {n} different rephrased versions that capture the same intent.
# Keep them concise and semantically similar.

# Original Query:
# {query}

# Expanded Queries:
# """

#         # Call the local Gemma 4 model
#         response = ollama.chat(
#             model=self.model,
#             messages=[{"role": "user", "content": prompt}],
#             options={
#                 "temperature": 0.4  # Slightly higher for variety in rephrasing
#             }
#         )

#         # Access the text from the Ollama response object
#         text = response['message']['content']

#         # Split into lines and clean (Your original logic)
#         queries = []
#         for q in text.split("\n"):
#             q = q.strip()
#             q = re.sub(r"^\d+\.\s*", "", q)  # remove "1. ", "2. "
#             q = q.lstrip("- ").strip()
#             if q:
#                 queries.append(q)

#         # Ensure we only return the number of queries requested (n)
#         return queries[:n]