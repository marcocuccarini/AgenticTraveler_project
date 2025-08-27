"""
RAG system with Ollama + SBERT (sentence-transformers) + FAISS

Notes:
- You mentioned "sbertucciare"; web search shows that's an Italian verb. I assume you meant SBERT / sentence-transformers embeddings. If you really meant something else, tell me and I can adjust.

This single-file example provides:
1. Building embeddings for a list of passages (or documents) using sentence-transformers
2. Indexing embeddings with FAISS (flat index)
3. Querying top-k similar passages
4. Optional cross-encoder reranking
5. Passing retrieved context to Ollama (local) to produce a grounded answer

Requirements:
pip install sentence-transformers faiss-cpu requests numpy

Run a local Ollama instance (default host/port used below):
- Ollama typically listens at http://localhost:11434
- The example uses the /api/generate endpoint; depending on Ollama version you may use the OpenAI-compatible /v1/completions or /v1/chat/completions endpoint. See comments below.

"""

from typing import List, Tuple, Optional
import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer, CrossEncoder
import subprocess
import time
import json


class RAG:
    def __init__(self,
                 embed_model_name: str = "all-MiniLM-L6-v2",
                 use_cross_encoder: bool = False,
                 cross_encoder_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 ollama_url: str = "http://localhost:11434/api/generate",
                 device: str = "cpu"):
        """Initialize RAG helper.

        embed_model_name: sentence-transformers model for embeddings
        use_cross_encoder: if True, will use cross-encoder to rerank top candidates
        ollama_url: URL to Ollama generation endpoint
        """
        self.embed_model = SentenceTransformer(embed_model_name)
        self.ollama_url = ollama_url
        self.embeddings = None
        self.passages = []
        self.index = None
        self.dim = self.embed_model.get_sentence_embedding_dimension()

        self.use_cross = use_cross_encoder
        if use_cross_encoder:
            self.cross = CrossEncoder(cross_encoder_name)
        else:
            self.cross = None

    def build_index(self, passages: List[str]):
        """Compute embeddings and build a FAISS index (IndexFlatIP with normalized vectors).
        We store L2-normalized embeddings so dot-product ~ cosine similarity.
        """
        self.passages = passages
        print(f"Encoding {len(passages)} passages using {self.embed_model.__class__.__name__}...")
        embs = self.embed_model.encode(passages, convert_to_numpy=True, show_progress_bar=True)
        # normalize
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms==0] = 1e-9
        embs = embs / norms
        self.embeddings = embs.astype('float32')

        # build FAISS index for inner product (cosine since normalized)
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(self.embeddings)
        print(f"FAISS index built. Total vectors: {self.index.ntotal}")

    def query(self, query: str, top_k: int = 5, rerank_top: Optional[int] = 20) -> List[Tuple[float, str]]:
        """Return top_k (score, passage) pairs for the query.

        If cross-encoder reranking is enabled: retrieve `rerank_top` with FAISS, then rerank with cross-encoder.
        """
        q_emb = self.embed_model.encode([query], convert_to_numpy=True)
        q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-9)
        q_emb = q_emb.astype('float32')

        nprobe = min(self.index.ntotal, rerank_top or top_k)
        scores, ids = self.index.search(q_emb, nprobe)
        scores = scores[0]
        ids = ids[0]

        candidates = []
        for s, i in zip(scores, ids):
            if i < 0:
                continue
            candidates.append((float(s), self.passages[i], int(i)))

        if self.use_cross and self.cross is not None:
            # prepare pairs for cross encoder: (query, passage)
            pairs = [[query, p] for (_, p, _) in candidates]
            rerank_scores = self.cross.predict(pairs)
            # combine with candidate indices and sort by rerank score
            reranked = sorted(zip(rerank_scores, candidates), key=lambda x: x[0], reverse=True)
            top = [(float(r[0]), r[1][1]) for r in reranked[:top_k]]
            return top
        else:
            # already sorted by FAISS score
            top = [(s, p) for (s, p, _) in candidates[:top_k]]
            return top

    @staticmethod
    def ensure_ollama_model(model: str, ollama_host: str = "http://localhost:11434"):
        """
        Check if the Ollama model is present. If not, download (pull) it.
        """
        try:
            # List models via Ollama API
            resp = requests.get(f"{ollama_host}/api/tags", timeout=10)
            resp.raise_for_status()
            tags = resp.json().get("models", [])
            model_names = [m.get("name", "") for m in tags]
            if model in model_names or any(model.split(":")[0] in m for m in model_names):
                print(f"Ollama model '{model}' is already present.")
                return
            print(f"Ollama model '{model}' not found. Pulling...")
        except Exception as e:
            print(f"Warning: Could not verify Ollama model presence ({e}), attempting to pull anyway.")

        # Pull the model using subprocess (requires ollama CLI)
        try:
            result = subprocess.run(
                ["ollama", "pull", model],
                check=True,
                capture_output=True,
                text=True
            )
            print(result.stdout)
        except Exception as e:
            raise RuntimeError(f"Failed to pull Ollama model '{model}': {e}")

        # Wait for the model to be available (optional, simple wait)
        for _ in range(10):
            try:
                resp = requests.get(f"{ollama_host}/api/tags", timeout=10)
                resp.raise_for_status()
                tags = resp.json().get("models", [])
                model_names = [m.get("name", "") for m in tags]
                if model in model_names or any(model.split(":")[0] in m for m in model_names):
                    print(f"Ollama model '{model}' is now available.")
                    return
            except Exception:
                pass
            time.sleep(2)
        print(f"Warning: Model '{model}' may not be available yet.")

    def generate_with_ollama(self, system_prompt: str, user_query: str, context_passages: List[str],
                             model: str = "llama2", stream: bool = False, options: dict = None) -> str:
        """Call Ollama's API to generate an answer. This uses the /api/generate endpoint format.

        If your Ollama exposes a different endpoint (OpenAI-compatible /v1/chat/completions), adapt accordingly.
        """
        # Ensure the model is present before generating
        self.ensure_ollama_model(model, ollama_host=self.ollama_url.rsplit("/", 1)[0])

        # Build prompt: you can craft templates to include the context
        context_text = "\n\n".join([f"Passage {i+1}: {p}" for i, p in enumerate(context_passages)])
        prompt = f"{system_prompt}\n\nContext:\n{context_text}\n\nUser: {user_query}\nAssistant:"

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream
        }
        if options:
            payload["options"] = options

        resp = requests.post(self.ollama_url, json=payload, timeout=120)
        try:
            resp.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"Ollama request failed: {e} - response text: {resp.text}")

        # Ollama API returns JSON or a streaming chunked response depending on settings.
        data = resp.json()
        # The structure may vary by Ollama version. A common key for the generated text is 'text' or 'choices'.
        # Attempt to extract safely.
        if isinstance(data, dict):
            # example: { 'text': '...' }
            if 'text' in data:
                return data['text']
            if 'choices' in data and len(data['choices'])>0:
                # OpenAI-compatible style
                return data['choices'][0].get('message', {}).get('content') or data['choices'][0].get('text')
        # fallback
        return str(data)

    @staticmethod
    def split_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
        """
        Split a long text into chunks of approximately `chunk_size` characters with `overlap`.
        Returns a list of text chunks.
        """
        chunks = []
        start = 0
        text_length = len(text)
        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunk = text[start:end]
            chunks.append(chunk)
            if end == text_length:
                break
            start += chunk_size - overlap
        return chunks


if __name__ == '__main__':
    # Example usage: consider one image description (monument/historical site) as input, split into chunks of configurable size
    image_description = (
        "The image shows the Colosseum in Rome, Italy, under a clear blue sky. "
        "The ancient amphitheater, built of stone and concrete, stands majestically with its iconic arches and partially ruined outer walls. "
        "Tourists are gathered around the base, some taking photographs, while others listen to guides explaining the history of the site. "
        "Green grass and a few scattered trees surround the monument, and the sunlight casts dramatic shadows on the structure. "
        "In the background, parts of the Roman Forum are visible, hinting at the city's rich historical past. "
        "Vendors selling souvenirs can be seen near the entrance, and a group of school children is sketching the Colosseum from a distance. "
        "The overall atmosphere is vibrant, blending the grandeur of ancient architecture with the lively presence of modern visitors."
    )

    # Set chunk size as a parameter
    chunk_size = 80  # You can change this value as needed
    overlap = 15     # You can also make overlap a parameter if desired

    # Split the image description into passages/chunks using the parameter
    rag = RAG(embed_model_name="all-MiniLM-L6-v2", use_cross_encoder=False,
              ollama_url="http://localhost:11434/api/generate")
    passages = rag.split_text(image_description, chunk_size=chunk_size, overlap=overlap)
    rag.build_index(passages)

    # Questions about the monument/historical site image
    test_questions = [
        "What monument is depicted in the image?",
        "Where is this historical site located?",
        "What materials is the monument made of?",
        "What activities are the tourists engaged in?",
        "Describe the weather and lighting conditions in the image.",
        "Are there any signs of the monument's age or damage?",
        "What can be seen in the background of the image?",
        "Are there any vendors or commercial activity near the site?",
        "How does the image convey the atmosphere around the monument?",
        "What are the visitors doing near the monument?"
    ]

    system_prompt = "You are a helpful assistant. Use the provided context to answer the user's question about the monument or historical site shown in the image and cite passage numbers when helpful."

    for idx, q in enumerate(test_questions[:min(10, len(passages))]):
        print(f"\n--- Test Sample {idx+1} ---")
        top = rag.query(q, top_k=3)
        print("Top retrieved:")
        for s, p in top:
            print(f"score={s:.4f}\t{p}")
        context = [p for (_, p) in top]
        try:
            answer = rag.generate_with_ollama(
                system_prompt=system_prompt,
                user_query=q,
                context_passages=context,
                model="llama3.1"
            )
            # Convert answer to dictionary if it's a string
            if isinstance(answer, str):
                try:
                    answer_dict = json.loads(answer)
                except Exception:
                    answer_dict = {"response": answer}
            elif isinstance(answer, dict):
                answer_dict = answer
            else:
                answer_dict = {"response": str(answer)}
            # Print only the response field if present
            print(answer_dict.get('response', answer_dict))
        except Exception as e:
            print(f"Error calling Ollama for question {idx+1}: {q}\n{e}")
            print("If your Ollama uses a different endpoint (for example /v1/chat/completions), adapt `ollama_url` and payload accordingly.")
            answer = rag.generate_with_ollama(system_prompt=system_prompt, user_query=q, context_passages=context, model="llama3.1")
            print("\nOllama answer:\n", answer)
        except Exception as e:
            print("Error calling Ollama:", e)
            print("If your Ollama uses a different endpoint (for example /v1/chat/completions), adapt `ollama_url` and payload accordingly.")
            print("Error calling Ollama:", e)
            print("If your Ollama uses a different endpoint (for example /v1/chat/completions), adapt `ollama_url` and payload accordingly.")
            print("If your Ollama uses a different endpoint (for example /v1/chat/completions), adapt `ollama_url` and payload accordingly.")
            print("Error calling Ollama:", e)
            print("If your Ollama uses a different endpoint (for example /v1/chat/completions), adapt `ollama_url` and payload accordingly.")
