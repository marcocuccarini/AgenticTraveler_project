"""
RAG system with Smolagents + SBERT (sentence-transformers) + FAISS

This implementation replaces Ollama with Smolagents for text generation while maintaining
the same interface as the original RAG system.

Key differences from rag_sytem.py:
- Uses Smolagents CodeAgent instead of Ollama HTTP API
- Implements custom RAG tool for agent integration
- Maintains identical public API for compatibility

Features:
1. Building embeddings for passages using sentence-transformers
2. Indexing embeddings with FAISS (flat index)
3. Querying top-k similar passages
4. Optional cross-encoder reranking
5. Passing retrieved context to Smolagents CodeAgent to produce a grounded answer

Requirements:
pip install sentence-transformers faiss-cpu numpy smolagents

For Smolagents to work properly, you need:
- Internet connection for InferenceClientModel (uses Hugging Face Inference API)
- Or local models with TransformersModel
- HuggingFace token in environment variable HF_TOKEN
"""

from typing import List, Tuple, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
import time
import json
import traceback

# Smolagents imports
from smolagents import CodeAgent, TransformersModel, Tool


class RAGResponseTool(Tool):
    """Custom tool for RAG response generation within Smolagents"""
    
    name = "rag_response_generator"
    description = "Generates responses based on retrieved context passages and user queries for RAG systems"
    inputs = {
        "system_prompt": {
            "type": "string",
            "description": "The system prompt that defines the assistant's role and behavior"
        },
        "user_query": {
            "type": "string", 
            "description": "The user's question or query to be answered"
        },
        "context_passages": {
            "type": "string",
            "description": "Retrieved context passages formatted as a single string, separated by clear delimiters"
        }
    }
    output_type = "string"
    
    def forward(self, system_prompt: str, user_query: str, context_passages: str) -> str:
        """
        Generate a response based on the provided context and query.
        This method formats the input for optimal LLM processing.
        """
        # Format the complete prompt for the LLM
        formatted_prompt = f"""System: {system_prompt}

Context Information:
{context_passages}

User Query: {user_query}

Please provide a comprehensive answer based on the context information provided above. If the context doesn't contain enough information to fully answer the query, acknowledge this limitation."""

        # Return the formatted prompt - the CodeAgent will handle the actual LLM generation
        return formatted_prompt


class RAGSmolagent:
    def __init__(self,
                 embed_model_name: str = "all-MiniLM-L6-v2",
                 use_cross_encoder: bool = False,
                 cross_encoder_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 model_id: str = "Qwen/Qwen2.5-Coder-32B-Instruct",
                 device: str = "cpu"):
        """Initialize RAG system with Smolagents.

        Args:
            embed_model_name: sentence-transformers model for embeddings
            use_cross_encoder: if True, will use cross-encoder to rerank top candidates
            cross_encoder_name: cross-encoder model name for reranking
            model_id: Hugging Face model ID for Smolagents
            device: device for sentence transformers models
        """
        print(f"🤖 Initializing RAG system with Smolagents...")
        print(f"📦 Embedding model: {embed_model_name}")
        print(f"🧠 LLM model: {model_id}")
        
        # Initialize embedding model
        try:
            self.embed_model = SentenceTransformer(embed_model_name, device=device)
            print(f"✅ Embedding model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load embedding model: {e}")
            raise
        
        self.embeddings = None
        self.passages = []
        self.index = None
        self.dim = self.embed_model.get_sentence_embedding_dimension()

        # Initialize cross-encoder if needed
        self.use_cross = use_cross_encoder
        if use_cross_encoder:
            try:
                self.cross = CrossEncoder(cross_encoder_name)
                print(f"✅ Cross-encoder loaded: {cross_encoder_name}")
            except Exception as e:
                print(f"⚠️ Cross-encoder loading failed: {e}, continuing without reranking")
                self.cross = None
                self.use_cross = False
        else:
            self.cross = None

        # Initialize Smolagents
        self.model_id = model_id
        self.agent = None
        self._initialize_smolagent()

    def _initialize_smolagent(self):
        """Initialize the Smolagents CodeAgent with custom RAG tool"""
        try:
            print(f"🔧 Setting up Smolagents CodeAgent...")
            
            # Create the custom RAG tool
            rag_tool = RAGResponseTool()
            
            # Initialize the model
            model = TransformersModel(
                        model_id="/leonardo_work/uTS25_Pinna/phd_proj/TurismAgent/TurismAgent/model/Qwen2.5-Coder-7B-Instruct",
                        trust_remote_code=True,
                        device_map="cuda",
                        # torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        # max_memory={0: "6GB"} if torch.cuda.is_available() else None  # Limita memoria GPU
                    )
            
            # Create the CodeAgent with our custom tool
            self.agent = CodeAgent(
                tools=[rag_tool],
                model=model,
                max_steps=3,
                verbosity_level=0  # Keep it quiet for cleaner output
            )
            print(f"✅ Smolagents CodeAgent initialized successfully")
            
        except Exception as e:
            print(f"❌ Failed to initialize Smolagents: {e}")
            print(f"💡 Make sure you have HF_TOKEN set and internet connection")
            self.agent = None

    def build_index(self, passages: List[str]):
        """Compute embeddings and build a FAISS index (IndexFlatIP with normalized vectors).
        We store L2-normalized embeddings so dot-product ~ cosine similarity.
        """
        self.passages = passages
        print(f"📊 Encoding {len(passages)} passages using {self.embed_model.__class__.__name__}...")
        
        try:
            embs = self.embed_model.encode(passages, convert_to_numpy=True, show_progress_bar=True)
            
            # normalize
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            norms[norms==0] = 1e-9
            embs = embs / norms
            self.embeddings = embs.astype('float32')

            # build FAISS index for inner product (cosine since normalized)
            self.index = faiss.IndexFlatIP(self.dim)
            self.index.add(self.embeddings)
            print(f"🔍 FAISS index built. Total vectors: {self.index.ntotal}")
            
        except Exception as e:
            print(f"❌ Error building index: {e}")
            raise

    def query(self, query: str, top_k: int = 5, rerank_top: Optional[int] = 20) -> List[Tuple[float, str]]:
        """Return top_k (score, passage) pairs for the query.

        If cross-encoder reranking is enabled: retrieve `rerank_top` with FAISS, then rerank with cross-encoder.
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
            
        try:
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
                
        except Exception as e:
            print(f"❌ Error during query: {e}")
            return []

    def generate_with_smolagent(self, system_prompt: str, user_query: str, context_passages: List[str], 
                               **kwargs) -> str:
        """Generate answer using Smolagents CodeAgent with retrieved context.

        Args:
            system_prompt: System instructions for the assistant
            user_query: The user's question
            context_passages: List of retrieved context passages
            **kwargs: Additional parameters (maintained for compatibility, currently unused)

        Returns:
            Generated response string
        """
        if not self.agent:
            return "❌ Smolagents CodeAgent not available. Check initialization errors."

        try:
            # Format context passages for the agent
            formatted_context = "\n\n".join([
                f"Passage {i+1}: {passage}" 
                for i, passage in enumerate(context_passages)
            ])

            # Prepare the task for the agent
            task = f"""Please answer the following question based on the provided context passages.

System Instructions: {system_prompt}

Context:
{formatted_context}

Question: {user_query}

Please provide a comprehensive answer based on the context above. If the context doesn't fully address the question, mention what information is missing."""

            # Use the agent to generate response
            print("🤔 Generating response with Smolagents...")
            response = self.agent.run(task)
            
            return str(response)

        except Exception as e:
            error_msg = f"❌ Smolagents generation failed: {str(e)}"
            print(error_msg)
            return error_msg

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

    def get_system_status(self) -> dict:
        """Get status of all system components for debugging"""
        return {
            "embedding_model": self.embed_model.__class__.__name__ if self.embed_model else None,
            "cross_encoder": self.cross.__class__.__name__ if self.cross else None,
            "smolagents_agent": "Available" if self.agent else "Not Available",
            "model_id": self.model_id,
            "passages_count": len(self.passages),
            "index_built": self.index is not None,
            "embedding_dimension": self.dim
        }


if __name__ == '__main__':
    # Example usage: RAG system with Smolagents instead of Ollama
    print("🚀 Testing RAG system with Smolagents...")
    
    # Sample monument description for testing
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
    chunk_size = 80
    overlap = 15

    try:
        # Initialize RAG system with Smolagents
        rag = RAGSmolagent(
            embed_model_name="all-MiniLM-L6-v2", 
            use_cross_encoder=False,
            model_id="Qwen/Qwen2.5-Coder-32B-Instruct"
        )
        
        # Print system status
        status = rag.get_system_status()
        print(f"📊 System Status: {json.dumps(status, indent=2)}")

        # Split the image description into passages/chunks
        passages = rag.split_text(image_description, chunk_size=chunk_size, overlap=overlap)
        print(f"📚 Created {len(passages)} text chunks")
        
        # Build the index
        rag.build_index(passages)

        # Test questions about the monument/historical site image
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

        system_prompt = ("You are a helpful assistant specializing in monuments and cultural heritage. "
                        "Use the provided context to answer the user's question about the monument or "
                        "historical site and cite passage numbers when helpful.")

        # Test each question
        for idx, question in enumerate(test_questions[:3]):  # Test first 3 questions
            print(f"\n{'='*60}")
            print(f"🔍 Test Question {idx+1}: {question}")
            print(f"{'='*60}")
            
            # Query for relevant passages
            top_results = rag.query(question, top_k=3)
            print(f"📝 Top retrieved passages:")
            for i, (score, passage) in enumerate(top_results, 1):
                print(f"  {i}. Score: {score:.4f} | {passage}")
            
            # Generate answer with Smolagents
            context_passages = [passage for _, passage in top_results]
            try:
                answer = rag.generate_with_smolagent(
                    system_prompt=system_prompt,
                    user_query=question,
                    context_passages=context_passages
                )
                print(f"\n🤖 Generated Answer:\n{answer}")
                
            except Exception as e:
                print(f"❌ Error generating answer: {e}")
            
            print(f"\n{'='*60}")

    except Exception as e:
        print(f"❌ Error initializing or running RAG system: {e}")
        print(f"📋 Traceback: {traceback.format_exc()}")
        print(f"\n💡 Troubleshooting tips:")
        print(f"   - Ensure HF_TOKEN is set in environment variables")
        print(f"   - Check internet connection for Hugging Face Inference API")
        print(f"   - Verify all dependencies are installed: pip install sentence-transformers faiss-cpu smolagents")