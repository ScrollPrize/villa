from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import numpy as np
import os
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class BaseEmbedder:
    def embed_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        raise NotImplementedError
    
    def embed_query(self, query: str) -> np.ndarray:
        raise NotImplementedError


class QwenEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-4B"):
        self.model = SentenceTransformer(model_name)
        
    def embed_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        texts = [doc["page_content"] for doc in documents]
        
        # Batch encode all documents
        embeddings = self.model.encode(texts, batch_size=32, show_progress_bar=True)
        
        # Add embeddings to documents
        for i, doc in enumerate(documents):
            doc["embedding"] = embeddings[i].tolist()
            
        return documents
    
    def embed_query(self, query: str) -> np.ndarray:
        # Use the query prompt for better retrieval performance
        return self.model.encode(query, prompt_name="query")


class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = "text-embedding-3-large", batch_size: int = 100):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
        
        self.model_name = model_name
        self.batch_size = batch_size  # OpenAI allows up to 2048 inputs per request
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables. Please add it to your .env file.")
        
        self.client = OpenAI(api_key=api_key)
        
    def _get_embedding(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            model=self.model_name,
            input=text
        )
        return response.data[0].embedding
    
    def _get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a batch of texts."""
        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts
        )
        # Sort by index to ensure correct order
        embeddings = sorted(response.data, key=lambda x: x.index)
        return [emb.embedding for emb in embeddings]
    
    def embed_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        texts = [doc["page_content"] for doc in documents]
        total_docs = len(documents)
        
        print(f"Embedding {total_docs} documents with OpenAI {self.model_name}...")
        print(f"Using batch size: {self.batch_size}")
        
        # Track timing for progress estimation
        start_time = time.time()
        all_embeddings = []
        
        # Format time nicely
        def format_time(seconds):
            if seconds < 60:
                return f"{seconds:.0f}s"
            elif seconds < 3600:
                return f"{seconds/60:.1f}m"
            else:
                return f"{seconds/3600:.1f}h"
        
        # Process in batches
        for batch_start in range(0, total_docs, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total_docs)
            batch_texts = texts[batch_start:batch_end]
            
            try:
                # Get embeddings for the batch
                batch_embeddings = self._get_embeddings_batch(batch_texts)
                all_embeddings.extend(batch_embeddings)
                
                # Update progress
                processed_docs = len(all_embeddings)
                elapsed_time = time.time() - start_time
                avg_time_per_doc = elapsed_time / processed_docs if processed_docs > 0 else 0
                remaining_docs = total_docs - processed_docs
                estimated_remaining = remaining_docs * avg_time_per_doc
                
                progress_pct = (processed_docs / total_docs) * 100
                print(f"Progress: {processed_docs}/{total_docs} ({progress_pct:.1f}%) | "
                      f"Elapsed: {format_time(elapsed_time)} | "
                      f"Remaining: ~{format_time(estimated_remaining)} | "
                      f"Speed: {processed_docs/elapsed_time:.1f} docs/s")
                
            except Exception as e:
                print(f"Error in batch {batch_start}-{batch_end}: {str(e)}")
                # Fall back to individual processing for this batch
                print("Falling back to individual processing for this batch...")
                for text in batch_texts:
                    try:
                        embedding = self._get_embedding(text)
                        all_embeddings.append(embedding)
                    except Exception as e2:
                        print(f"Error embedding document: {str(e2)}")
                        # Use zero vector as fallback
                        all_embeddings.append([0.0] * 1536)  # Default dimension for text-embedding-3-small
        
        # Add embeddings to documents
        for i, doc in enumerate(documents):
            doc["embedding"] = all_embeddings[i]
        
        total_time = time.time() - start_time
        print(f"\nCompleted embedding {total_docs} documents in {format_time(total_time)}")
        print(f"Average speed: {total_docs/total_time:.1f} docs/s")
            
        return documents
    
    def embed_query(self, query: str) -> np.ndarray:
        embedding = self._get_embedding(query)
        return np.array(embedding)


def create_embedder(embedder_type: str = "qwen", **kwargs) -> BaseEmbedder:
    if embedder_type.lower() == "qwen":
        return QwenEmbedder(**kwargs)
    elif embedder_type.lower() == "openai":
        return OpenAIEmbedder(**kwargs)
    else:
        raise ValueError(f"Unknown embedder type: {embedder_type}. Choose 'qwen' or 'openai'.")


# For backward compatibility
DiscordEmbedder = QwenEmbedder