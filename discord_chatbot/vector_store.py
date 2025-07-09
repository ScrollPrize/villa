from langchain.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain.docstore.document import Document
from typing import List, Dict, Any
import numpy as np
from embedder import BaseEmbedder
import json
import os


class CustomEmbeddings(Embeddings):
    def __init__(self, embedder: BaseEmbedder):
        self.embedder = embedder
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Create dummy documents for embedding
        docs = [{"page_content": text} for text in texts]
        embedded_docs = self.embedder.embed_documents(docs)
        return [doc["embedding"] for doc in embedded_docs]
    
    def embed_query(self, text: str) -> List[float]:
        embedding = self.embedder.embed_query(text)
        return embedding.tolist()


class DiscordVectorStore:
    def __init__(self, embedder: BaseEmbedder):
        self.embedder = embedder
        self.custom_embeddings = CustomEmbeddings(embedder)
        self.vector_store = None
        
    def create_vector_store(self, documents: List[Dict[str, Any]]) -> FAISS:
        # Convert to LangChain Document objects
        langchain_docs = []
        for doc in documents:
            langchain_doc = Document(
                page_content=doc["page_content"],
                metadata=doc["metadata"]
            )
            langchain_docs.append(langchain_doc)
        
        # Create FAISS vector store
        self.vector_store = FAISS.from_documents(
            documents=langchain_docs,
            embedding=self.custom_embeddings
        )
        
        return self.vector_store
    
    def save_vector_store(self, path: str = "./discord_vector_store"):
        if self.vector_store:
            self.vector_store.save_local(path)
            # Save metadata about the embedder configuration
            metadata = {
                "embedder_type": self.embedder.__class__.__name__.replace("Embedder", "").lower(),
                "embedder_model": getattr(self.embedder, 'model_name', 'unknown')
            }
            metadata_path = os.path.join(path, "embedder_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
    def load_vector_store(self, path: str = "./discord_vector_store"):
        self.vector_store = FAISS.load_local(
            path, 
            self.custom_embeddings,
            allow_dangerous_deserialization=True
        )
        return self.vector_store
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        
        return self.vector_store.similarity_search(query, k=k)
    
    def similarity_search_with_score(self, query: str, k: int = 5) -> List[tuple]:
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        
        return self.vector_store.similarity_search_with_score(query, k=k)