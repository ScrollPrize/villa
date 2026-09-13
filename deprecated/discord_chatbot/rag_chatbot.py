import os
import json
from typing import List, Dict, Any, Optional
from openai import OpenAI
from vector_store import DiscordVectorStore
from embedder import create_embedder
from langchain.docstore.document import Document


class DiscordRAGChatbot:
    def __init__(self, vector_store_path: str = "./discord_vector_store", 
                 api_key: Optional[str] = None):
        self.vector_store_path = vector_store_path
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.vector_store = None
        self.embedder = None
        
        # Load vector store and embedder configuration
        self._load_vector_store()
    
    def _load_vector_store(self):
        # Load embedder metadata
        metadata_path = os.path.join(self.vector_store_path, "embedder_metadata.json")
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Embedder metadata not found at {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Create embedder based on saved configuration
        embedder_type = metadata.get("embedder_type", "qwen")
        embedder_model = metadata.get("embedder_model", None)
        
        print(f"Loading embedder: {embedder_type} with model {embedder_model}")
        self.embedder = create_embedder(embedder_type, model_name=embedder_model)
        
        # Load vector store
        self.vector_store = DiscordVectorStore(self.embedder)
        self.vector_store.load_vector_store(self.vector_store_path)
        print(f"Vector store loaded from {self.vector_store_path}")
    
    def _format_context(self, documents: List[Document]) -> str:
        context_parts = []
        
        for i, doc in enumerate(documents):
            author = doc.metadata.get('author', 'Unknown')
            channel = doc.metadata.get('channel', 'Unknown')
            timestamp = doc.metadata.get('timestamp', 'Unknown')
            
            context_parts.append(f"[Message {i+1}]")
            context_parts.append(f"Author: {author}")
            context_parts.append(f"Channel: {channel}")
            context_parts.append(f"Time: {timestamp}")
            context_parts.append(f"Content: {doc.page_content}")
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def query(self, question: str, k: int = 6, temperature: float = 0.7, 
              max_tokens: int = 500) -> Dict[str, Any]:
        # Retrieve relevant documents
        results = self.vector_store.similarity_search_with_score(question, k=k)
        
        # Extract documents and scores
        documents = [doc for doc, score in results]
        scores = [score for doc, score in results]
        
        # Format context from retrieved documents
        context = self._format_context(documents)
        
        # Create prompt for GPT-4.0 mini
        system_prompt = """You are a helpful assistant that answers questions based on Discord chat history. 
Use the provided context to answer the user's question. If the answer cannot be found in the context, 
say so clearly. Always cite which messages you're basing your answer on."""
        
        user_prompt = f"""
Short History of the Vesuvius Challenge project:
# Vesuvius Challenge Project History Summary

## Project Overview

The Vesuvius Challenge is a groundbreaking initiative to digitally read ancient scrolls from Herculaneum that were carbonized during the eruption of Mount Vesuvius in 79 CE. The project combines cutting-edge technology including X-ray tomography, machine learning, and virtual unwrapping to recover text from scrolls that cannot be physically opened.

## Key Technological Foundation (Pre-2023)

### Core Technologies and Pioneers

- Dr. Brent Seales (University of Kentucky EduceLab): Lead researcher who pioneered virtual unwrapping techniques

### Institutional Partners

- EduceLab at University of Kentucky: Led scanning efforts and software development
- Officina dei Papiri Ercolanesi (Naples): Custodians of the scrolls, provided access to the ancient materials
- University of Naples Federico II: Academic partner in Italy

## Contest Launch (March 2023)

### Initial Setup

The contest was launched by Nat Friedman with an initial prize pool of $150,000. The project gained immediate traction with massive media coverage and community engagement.

## Major Breakthrough Timeline

### First Letters Discovery (October 2023)

- Youssef Nader: Independent discovery of letters using machine learning approaches

### 2023 Grand Prize Winners (February 2024)

The $700,000 Grand Prize was awarded to a team of three:

- Youssef Nader (PhD student in Berlin)
- Historic Achievement: They successfully read *over 2,000 characters (15 columns) of ancient Greek philosophy - the first time text had ever been recovered non-invasively from a complete Herculaneum scroll.

## Technical Breakthroughs

### Scanning Innovations

- High-resolution X-ray tomography: Scanning at Diamond Light Source with resolutions as fine as 3.24 µm
- Multi-energy scanning: Using different X-ray energy levels to enhance ink detection
- Volume scanning: Creating 3D models of entire scrolls for virtual unwrapping

### Machine Learning Advances

- Ink detection models: ML algorithms trained to identify carbon-based ink invisible to the naked eye
- Surface segmentation: Automated tracing of papyrus surfaces through complex 3D scroll structures
- Virtual unwrapping: Converting 3D scroll data into readable 2D text surfaces

### Software Development

- Volume Cartographer: Core software for virtual unwrapping (Seth Parker)
- Various ML models: Open-source ink detection and segmentation tools

## Prize Distribution ($10,000+ Awards)

### Major Prizes Awarded

- $700,000 Grand Prize (2023): Youssef Nader, Julian Schilliger, Luke Farritor
- $60,000 First Title Prize: For discovering the title "Philodemus, On Vices, Book 1(?)"
- $50,000 First Letters Prizes: Multiple awards for discovering first letters in different scrolls
- $40,000 First Letters Open Source Prize: For finding first letters with open-source methods
- $30,000 Gold Aureus Prize++: Multiple awards for automated segmentation breakthroughs
- $25,000 Open Source Prize: Expanded from $20,000 due to high-quality submissions
- $20,000 Progress Prizes: Monthly awards for significant technical contributions
- $15,000-$10,000 Awards: Numerous progress prizes for tools, segmentation, and ink detection advances

## Content Discoveries

### Scroll Texts Recovered

- Greek philosophical texts: Primarily works by Philodemus of Gadara
- "On Vices, Book 1": First complete title discovered

### Scroll Inventory

- 5 complete scrolls initially scanned: Scrolls 1-5 (PHerc 1, 2, 3, 4, and 172)
- 20 additional scrolls scanned: Major scanning session expanded the dataset
- 300+ scrolls total: Ultimate goal to scan all known Herculaneum scrolls

## Community and Open Source Impact

### Technical Contributions

- Open-source tools: Volume Cartographer, various ML models, segmentation tools
- Collaborative development: Global community of researchers, students, and developers
- Kaggle competitions: Ink detection challenges with hundreds of participants
- Discord community: Over 400 active members sharing techniques and discoveries

## Current Status and Future Goals

### 2024 Objectives

- $200,000 Grand Prize: For reading 90% of four scrolls (doubled from $100,000)
- Automated segmentation: $100,000 prize for reproducing 2023 results faster
- Additional scroll reading: First Letters prizes for Scrolls 2, 3, and 4
- Monthly progress prizes: $350,000 allocated for ongoing technical improvements

### Technical Challenges

- Segmentation scalability: Need for faster, more automated approaches
- Ink detection accuracy: Improving ML models for different scroll conditions
- Complete scroll reading: Scaling from 5% to 90% coverage
- Multi-scroll processing: Handling variations between different papyri

### Long-term Vision

- Complete library recovery: Reading all 300+ Herculaneum scrolls
- Historical impact: Recovering potentially thousands of lost ancient texts
- Technological advancement: Pushing boundaries of digital archaeology
- Open access: Making discoveries freely available to scholars worldwide

## Technical Methodology

### Scanning Process

1. Physical preparation: Careful handling of 2,000-year-old carbonized scrolls
2. X-ray tomography: High-resolution scanning using particle accelerator
3. 3D reconstruction: Creating detailed volumetric models
4. Surface segmentation: Tracing individual papyrus layers
5. Virtual unwrapping: Converting 3D surfaces to readable 2D images
6. Ink detection: Using ML to identify carbon-based ink
7. Text recovery: Combining techniques to produce readable ancient text

### Key Technical Metrics

- Resolution: As fine as 2.2 µm pixel size
- Data size: Terabytes of scan data per scroll
- Accuracy targets: 90% text recovery for prize completion

This project represents one of the most successful intersections of modern technology and ancient scholarship, combining cutting-edge AI with classical studies to recover lost literature from one of history's most famous natural disasters. The open-source, collaborative approach has created a global community of contributors working to unlock the secrets of the ancient world.

Query Context from Discord chat history:
{context}

Question: {question}

Provide a helpful answer based on the context above"""
        
        # Generate response using GPT-4.0 mini
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        answer = response.choices[0].message.content
        
        return {
            "question": question,
            "answer": answer,
            "sources": [
                {
                    "content": doc.page_content,
                    "author": doc.metadata.get('author', 'Unknown'),
                    "channel": doc.metadata.get('channel', 'Unknown'),
                    "channel_id": doc.metadata.get('channel_id', 'Unknown'),
                    "message_id": doc.metadata.get('message_id', 'Unknown'),
                    "guild_id": doc.metadata.get('guild_id', 'Unknown'),
                    "timestamp": doc.metadata.get('timestamp', 'Unknown'),
                    "score": score
                }
                for doc, score in zip(documents, scores)
            ],
            "model": "gpt-4o-mini",
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }
    
    def interactive_chat(self):
        print("Discord RAG Chatbot initialized. Type 'exit' to quit.")
        print("-" * 50)
        
        while True:
            try:
                question = input("\nYour question: ").strip()
                
                if question.lower() in ['exit', 'quit', 'bye']:
                    print("Goodbye!")
                    break
                
                if not question:
                    print("Please enter a question.")
                    continue
                
                print("\nSearching for relevant context...")
                result = self.query(question)
                
                print(f"\nAnswer: {result['answer']}")
                
                # Optionally show sources
                show_sources = input("\nShow sources? (y/n): ").strip().lower()
                if show_sources == 'y':
                    print("\nSources:")
                    for i, source in enumerate(result['sources']):
                        print(f"\n[Source {i+1}] (Score: {source['score']:.4f})")
                        print(f"Author: {source['author']} | Channel: {source['channel']}")
                        print(f"Content: {source['content'][:200]}...")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {str(e)}")