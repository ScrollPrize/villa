import os
import argparse
import glob
from typing import List, Dict, Any
from discord_parser import DiscordChatParser
from embedder import create_embedder
from vector_store import DiscordVectorStore
from document_filter import filter_criteria


def parse_json_files(input_path: str) -> List[Dict[str, Any]]:
    """Parse JSON files from either a single file or directory."""
    all_documents = []
    
    if os.path.isfile(input_path):
        # Single file
        json_files = [input_path]
    elif os.path.isdir(input_path):
        # Directory - find all JSON files
        json_files = glob.glob(os.path.join(input_path, "*.json"))
        json_files.sort()  # Sort for consistent processing order
    else:
        raise ValueError(f"Input path {input_path} is neither a file nor a directory")
    
    if not json_files:
        raise ValueError(f"No JSON files found in {input_path}")
    
    print(f"Found {len(json_files)} JSON file(s) to process")
    
    # Parse each JSON file
    for json_file in json_files:
        try:
            print(f"\nProcessing: {os.path.basename(json_file)}")
            parser = DiscordChatParser(json_file)
            documents = parser.get_documents()
            print(f"  Parsed {len(documents)} messages")
            all_documents.extend(documents)
        except Exception as e:
            print(f"  Error processing {json_file}: {str(e)}")
            print("  Skipping this file...")
            continue
    
    return all_documents


def build_discord_rag(input_path: str, output_path: str = "./discord_vector_store", 
                      embedder_type: str = "qwen", embedder_model: str = None):
    print(f"Loading Discord chat exports from: {input_path}")
    
    # Parse all JSON files
    documents = parse_json_files(input_path)
    
    if not documents:
        raise ValueError("No documents were successfully parsed from the input files")
    
    print(f"\nTotal messages parsed: {len(documents)}")
    
    # Apply filtering
    documents = filter_criteria(documents)
    
    # Initialize embedder based on type
    print(f"\nInitializing {embedder_type} embedding model...")
    if embedder_type == "qwen":
        model_name = embedder_model or "Qwen/Qwen3-Embedding-0.6B"
        embedder = create_embedder("qwen", model_name=model_name)
    elif embedder_type == "openai":
        model_name = embedder_model or "text-embedding-3-small"
        embedder = create_embedder("openai", model_name=model_name)
    else:
        raise ValueError(f"Unknown embedder type: {embedder_type}")
    
    # Create vector store
    print("\nCreating vector store...")
    vector_store = DiscordVectorStore(embedder)
    vector_store.create_vector_store(documents)
    
    # Save vector store
    print(f"\nSaving vector store to: {output_path}")
    vector_store.save_vector_store(output_path)
    
    print("\nRAG system built successfully!")
    print(f"Processed {len(documents)} messages from {input_path}")
    return vector_store


def test_retrieval(vector_store_path: str = "./discord_vector_store",
                   embedder_type: str = "qwen", embedder_model: str = None):
    print("Loading vector store for testing...")
    
    # Initialize embedder based on type
    if embedder_type == "qwen":
        model_name = embedder_model or "Qwen/Qwen3-Embedding-8B"
        embedder = create_embedder("qwen", model_name=model_name)
    elif embedder_type == "openai":
        model_name = embedder_model or "text-embedding-3-small"
        embedder = create_embedder("openai", model_name=model_name)
    else:
        raise ValueError(f"Unknown embedder type: {embedder_type}")
    
    vector_store = DiscordVectorStore(embedder)
    vector_store.load_vector_store(vector_store_path)
    

    # Test queries
    test_queries = [
        "What is the Vesuvius Challenge?",
        "How can I get help?",
        "What happened in the grand prize?",
        'Who won the vesuvius challenge grand prize?',
        'Hi I am new contestant to the challenge, I don\'t know how to get started, how can I get help?'
    ]
    
    print("\nTesting retrieval with sample queries:")
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = vector_store.similarity_search_with_score(query, k=3)
        
        for i, (doc, score) in enumerate(results):
            print(f"\nResult {i+1} (Score: {score:.4f}):")
            print(f"Content: {doc.page_content[:500]}...")
            print(f"Author: {doc.metadata.get('author')}")
            print(f"Channel: {doc.metadata.get('channel')}")


def main():
    parser = argparse.ArgumentParser(description="Build Discord RAG system from JSON exports")
    parser.add_argument("--input", type=str, default="sample_text.json",
                       help="Path to Discord JSON export file or directory containing JSON files")
    parser.add_argument("--output", type=str, default="./discord_vector_store",
                       help="Path to save vector store")
    parser.add_argument("--embedder", type=str, default="qwen",
                       choices=["qwen", "openai"],
                       help="Embedder type to use (qwen or openai)")
    parser.add_argument("--embedder-model", type=str, default=None,
                       help="Specific embedding model to use (e.g., text-embedding-3-large for OpenAI)")
    parser.add_argument("--test", action="store_true",
                       help="Run test queries after building")
    
    args = parser.parse_args()
    
    # Build RAG system
    vector_store = build_discord_rag(args.input, args.output, 
                                     args.embedder, args.embedder_model)
    
    # Optionally test retrieval
    if args.test:
        test_retrieval(args.output, args.embedder, args.embedder_model)


if __name__ == "__main__":
    main()