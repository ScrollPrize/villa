#!/usr/bin/env python3
import os
from rag_chatbot import DiscordRAGChatbot


def test_rag_pipeline():
    print("Testing RAG Pipeline...")
    print("=" * 50)
    
    # Check if vector store exists
    vector_store_path = "./discord_vector_store"
    if not os.path.exists(vector_store_path):
        print(f"Error: Vector store not found at {vector_store_path}")
        print("Please run: python build_rag.py --input sample_text.json")
        return
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key: export OPENAI_API_KEY='your-key-here'")
        return
    
    try:
        # Initialize chatbot
        print("\n1. Initializing chatbot...")
        chatbot = DiscordRAGChatbot(vector_store_path=vector_store_path)
        print("✓ Chatbot initialized successfully")
        
        # Test queries
        test_queries = [
            "What is the Vesuvius Challenge?",
            "How can I get help with the challenge?",
            "Who won the grand prize?"
        ]
        
        print("\n2. Testing queries...")
        for i, query in enumerate(test_queries, 1):
            print(f"\nQuery {i}: {query}")
            print("-" * 40)
            
            result = chatbot.query(query, k=3)
            
            print(f"Answer: {result['answer'][:200]}...")
            print(f"Sources retrieved: {len(result['sources'])}")
            print(f"Top source score: {result['sources'][0]['score']:.4f}")
            print(f"Tokens used: {result['usage']['total_tokens']}")
        
        print("\n✓ All tests passed!")
        print("\nYou can now use the chatbot with:")
        print("  - Interactive mode: python chat.py")
        print("  - Single query: python chat.py --query 'Your question here'")
        
    except Exception as e:
        print(f"\n✗ Error during testing: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Make sure you've built the vector store: python build_rag.py --input sample_text.json")
        print("2. Set your OpenAI API key: export OPENAI_API_KEY='your-key-here'")
        print("3. Check that all dependencies are installed")


if __name__ == "__main__":
    test_rag_pipeline()