#!/usr/bin/env python3
import os
import argparse
from rag_chatbot import DiscordRAGChatbot


def main():
    parser = argparse.ArgumentParser(description="Discord RAG Chatbot - Ask questions about Discord chat history")
    parser.add_argument("--vector-store", type=str, default="./discord_vector_store",
                       help="Path to the vector store directory")
    parser.add_argument("--api-key", type=str, default=None,
                       help="OpenAI API key (defaults to OPENAI_API_KEY env var)")
    parser.add_argument("--k", type=int, default=5,
                       help="Number of relevant documents to retrieve")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Temperature for GPT-4 generation (0-1)")
    parser.add_argument("--max-tokens", type=int, default=500,
                       help="Maximum tokens for response generation")
    parser.add_argument("--query", type=str, default=None,
                       help="Single query to run (non-interactive mode)")
    
    args = parser.parse_args()
    
    try:
        # Initialize chatbot
        print("Initializing Discord RAG Chatbot...")
        chatbot = DiscordRAGChatbot(
            vector_store_path=args.vector_store,
            api_key=args.api_key
        )
        
        if args.query:
            # Non-interactive mode - single query
            print(f"\nQuery: {args.query}")
            result = chatbot.query(
                args.query, 
                k=args.k,
                temperature=args.temperature,
                max_tokens=args.max_tokens
            )
            
            print(f"\nAnswer: {result['answer']}")
            
            print("\nSources:")
            for i, source in enumerate(result['sources']):
                print(f"\n[Source {i+1}] (Score: {source['score']:.4f})")
                print(f"Author: {source['author']} | Channel: {source['channel']} | Time: {source['timestamp']}")
                print(f"Content: {source['content'][:300]}...")
            
            print(f"\nTokens used: {result['usage']['total_tokens']}")
        else:
            # Interactive mode
            chatbot.interactive_chat()
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())