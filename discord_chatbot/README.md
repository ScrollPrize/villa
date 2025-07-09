# Discord RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions based on Discord chat history using GPT-4.0 mini.

## Features

- Parse Discord chat exports (JSON format)
- Create vector embeddings using Qwen or OpenAI models
- Store embeddings in FAISS vector database
- Retrieve relevant messages based on semantic similarity
- Generate contextual answers using GPT-4.0 mini
- Interactive chat interface and command-line query mode
- Discord bot integration with slash commands and message citations

## Prerequisites

- Python 3.8+
- OpenAI API key for GPT-4.0 mini generation
- Discord bot token (for Discord bot functionality)

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Copy `.env.example` to `.env` and set your credentials:
```bash
cp .env.example .env
# Edit .env with your OpenAI API key and Discord bot token
```

## Usage

### 1. Build Vector Store

First, create the vector store from your Discord chat export(s):

```bash
# Single JSON file
python build_rag.py --input your_discord_export.json

# Directory containing multiple JSON files
python build_rag.py --input /path/to/discord/exports/

# Using OpenAI embeddings
python build_rag.py --input /path/to/discord/exports/ --embedder openai --embedder-model text-embedding-3-large

# Build and test retrieval
python build_rag.py --input /path/to/discord/exports/ --test
```

Options:
- `--input`: Path to Discord JSON export file or directory containing JSON files
- `--output`: Path to save vector store (default: ./discord_vector_store)
- `--embedder`: Embedder type - 'qwen' or 'openai' (default: qwen)
- `--embedder-model`: Specific embedding model to use
- `--test`: Run test queries after building

### 2. Chat with the Bot

#### Interactive Mode
```bash
python chat.py
```

#### Single Query Mode
```bash
python chat.py --query "What is the Vesuvius Challenge?"
```

Options:
- `--vector-store`: Path to vector store directory
- `--api-key`: OpenAI API key (defaults to OPENAI_API_KEY env var)
- `--k`: Number of documents to retrieve (default: 5)
- `--temperature`: GPT-4 temperature 0-1 (default: 0.7)
- `--max-tokens`: Max tokens for response (default: 500)
- `--query`: Single query for non-interactive mode

### 3. Test the Pipeline

Run the test script to verify everything is working:

```bash
python test_rag_pipeline.py
```

### 4. Run Discord Bot

Start the Discord bot with slash commands:

```bash
python discord_bot.py
```

Discord Commands:
- `/ask [question]`: Ask a question about the Discord chat history
- `/rag_help`: Get help with the RAG chatbot

The bot will respond with:
- AI-generated answer based on relevant messages
- Citations with jump links to original messages
- Relevance scores for each source

## Architecture

1. **Discord Parser** (`discord_parser.py`): Extracts messages from Discord JSON exports
2. **Embedder** (`embedder.py`): Creates vector embeddings using Qwen or OpenAI models
3. **Vector Store** (`vector_store.py`): Manages FAISS vector database with metadata
4. **RAG Chatbot** (`rag_chatbot.py`): Retrieves context and generates answers using GPT-4.0 mini
5. **Chat Interface** (`chat.py`): Command-line interface for interaction
6. **Discord Bot** (`discord_bot.py`): Discord bot with slash commands and message citations

## How It Works

1. **Indexing Phase**:
   - Parse Discord messages from JSON export
   - Create embeddings for each message
   - Store embeddings and metadata in FAISS vector database
   - Save embedder configuration for consistent retrieval

2. **Query Phase**:
   - Embed user query using same model as indexing
   - Find k most similar messages using vector similarity
   - Format retrieved messages as context
   - Generate answer using GPT-4.0 mini with context

## Example

```bash
# Build vector store from single file
python build_rag.py --input sample_text.json --test

# Build from directory with multiple Discord exports
python build_rag.py --input /opt/dlami/nvme/discordout-2025-07-08 --output ./discord_vector_store_full

# Or use the convenience script
./build_from_directory.sh

# Ask questions about the chat history
python chat.py --query "How do I get started with the Vesuvius Challenge?" --vector-store ./discord_vector_store_full

# Interactive chat session
python chat.py --vector-store ./discord_vector_store_full
```

## Notes

- The system preserves message metadata (author, channel, timestamp, message IDs)
- Embedder configuration is saved to ensure consistent retrieval
- GPT-4.0 mini is used for cost-effective generation
- Sources are provided with each answer for transparency
- Discord bot creates clickable jump links to original messages
- Vector store must be rebuilt to include new metadata fields for existing data
- OpenAI embeddings use batch processing (100 texts per request) for efficiency
- Progress tracking shows elapsed time, remaining time estimate, and processing speed