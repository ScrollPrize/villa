#!/bin/bash

# Script to build vector store from Discord export directory

DISCORD_EXPORT_DIR="/opt/dlami/nvme/discordout-2025-07-08"
OUTPUT_DIR="./discord_vector_store_full_filtered"
EMBEDDER_TYPE="openai"
EMBEDDER_MODEL="text-embedding-3-large"

echo "Building Discord RAG vector store from directory: $DISCORD_EXPORT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Embedder: $EMBEDDER_TYPE ($EMBEDDER_MODEL)"
echo ""

# Check if directory exists
if [ ! -d "$DISCORD_EXPORT_DIR" ]; then
    echo "Error: Directory $DISCORD_EXPORT_DIR does not exist!"
    exit 1
fi

# Count JSON files
JSON_COUNT=$(find "$DISCORD_EXPORT_DIR" -name "*.json" -type f | wc -l)
echo "Found $JSON_COUNT JSON files in directory"
echo ""

# Build the vector store
python3 build_rag.py \
    --input "$DISCORD_EXPORT_DIR" \
    --output "$OUTPUT_DIR" \
    --embedder "$EMBEDDER_TYPE" \
    --embedder-model "$EMBEDDER_MODEL"

# Check if build was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Vector store built successfully!"
    echo "You can now use it with:"
    echo "  python3 chat.py --vector-store $OUTPUT_DIR"
    echo "  python3 discord_bot.py  # (set VECTOR_STORE_PATH=$OUTPUT_DIR in .env)"
else
    echo ""
    echo "❌ Error building vector store"
    exit 1
fi
