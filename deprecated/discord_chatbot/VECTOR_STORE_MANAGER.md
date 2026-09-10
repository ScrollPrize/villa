# Vector Store Manager

A utility script for inspecting and managing the Discord vector store.

## Usage

### Basic Commands

```bash
# Show basic summary
python vector_store_manager.py

# Show detailed statistics
python vector_store_manager.py --stats

# Plot message length histograms
python vector_store_manager.py --plot

# Search for messages containing text
python vector_store_manager.py --search "vesuvius challenge"

# Search in specific field (author, channel)
python vector_store_manager.py --search "john" --search-field author

# Export all messages to JSON
python vector_store_manager.py --export messages_backup.json

# Show more results (default is 10)
python vector_store_manager.py --search "scroll" --limit 20

# Use a different vector store path
python vector_store_manager.py --path ./discord_vector_store_full --stats
```

### Message Removal

```bash
# Remove very short messages (dry run by default)
python vector_store_manager.py --min-length 10

# Remove very long messages
python vector_store_manager.py --max-length 5000

# Remove messages from specific authors
python vector_store_manager.py --remove-authors "bot1" "spammer"

# Remove messages from specific channels
python vector_store_manager.py --remove-channels "spam" "test"

# Keep only messages from specific authors
python vector_store_manager.py --keep-authors "john" "alice"

# Keep only messages from specific channels
python vector_store_manager.py --keep-channels "general" "announcements"

# Combine criteria (remove short messages from specific channels)
python vector_store_manager.py --max-length 50 --remove-channels "memes" "random"

# Actually perform removal (not a dry run)
python vector_store_manager.py --min-length 5 --no-dry-run

# Remove by specific message IDs
python vector_store_manager.py --remove 123456789 987654321
```

## Advanced Workflow: JSON Export/Edit/Import

For complex filtering or manual editing:

```bash
# 1. Export vector store to JSON WITH embeddings (preserves embeddings)
python vector_store_manager.py --export-with-embeddings messages_with_embeddings.json

# 2. Edit the JSON file using the helper script or manually
python edit_vector_json.py messages_with_embeddings.json filtered.json \
  --remove-channels "spam" "test" \
  --min-length 10 \
  --max-length 5000 \
  --remove-urls \
  --show-sample 5

# Or edit manually - the JSON structure is:
# {
#   "documents": [
#     {
#       "content": "message text",
#       "metadata": { "author": "...", "channel": "...", ... },
#       "embedding": [0.123, -0.456, ...]  // embedding vector
#     },
#     ...
#   ],
#   "embedder_metadata": { "embedder_type": "qwen" },
#   "total_documents": 12345,
#   "includes_embeddings": true
# }

# 3. Create new vector store from edited JSON (reuses embeddings!)
python vector_store_manager.py --import-json messages_edited.json --import-output ./new_vector_store

# Alternative: Export without embeddings (smaller file, but slower import)
python vector_store_manager.py --export messages_no_embeddings.json
```

## Features

- **Statistics**: View total messages, character counts, top authors, top channels
- **Histograms**: Visualize message length distribution (linear and log scale)
- **Search**: Find messages by content, author, or channel
- **Export/Import**: 
  - Export to JSON with or without embeddings
  - Edit JSON externally for complex filtering
  - Import from JSON, preserving embeddings to avoid re-computation
- **Message Preview**: See message details including IDs for debugging
- **Bulk Removal**: Remove messages by length, author, or channel criteria
- **Dry Run**: Preview what would be removed before actually doing it
- **Automatic Backup**: Creates timestamped backup before rebuilding vector store

## Safety Features

- All removal operations are **dry run by default** - use `--no-dry-run` to actually remove
- Automatic backup created before any vector store rebuild
- Shows preview of messages to be removed before confirmation
- Requires explicit confirmation for actual removal