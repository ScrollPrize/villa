#!/usr/bin/env python3
"""
Vector Store Manager - Utility script for managing Discord vector store
"""

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional
from datetime import datetime
from collections import Counter
import argparse
from tabulate import tabulate

from langchain_community.vectorstores import FAISS
from langchain_community.docstore.document import Document
from dotenv import load_dotenv

load_dotenv()


class VectorStoreManager:
    def __init__(self, vector_store_path: str = "./discord_vector_store"):
        self.vector_store_path = vector_store_path
        self.vector_store = None
        self.documents = []
        self._load_vector_store()
    
    def _load_vector_store(self):
        """Load the vector store from disk"""
        try:
            # First, check which embedder was used
            embedder_metadata_path = os.path.join(self.vector_store_path, "embedder_metadata.json")
            if os.path.exists(embedder_metadata_path):
                import json
                with open(embedder_metadata_path, 'r') as f:
                    metadata = json.load(f)
                    embedder_type = metadata.get('embedder_type', 'unknown')
                    print(f"Vector store was created with embedder: {embedder_type}")
            
            # Load the FAISS vector store properly
            from embedder import create_embedder
            embedder = create_embedder(embedder_type if 'embedder_type' in locals() else 'qwen')
            
            self.vector_store = FAISS.load_local(
                self.vector_store_path, 
                embedder,
                allow_dangerous_deserialization=True
            )
            print(f"✓ Loaded vector store from {self.vector_store_path}")
            
            # Extract documents
            self._extract_documents()
        except Exception as e:
            print(f"✗ Error loading vector store: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    def _extract_documents(self):
        """Extract all documents from the vector store"""
        try:
            # Method 1: Try to get all documents via similarity search with a dummy query
            if hasattr(self.vector_store, 'docstore'):
                # Access the docstore directly
                if hasattr(self.vector_store.docstore, '_dict'):
                    self.documents = list(self.vector_store.docstore._dict.values())
                    print(f"Extracted {len(self.documents)} documents from docstore")
                else:
                    # Try to iterate through all document IDs
                    self.documents = []
                    if hasattr(self.vector_store, 'index_to_docstore_id'):
                        for doc_id in self.vector_store.index_to_docstore_id.values():
                            try:
                                doc = self.vector_store.docstore.search(doc_id)
                                if doc and doc != "Document not found.":
                                    self.documents.append(doc)
                            except:
                                pass
                    print(f"Extracted {len(self.documents)} documents via ID lookup")
            else:
                print("Warning: Could not access docstore directly")
                self.documents = []
        except Exception as e:
            print(f"Error extracting documents: {e}")
            import traceback
            traceback.print_exc()
            self.documents = []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        stats = {
            "total_messages": len(self.documents),
            "total_characters": sum(len(doc.page_content) for doc in self.documents),
            "avg_message_length": 0,
            "authors": Counter(),
            "channels": Counter(),
            "dates": Counter(),
            "message_lengths": []
        }
        
        if self.documents:
            stats["avg_message_length"] = stats["total_characters"] / stats["total_messages"]
            
            for doc in self.documents:
                # Message length
                stats["message_lengths"].append(len(doc.page_content))
                
                # Author stats
                author = doc.metadata.get("author", "Unknown")
                stats["authors"][author] += 1
                
                # Channel stats
                channel = doc.metadata.get("channel", "Unknown")
                stats["channels"][channel] += 1
                
                # Date stats (group by day)
                timestamp = doc.metadata.get("timestamp", "")
                if timestamp:
                    try:
                        date = timestamp.split("T")[0]
                        stats["dates"][date] += 1
                    except:
                        pass
        
        return stats
    
    def display_stats(self):
        """Display statistics in a formatted way"""
        stats = self.get_stats()
        
        print("\n" + "="*60)
        print("VECTOR STORE STATISTICS")
        print("="*60)
        
        print(f"\nTotal Messages: {stats['total_messages']:,}")
        print(f"Total Characters: {stats['total_characters']:,}")
        print(f"Average Message Length: {stats['avg_message_length']:.1f} characters")
        
        # Top authors
        print(f"\nTop 30 Authors:")
        top_authors = stats["authors"].most_common(30)
        author_table = [[author, count] for author, count in top_authors]
        print(tabulate(author_table, headers=["Author", "Messages"], tablefmt="grid"))
        
        # Top channels
        print(f"\nTop 10 Channels:")
        top_channels = stats["channels"].most_common(10)
        channel_table = [[channel, count] for channel, count in top_channels]
        print(tabulate(channel_table, headers=["Channel", "Messages"], tablefmt="grid"))
        
        # Message length distribution
        lengths = stats["message_lengths"]
        if lengths:
            print(f"\nMessage Length Distribution:")
            print(f"  Min: {min(lengths)} chars")
            print(f"  Max: {max(lengths):,} chars")
            print(f"  Median: {np.median(lengths):.0f} chars")
            print(f"  95th percentile: {np.percentile(lengths, 95):.0f} chars")
    
    def plot_histograms(self):
        """Create histograms for message length distribution"""
        stats = self.get_stats()
        lengths = stats["message_lengths"]
        
        if not lengths:
            print("No messages to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Linear scale histogram
        ax1.hist(lengths, bins=50, edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Message Length (characters)')
        ax1.set_ylabel('Count')
        ax1.set_title('Message Length Distribution (Linear Scale)')
        ax1.grid(True, alpha=0.3)
        
        # Log scale histogram
        ax2.hist(lengths, bins=50, edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Message Length (characters)')
        ax2.set_ylabel('Count (log scale)')
        ax2.set_title('Message Length Distribution (Log Scale)')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('message_length_distribution.png', dpi=150)
        print("\n✓ Saved histogram to message_length_distribution.png")
        plt.show()
    
    def search_messages(self, query: str, field: str = "content") -> List[Document]:
        """Search for messages containing specific text"""
        results = []
        query_lower = query.lower()
        
        for doc in self.documents:
            if field == "content":
                if query_lower in doc.page_content.lower():
                    results.append(doc)
            elif field in doc.metadata:
                if query_lower in str(doc.metadata[field]).lower():
                    results.append(doc)
        
        return results
    
    def display_messages(self, messages: List[Document], limit: int = 10):
        """Display messages in a formatted way"""
        print(f"\nShowing {min(limit, len(messages))} of {len(messages)} messages:")
        print("-" * 80)
        
        for i, doc in enumerate(messages[:limit]):
            print(f"\n[{i+1}] {doc.metadata.get('author', 'Unknown')} - {doc.metadata.get('timestamp', 'Unknown')}")
            print(f"    Channel: {doc.metadata.get('channel', 'Unknown')}")
            print(f"    Length: {len(doc.page_content)} chars")
            print(f"    Preview: {doc.page_content[:200]}..." if len(doc.page_content) > 200 else f"    Content: {doc.page_content}")
            print(f"    Message ID: {doc.metadata.get('message_id', 'Unknown')}")
    
    def filter_messages(self, 
                       min_length: Optional[int] = None,
                       max_length: Optional[int] = None,
                       authors: Optional[List[str]] = None,
                       channels: Optional[List[str]] = None,
                       exclude_authors: Optional[List[str]] = None,
                       exclude_channels: Optional[List[str]] = None) -> List[Document]:
        """Filter messages based on various criteria
        
        Returns messages that should be REMOVED based on the criteria.
        """
        # Convert lists to lowercase sets for faster lookup
        authors_set = set(a.lower() for a in (authors or [])) if authors else None
        channels_set = set(c.lower() for c in (channels or [])) if channels else None
        exclude_authors_set = set(a.lower() for a in (exclude_authors or [])) if exclude_authors else None
        exclude_channels_set = set(c.lower() for c in (exclude_channels or [])) if exclude_channels else None
        
        to_remove = []
        total_docs = len(self.documents)
        
        print(f"Filtering {total_docs} messages...")
        
        for i, doc in enumerate(self.documents):
            if i % 1000 == 0 and i > 0:
                print(f"  Progress: {i}/{total_docs} ({i/total_docs*100:.1f}%)")
            
            should_remove = False
            
            # Get metadata once
            author = doc.metadata.get('author', '').lower()
            channel = doc.metadata.get('channel', '').lower()
            
            # Use exact matching with sets for better performance
            # Remove if author is in exclude list
            if exclude_authors_set and author in exclude_authors_set:
                should_remove = True
            
            # Remove if channel is in exclude list
            elif exclude_channels_set and channel in exclude_channels_set:
                should_remove = True
            
            # Remove if author is NOT in keep list (when keep list exists)
            elif authors_set and author not in authors_set:
                should_remove = True
            
            # Remove if channel is NOT in keep list (when keep list exists)
            elif channels_set and channel not in channels_set:
                should_remove = True
            
            # Check length criteria only if not already marked for removal
            # and only if no author/channel criteria were specified
            elif not any([authors_set, channels_set, exclude_authors_set, exclude_channels_set]):
                content_length = len(doc.page_content)
                if min_length and content_length < min_length:
                    should_remove = True
                elif max_length and content_length > max_length:
                    should_remove = True
            
            # If length criteria AND author/channel criteria are specified,
            # treat them as OR conditions (remove if matches ANY criteria)
            else:
                content_length = len(doc.page_content)
                if min_length and content_length < min_length:
                    should_remove = True
                elif max_length and content_length > max_length:
                    should_remove = True
            
            if should_remove:
                to_remove.append(doc)
        
        print(f"  Found {len(to_remove)} messages to remove")
        return to_remove
    
    def remove_messages_by_criteria(self,
                                  min_length: Optional[int] = None,
                                  max_length: Optional[int] = None,
                                  authors: Optional[List[str]] = None,
                                  channels: Optional[List[str]] = None,
                                  exclude_authors: Optional[List[str]] = None,
                                  exclude_channels: Optional[List[str]] = None,
                                  dry_run: bool = True):
        """Remove messages matching specific criteria"""
        # Find messages to remove
        to_remove = self.filter_messages(
            min_length=min_length,
            max_length=max_length,
            authors=authors,
            channels=channels,
            exclude_authors=exclude_authors,
            exclude_channels=exclude_channels
        )
        
        if not to_remove:
            print("No messages match the removal criteria")
            return
        
        # Calculate what will remain
        to_keep = [doc for doc in self.documents if doc not in to_remove]
        
        print(f"\nRemoval Summary:")
        print(f"  Total messages: {len(self.documents)}")
        print(f"  Messages to remove: {len(to_remove)}")
        print(f"  Messages to keep: {len(to_keep)}")
        
        # Show sample of messages to be removed
        print(f"\nSample of messages to be removed (showing up to 10):")
        self.display_messages(to_remove, limit=10)
        
        if dry_run:
            print("\n** DRY RUN - No actual removal performed **")
            print("Use --no-dry-run to actually remove messages")
        else:
            response = input("\nProceed with removal and rebuild vector store? (yes/no): ")
            if response.lower() == 'yes':
                self._rebuild_vector_store(to_keep)
    
    def _rebuild_vector_store(self, documents_to_keep: List[Document]):
        """Rebuild the vector store with only the specified documents"""
        try:
            from embedder import create_embedder
            import json
            import shutil
            from langchain_community.vectorstores import FAISS
            
            # Load embedder metadata
            embedder_metadata_path = os.path.join(self.vector_store_path, "embedder_metadata.json")
            embedder_type = 'qwen'
            if os.path.exists(embedder_metadata_path):
                with open(embedder_metadata_path, 'r') as f:
                    metadata = json.load(f)
                    embedder_type = metadata.get('embedder_type', 'qwen')
            
            print(f"\nRebuilding vector store with {len(documents_to_keep)} documents...")
            
            # Create backup
            backup_path = f"{self.vector_store_path}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copytree(self.vector_store_path, backup_path)
            print(f"Created backup at: {backup_path}")
            
            # Create new vector store
            embedder = create_embedder(embedder_type)
            
            # Extract texts and metadatas
            texts = [doc.page_content for doc in documents_to_keep]
            metadatas = [doc.metadata for doc in documents_to_keep]
            
            # Create new FAISS index
            new_vector_store = FAISS.from_texts(
                texts=texts,
                embedding=embedder,
                metadatas=metadatas
            )
            
            # Save the new vector store
            new_vector_store.save_local(self.vector_store_path)
            
            # Save embedder metadata
            with open(embedder_metadata_path, 'w') as f:
                json.dump({'embedder_type': embedder_type}, f)
            
            print(f"✓ Successfully rebuilt vector store")
            print(f"  Original messages: {len(self.documents)}")
            print(f"  New messages: {len(documents_to_keep)}")
            print(f"  Removed: {len(self.documents) - len(documents_to_keep)}")
            
        except Exception as e:
            print(f"✗ Error rebuilding vector store: {e}")
            import traceback
            traceback.print_exc()
    
    def remove_messages(self, message_ids: List[str]):
        """Remove messages by their Discord message IDs"""
        to_keep = [doc for doc in self.documents 
                  if doc.metadata.get('message_id') not in message_ids]
        
        to_remove = [doc for doc in self.documents 
                    if doc.metadata.get('message_id') in message_ids]
        
        if to_remove:
            print(f"\nFound {len(to_remove)} messages to remove:")
            self.display_messages(to_remove)
            
            response = input("\nProceed with removal? (yes/no): ")
            if response.lower() == 'yes':
                self._rebuild_vector_store(to_keep)
        else:
            print("No messages found with the specified IDs")
    
    def export_messages(self, output_file: str = "vector_store_export.json", include_embeddings: bool = False):
        """Export all messages to a JSON file, optionally with embeddings"""
        import json
        import base64
        
        export_data = []
        
        # Get embeddings if requested
        embeddings_dict = {}
        if include_embeddings and hasattr(self.vector_store, 'index'):
            print("Extracting embeddings...")
            # Get all embeddings from FAISS index
            try:
                for i, doc_id in enumerate(self.vector_store.index_to_docstore_id.values()):
                    if i < self.vector_store.index.ntotal:
                        # Get embedding vector from FAISS
                        embedding = self.vector_store.index.reconstruct(i)
                        # Convert to list and store with doc_id
                        embeddings_dict[doc_id] = embedding.tolist()
            except Exception as e:
                print(f"Warning: Could not extract embeddings: {e}")
                include_embeddings = False
        
        # Export documents with metadata
        for i, doc in enumerate(self.documents):
            doc_data = {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            
            # Add embedding if available
            if include_embeddings:
                # Find the doc_id for this document
                doc_id = None
                for idx, did in self.vector_store.index_to_docstore_id.items():
                    if did in self.vector_store.docstore._dict and self.vector_store.docstore._dict[did] == doc:
                        doc_id = did
                        break
                
                if doc_id and doc_id in embeddings_dict:
                    doc_data["embedding"] = embeddings_dict[doc_id]
            
            export_data.append(doc_data)
        
        # Save embedder metadata
        embedder_metadata = None
        embedder_metadata_path = os.path.join(self.vector_store_path, "embedder_metadata.json")
        if os.path.exists(embedder_metadata_path):
            with open(embedder_metadata_path, 'r') as f:
                embedder_metadata = json.load(f)
        
        export_package = {
            "documents": export_data,
            "embedder_metadata": embedder_metadata,
            "total_documents": len(export_data),
            "export_date": datetime.now().isoformat(),
            "includes_embeddings": include_embeddings
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_package, f, indent=2, ensure_ascii=False)
        
        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"✓ Exported {len(export_data)} messages to {output_file} ({file_size_mb:.1f} MB)")
        if include_embeddings:
            print(f"  Embeddings included: Yes")
        else:
            print(f"  Embeddings included: No (will need re-embedding when importing)")


def create_vector_store_from_json(json_file: str, output_path: str):
    """Create a new vector store from an exported JSON file"""
    import json
    from embedder import create_embedder
    from langchain_community.vectorstores import FAISS
    
    print(f"Loading data from {json_file}...")
    with open(json_file, 'r', encoding='utf-8') as f:
        export_package = json.load(f)
    
    # Handle both old format (list) and new format (dict)
    if isinstance(export_package, list):
        # Old format - just a list of documents
        documents = export_package
        embedder_metadata = {'embedder_type': 'qwen'}  # Default
        includes_embeddings = False
    else:
        # New format with metadata
        documents = export_package.get('documents', [])
        embedder_metadata = export_package.get('embedder_metadata', {'embedder_type': 'qwen'})
        includes_embeddings = export_package.get('includes_embeddings', False)
    
    print(f"Found {len(documents)} documents")
    print(f"Embedder type: {embedder_metadata.get('embedder_type', 'unknown')}")
    print(f"Includes embeddings: {includes_embeddings}")
    
    # Create embedder
    embedder = create_embedder(embedder_metadata.get('embedder_type', 'qwen'))
    
    if includes_embeddings and all('embedding' in doc for doc in documents):
        print("Using existing embeddings from JSON file...")
        
        # Extract texts, metadatas, and embeddings
        texts = [doc['content'] for doc in documents]
        metadatas = [doc['metadata'] for doc in documents]
        embeddings = [doc['embedding'] for doc in documents]
        
        # Create FAISS index directly with embeddings
        text_embedding_pairs = list(zip(texts, embeddings))
        vector_store = FAISS.from_embeddings(
            text_embeddings=text_embedding_pairs,
            embedding=embedder,
            metadatas=metadatas
        )
        print(f"✓ Created vector store with {len(documents)} documents (using existing embeddings)")
    else:
        print("Re-embedding documents (this may take a while)...")
        
        # Extract texts and metadatas
        texts = [doc['content'] for doc in documents]
        metadatas = [doc['metadata'] for doc in documents]
        
        # Create new FAISS index (will embed the texts)
        vector_store = FAISS.from_texts(
            texts=texts,
            embedding=embedder,
            metadatas=metadatas
        )
        print(f"✓ Created vector store with {len(documents)} documents (new embeddings)")
    
    # Save the vector store
    print(f"Saving vector store to {output_path}...")
    vector_store.save_local(output_path)
    
    # Save embedder metadata
    embedder_metadata_path = os.path.join(output_path, "embedder_metadata.json")
    with open(embedder_metadata_path, 'w') as f:
        json.dump(embedder_metadata, f)
    
    print(f"✓ Vector store saved to {output_path}")
    print(f"✓ Embedder metadata saved to {embedder_metadata_path}")


def main():
    parser = argparse.ArgumentParser(description="Manage Discord Vector Store")
    parser.add_argument("--path", default="./discord_vector_store_full", help="Path to vector store")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--plot", action="store_true", help="Plot message length histograms")
    parser.add_argument("--search", help="Search for messages containing text")
    parser.add_argument("--search-field", default="content", help="Field to search in (content, author, channel)")
    parser.add_argument("--export", help="Export messages to JSON file")
    parser.add_argument("--export-with-embeddings", help="Export messages to JSON file with embeddings (larger file)")
    parser.add_argument("--import-json", help="Import vector store from JSON file")
    parser.add_argument("--import-output", default="./imported_vector_store", help="Output path for imported vector store")
    parser.add_argument("--remove", nargs="+", help="Remove messages by Discord message ID")
    parser.add_argument("--limit", type=int, default=10, help="Limit number of messages to display")
    
    # Filtering options for removal
    parser.add_argument("--remove-by-length", action="store_true", help="Remove messages by length criteria")
    parser.add_argument("--min-length", type=int, help="Minimum message length to keep")
    parser.add_argument("--max-length", type=int, help="Maximum message length to keep")
    parser.add_argument("--remove-authors", nargs="+", help="Remove messages from these authors")
    parser.add_argument("--remove-channels", nargs="+", help="Remove messages from these channels")
    parser.add_argument("--keep-authors", nargs="+", help="Only keep messages from these authors")
    parser.add_argument("--keep-channels", nargs="+", help="Only keep messages from these channels")
    parser.add_argument("--no-dry-run", action="store_true", help="Actually perform removal (default is dry run)")
    
    args = parser.parse_args()
    
    # Handle import first (doesn't need manager)
    if args.import_json:
        create_vector_store_from_json(args.import_json, args.import_output)
        return
    
    # Initialize manager for other operations
    manager = VectorStoreManager(args.path)
    
    # Execute requested actions
    if args.stats:
        manager.display_stats()
    
    if args.plot:
        manager.plot_histograms()
    
    if args.search:
        results = manager.search_messages(args.search, args.search_field)
        if results:
            print(f"\nFound {len(results)} messages matching '{args.search}'")
            manager.display_messages(results, args.limit)
        else:
            print(f"No messages found matching '{args.search}'")
    
    if args.export:
        manager.export_messages(args.export, include_embeddings=False)
    
    if args.export_with_embeddings:
        manager.export_messages(args.export_with_embeddings, include_embeddings=True)
    
    if args.remove:
        manager.remove_messages(args.remove)
    
    # Handle filtering-based removal
    if any([args.remove_by_length, args.remove_authors, args.remove_channels, 
            args.keep_authors, args.keep_channels]):
        
        # Convert keep_* to exclude_* logic
        exclude_authors = None
        exclude_channels = None
        authors = None
        channels = None
        
        if args.keep_authors:
            # If keeping specific authors, we'll filter to only those
            authors = args.keep_authors
        elif args.remove_authors:
            # If removing specific authors, we'll exclude them
            exclude_authors = args.remove_authors
            
        if args.keep_channels:
            # If keeping specific channels, we'll filter to only those
            channels = args.keep_channels
        elif args.remove_channels:
            # If removing specific channels, we'll exclude them
            exclude_channels = args.remove_channels
        
        manager.remove_messages_by_criteria(
            min_length=args.min_length,
            max_length=args.max_length,
            authors=authors,
            channels=channels,
            exclude_authors=exclude_authors,
            exclude_channels=exclude_channels,
            dry_run=not args.no_dry_run
        )
    
    # If no specific action requested, show basic stats
    if not any([args.stats, args.plot, args.search, args.export, args.remove,
                args.remove_by_length, args.remove_authors, args.remove_channels,
                args.keep_authors, args.keep_channels]):
        print("\nVector Store Summary:")
        stats = manager.get_stats()
        print(f"Total Messages: {stats['total_messages']:,}")
        print(f"Total Characters: {stats['total_characters']:,}")
        print(f"\nUse --help to see available options")


if __name__ == "__main__":
    main()