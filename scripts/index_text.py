#!/usr/bin/env python3
"""
Document Ingestion Script - Extract, Chunk, Embed, and Upsert to Qdrant

Usage:
    python scripts/index_text.py --data-root /data --collection hydrogen_books
    python scripts/index_text.py --data-root /data --all-collections
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import hashlib
import json
from datetime import datetime

from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader
from llama_index.core import Document
from llama_index.core.node_parser import HierarchicalNodeParser
from langchain_openai import AzureOpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    PayloadSchemaType, TextIndexParams,
    TokenizerType, OptimizersConfigDiff,
    HnswConfigDiff, ScalarQuantization,
    ScalarQuantizationConfig, ScalarType
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for document ingestion"""
    
    # Qdrant
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_TIMEOUT = 60
    
    # Azure OpenAI
    AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")
    EMBEDDING_MODEL = "text-embedding-ada-002"
    EMBEDDING_DIMENSION = 1536
    
    # Chunking
    PARENT_CHUNK_SIZE = 2000
    CHILD_CHUNK_SIZE = 500
    CHUNK_OVERLAP = 150
    
    # Processing
    BATCH_SIZE = 100
    
    # State directory (for reports)
    STATE_DIR = "/state"


# ============================================================================
# QDRANT CLIENT INITIALIZATION
# ============================================================================

def get_qdrant_client() -> QdrantClient:
    """Initialize and return Qdrant client"""
    return QdrantClient(
        url=Config.QDRANT_URL,
        port=Config.QDRANT_PORT,
        timeout=Config.QDRANT_TIMEOUT
    )


def get_embeddings_model() -> AzureOpenAIEmbeddings:
    """Initialize and return embeddings model"""
    return AzureOpenAIEmbeddings(
        model=Config.EMBEDDING_MODEL,
        azure_endpoint=Config.AZURE_ENDPOINT,
        api_key=Config.AZURE_API_KEY,
        openai_api_version=Config.AZURE_API_VERSION
    )


# ============================================================================
# DOCUMENT LOADING & EXTRACTION
# ============================================================================

def extract_text_from_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Extract text from PDF or DOCX file with page/section awareness
    
    Args:
        file_path: Path to the document file
        
    Returns:
        List of page/section dictionaries with text and metadata
    """
    file_path = Path(file_path)
    extension = file_path.suffix.lower()
    
    pages = []
    
    try:
        if extension == '.pdf':
            loader = PyMuPDFLoader(str(file_path))
            docs = loader.load()
            
            for doc in docs:
                pages.append({
                    'text': doc.page_content,
                    'page_number': doc.metadata.get('page', 0) + 1,  # 1-indexed
                    'metadata': doc.metadata
                })
                
        elif extension in ['.docx', '.doc']:
            loader = Docx2txtLoader(str(file_path))
            docs = loader.load()
            
            # DOCX doesn't have page numbers, treat as single section
            for idx, doc in enumerate(docs):
                pages.append({
                    'text': doc.page_content,
                    'page_number': idx + 1,
                    'metadata': doc.metadata
                })
        else:
            logger.warning(f"Unsupported file type: {extension}")
            return []
            
    except Exception as e:
        logger.error(f"Failed to extract text from {file_path}: {e}")
        return []
    
    logger.info(f"Extracted {len(pages)} pages from {file_path.name}")
    return pages


# ============================================================================
# DOCUMENT ID & METADATA GENERATION
# ============================================================================

def generate_doc_id(file_path: str) -> str:
    """Generate unique document ID from file path"""
    return hashlib.md5(str(file_path).encode()).hexdigest()


def generate_file_metadata(file_path: str, collection_name: str) -> Dict[str, Any]:
    """
    Generate comprehensive metadata for a file
    
    Args:
        file_path: Path to the document
        collection_name: Name of the target collection
        
    Returns:
        Metadata dictionary
    """
    file_path = Path(file_path)
    
    # Determine collection folder from file path
    # Assumes structure: /data-root/collection_name/file.pdf
    parts = file_path.parts
    
    # Find the data root index (parent of collection folder)
    collection_folder = None
    for i, part in enumerate(parts):
        if i < len(parts) - 1 and parts[i + 1] == file_path.parent.name:
            collection_folder = parts[i + 1]
            break
    
    return {
        'doc_id': generate_doc_id(str(file_path)),
        'file_name': file_path.name,
        'file_stem': file_path.stem,
        'file_extension': file_path.suffix,
        'file_size': file_path.stat().st_size if file_path.exists() else 0,
        'collection': collection_name,
        'folder_name': file_path.parent.name,
        'full_path': str(file_path.absolute()),
        'created_at': datetime.utcnow().isoformat()
    }


# ============================================================================
# HIERARCHICAL CHUNKING
# ============================================================================

def hierarchical_chunk_pages(
    pages: List[Dict[str, Any]],
    file_metadata: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Create hierarchical chunks from extracted pages
    
    Args:
        pages: List of page dictionaries
        file_metadata: File-level metadata
        
    Returns:
        List of chunk dictionaries with parent-child relationships
    """
    # Combine all pages into one document for hierarchical parsing
    full_text = "\n\n".join([p['text'] for p in pages])
    
    # Create LlamaIndex document
    doc = Document(text=full_text, metadata=file_metadata)
    
    # Hierarchical parser: parent (2000 tokens) -> children (500 tokens)
    parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=[Config.PARENT_CHUNK_SIZE, Config.CHILD_CHUNK_SIZE],
        chunk_overlap=Config.CHUNK_OVERLAP
    )
    
    nodes = parser.get_nodes_from_documents([doc])
    
    # Build parent text lookup
    parent_text_map = {}
    for node in nodes:
        if not node.parent_node:  # This is a parent
            parent_text_map[node.id_] = node.text
    
    # Process nodes and match to pages
    chunks = []
    for node in nodes:
        # Determine which page this chunk belongs to
        page_number = _find_page_for_chunk(node.text, pages)
        
        chunk_data = {
            'id': node.id_,
            'text': node.text,
            'is_leaf': bool(node.parent_node),
            'parent_id': node.parent_node.id_ if node.parent_node else None,
            'parent_text': parent_text_map.get(node.parent_node.id_) if node.parent_node else None,
            'page_number': page_number,
            'metadata': node.metadata
        }
        chunks.append(chunk_data)
    
    logger.info(f"Created {len(chunks)} hierarchical chunks")
    return chunks


def _find_page_for_chunk(chunk_text: str, pages: List[Dict[str, Any]]) -> int:
    """
    Find which page a chunk belongs to by matching text
    
    Args:
        chunk_text: The chunk text
        pages: List of page dictionaries
        
    Returns:
        Page number (1-indexed)
    """
    # Simple heuristic: find the page with most overlap
    best_page = 1
    max_overlap = 0
    
    for page in pages:
        # Count common words
        chunk_words = set(chunk_text.lower().split())
        page_words = set(page['text'].lower().split())
        overlap = len(chunk_words & page_words)
        
        if overlap > max_overlap:
            max_overlap = overlap
            best_page = page['page_number']
    
    return best_page


# ============================================================================
# QDRANT COLLECTION SETUP
# ============================================================================

def setup_collection(
    client: QdrantClient,
    collection_name: str,
    recreate: bool = False
) -> None:
    """
    Setup Qdrant collection with optimized configuration
    
    Args:
        client: Qdrant client
        collection_name: Name of collection to create
        recreate: Whether to recreate if exists
    """
    collections = client.get_collections().collections
    exists = any(c.name == collection_name for c in collections)
    
    if exists:
        if recreate:
            logger.info(f"Deleting existing collection: {collection_name}")
            client.delete_collection(collection_name)
        else:
            logger.info(f"Collection '{collection_name}' already exists")
            return
    
    logger.info(f"Creating collection: {collection_name}")
    
    # Create collection with vector configuration
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=Config.EMBEDDING_DIMENSION,
            distance=Distance.COSINE,
            hnsw_config=HnswConfigDiff(
                m=16,
                ef_construct=100,
                full_scan_threshold=10000
            )
        ),
        quantization_config=ScalarQuantization(
            scalar=ScalarQuantizationConfig(
                type=ScalarType.INT8,
                quantile=0.99,
                always_ram=True
            )
        ),
        optimizers_config=OptimizersConfigDiff(
            indexing_threshold=20000,
            memmap_threshold=50000
        )
    )
    
    # Create payload indexes
    _create_payload_indexes(client, collection_name)
    
    logger.info(f"Collection '{collection_name}' created successfully")


def _create_payload_indexes(client: QdrantClient, collection_name: str) -> None:
    """Create all required payload indexes"""
    
    indexes = [
        ("doc_id", PayloadSchemaType.KEYWORD),
        ("collection", PayloadSchemaType.KEYWORD),
        ("file_name", PayloadSchemaType.KEYWORD),
        ("folder_name", PayloadSchemaType.KEYWORD),
        ("page_number", PayloadSchemaType.INTEGER),
    ]
    
    for field_name, schema_type in indexes:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=schema_type
            )
        except Exception as e:
            logger.warning(f"Could not create index for {field_name}: {e}")
    
    # Text index for keyword search
    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name="text",
            field_schema=TextIndexParams(
                type="text",
                tokenizer=TokenizerType.WORD,
                min_token_len=2,
                max_token_len=20,
            )
        )
    except Exception as e:
        logger.warning(f"Could not create text index: {e}")


# ============================================================================
# EMBEDDING & UPSERTING
# ============================================================================

def embed_and_upsert(
    chunks: List[Dict[str, Any]],
    file_metadata: Dict[str, Any],
    embeddings_model: AzureOpenAIEmbeddings,
    client: QdrantClient,
    collection_name: str
) -> Dict[str, Any]:
    """
    Generate embeddings and upsert chunks to Qdrant
    
    Args:
        chunks: List of chunk dictionaries
        file_metadata: File-level metadata
        embeddings_model: Embeddings model
        client: Qdrant client
        collection_name: Target collection
        
    Returns:
        Statistics dictionary
    """
    doc_id = file_metadata['doc_id']
    
    # Check if document already exists
    existing = client.scroll(
        collection_name=collection_name,
        scroll_filter={"must": [{"key": "doc_id", "match": {"value": doc_id}}]},
        limit=1
    )
    
    if existing[0]:
        logger.warning(f"Document already exists, deleting old version: {file_metadata['file_name']}")
        client.delete(
            collection_name=collection_name,
            points_selector={
                "filter": {"must": [{"key": "doc_id", "match": {"value": doc_id}}]}
            }
        )
    
    # Generate embeddings in batches
    logger.info(f"Generating embeddings for {len(chunks)} chunks...")
    texts = [chunk['text'] for chunk in chunks]
    
    all_embeddings = []
    for i in range(0, len(texts), Config.BATCH_SIZE):
        batch_texts = texts[i:i + Config.BATCH_SIZE]
        batch_embeddings = embeddings_model.embed_documents(batch_texts)
        all_embeddings.extend(batch_embeddings)
        logger.info(f"  Embedded {min(i + Config.BATCH_SIZE, len(texts))}/{len(texts)}")
    
    # Create points
    points = []
    for idx, (chunk, embedding) in enumerate(zip(chunks, all_embeddings)):
        point_id = abs(hash(f"{doc_id}_{chunk['id']}")) % (10**15)
        
        payload = {
            **file_metadata,
            'text': chunk['text'],
            'chunk_index': idx,
            'chunk_total': len(chunks),
            'is_leaf': chunk['is_leaf'],
            'parent_id': chunk['parent_id'],
            'parent_text': chunk['parent_text'],
            'page_number': chunk['page_number']
        }
        
        points.append(PointStruct(
            id=point_id,
            vector=embedding,
            payload=payload
        ))
    
    # Upsert in batches
    logger.info(f"Upserting {len(points)} points to collection '{collection_name}'...")
    for i in range(0, len(points), Config.BATCH_SIZE):
        batch = points[i:i + Config.BATCH_SIZE]
        client.upsert(collection_name=collection_name, points=batch)
        logger.info(f"  Upserted {min(i + Config.BATCH_SIZE, len(points))}/{len(points)}")
    
    return {
        'doc_id': doc_id,
        'file_name': file_metadata['file_name'],
        'chunks_created': len(chunks),
        'chunks_upserted': len(points),
        'status': 'success'
    }


# ============================================================================
# PROCESSING PIPELINE
# ============================================================================

def process_file(
    file_path: str,
    collection_name: str,
    embeddings_model: AzureOpenAIEmbeddings,
    client: QdrantClient
) -> Dict[str, Any]:
    """
    Complete pipeline: extract -> chunk -> embed -> upsert
    
    Args:
        file_path: Path to document file
        collection_name: Target collection
        embeddings_model: Embeddings model
        client: Qdrant client
        
    Returns:
        Processing statistics
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing: {file_path}")
    logger.info(f"{'='*80}")
    
    # Extract text
    pages = extract_text_from_file(file_path)
    if not pages:
        return {
            'file_name': Path(file_path).name,
            'status': 'failed',
            'error': 'Text extraction failed'
        }
    
    # Generate metadata
    file_metadata = generate_file_metadata(file_path, collection_name)
    
    # Chunk
    chunks = hierarchical_chunk_pages(pages, file_metadata)
    if not chunks:
        return {
            'file_name': file_metadata['file_name'],
            'status': 'failed',
            'error': 'Chunking failed'
        }
    
    # Embed and upsert
    stats = embed_and_upsert(chunks, file_metadata, embeddings_model, client, collection_name)
    
    logger.info(f"✅ Successfully processed {file_metadata['file_name']}")
    logger.info(f"   Chunks: {stats['chunks_upserted']}")
    
    return stats


def process_collection(
    data_root: str,
    collection_name: str,
    embeddings_model: AzureOpenAIEmbeddings,
    client: QdrantClient
) -> Dict[str, Any]:
    """
    Process all files in a collection folder
    
    Args:
        data_root: Root data directory
        collection_name: Collection name (subfolder under data_root)
        embeddings_model: Embeddings model
        client: Qdrant client
        
    Returns:
        Overall statistics
    """
    collection_path = Path(data_root) / collection_name
    
    if not collection_path.exists():
        logger.error(f"Collection folder not found: {collection_path}")
        return {'status': 'failed', 'error': 'Collection folder not found'}
    
    # Find all supported files
    supported_extensions = ['.pdf', '.docx', '.doc']
    files = []
    for ext in supported_extensions:
        files.extend(collection_path.glob(f"**/*{ext}"))
    
    if not files:
        logger.warning(f"No supported files found in {collection_path}")
        return {'status': 'success', 'files_processed': 0, 'total_chunks': 0}
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing Collection: {collection_name}")
    logger.info(f"{'='*80}")
    logger.info(f"Found {len(files)} files")
    
    # Setup collection
    setup_collection(client, collection_name, recreate=False)
    
    # Process each file
    results = []
    total_chunks = 0
    for file_path in files:
        result = process_file(str(file_path), collection_name, embeddings_model, client)
        results.append(result)
        if result.get('status') == 'success':
            total_chunks += result.get('chunks_upserted', 0)
    
    # Summary
    successful = sum(1 for r in results if r.get('status') == 'success')
    failed = len(results) - successful
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Collection Processing Complete")
    logger.info(f"{'='*80}")
    logger.info(f"Files processed: {successful}/{len(files)}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Total chunks: {total_chunks}")
    
    # Save report to state
    save_ingestion_report(collection_name, results)
    
    return {
        'collection': collection_name,
        'status': 'success',
        'files_processed': successful,
        'files_failed': failed,
        'total_chunks': total_chunks,
        'results': results
    }


def process_all_collections(
    data_root: str,
    embeddings_model: AzureOpenAIEmbeddings,
    client: QdrantClient
) -> Dict[str, Any]:
    """
    Process all collection folders under data_root
    
    Args:
        data_root: Root data directory
        embeddings_model: Embeddings model
        client: Qdrant client
        
    Returns:
        Overall statistics
    """
    data_path = Path(data_root)
    
    if not data_path.exists():
        logger.error(f"Data root not found: {data_root}")
        return {'status': 'failed', 'error': 'Data root not found'}
    
    # Find all subdirectories (collections)
    collections = [d.name for d in data_path.iterdir() if d.is_dir()]
    
    if not collections:
        logger.warning(f"No collection folders found in {data_root}")
        return {'status': 'success', 'collections_processed': 0}
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing All Collections")
    logger.info(f"{'='*80}")
    logger.info(f"Found {len(collections)} collections: {', '.join(collections)}")
    
    # Process each collection
    results = []
    for collection in collections:
        result = process_collection(data_root, collection, embeddings_model, client)
        results.append(result)
    
    # Summary
    successful = sum(1 for r in results if r.get('status') == 'success')
    total_files = sum(r.get('files_processed', 0) for r in results)
    total_chunks = sum(r.get('total_chunks', 0) for r in results)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"All Collections Processed")
    logger.info(f"{'='*80}")
    logger.info(f"Collections: {successful}/{len(collections)}")
    logger.info(f"Total files: {total_files}")
    logger.info(f"Total chunks: {total_chunks}")
    
    return {
        'status': 'success',
        'collections_processed': successful,
        'total_files': total_files,
        'total_chunks': total_chunks,
        'results': results
    }


# ============================================================================
# REPORTING
# ============================================================================

def save_ingestion_report(collection_name: str, results: List[Dict[str, Any]]) -> None:
    """
    Save ingestion report to state directory
    
    Args:
        collection_name: Collection name
        results: List of file processing results
    """
    state_dir = Path(Config.STATE_DIR)
    state_dir.mkdir(parents=True, exist_ok=True)
    
    report_file = state_dir / f"ingestion_{collection_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    report = {
        'collection': collection_name,
        'timestamp': datetime.utcnow().isoformat(),
        'files_processed': len([r for r in results if r.get('status') == 'success']),
        'files_failed': len([r for r in results if r.get('status') == 'failed']),
        'total_chunks': sum(r.get('chunks_upserted', 0) for r in results),
        'results': results
    }
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Report saved: {report_file}")


# ============================================================================
# CLI
# ============================================================================

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Ingest documents: extract text → chunk → embed → upsert to Qdrant',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single collection
  python scripts/index_text.py --data-root /data --collection hydrogen_books
  
  # Process all collections
  python scripts/index_text.py --data-root /data --all-collections
        """
    )
    
    parser.add_argument(
        '--data-root',
        type=str,
        required=True,
        help='Root directory containing collection folders'
    )
    
    parser.add_argument(
        '--collection',
        type=str,
        help='Single collection name to process (subfolder under data-root)'
    )
    
    parser.add_argument(
        '--all-collections',
        action='store_true',
        help='Process all collection folders under data-root'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.collection and not args.all_collections:
        parser.error("Either --collection or --all-collections must be specified")
    
    if args.collection and args.all_collections:
        parser.error("Cannot specify both --collection and --all-collections")
    
    # Initialize clients
    try:
        embeddings_model = get_embeddings_model()
        qdrant_client = get_qdrant_client()
    except Exception as e:
        logger.error(f"Failed to initialize clients: {e}")
        sys.exit(1)
    
    # Process
    try:
        if args.all_collections:
            result = process_all_collections(args.data_root, embeddings_model, qdrant_client)
        else:
            result = process_collection(args.data_root, args.collection, embeddings_model, qdrant_client)
        
        if result.get('status') == 'success':
            logger.info("\n✅ Ingestion complete!")
            sys.exit(0)
        else:
            logger.error(f"\n❌ Ingestion failed: {result.get('error')}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()