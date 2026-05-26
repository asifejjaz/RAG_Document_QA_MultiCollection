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
import time
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

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from scripts import embed_config

# Minimum total characters across PDF pages to consider extraction successful (scanned PDFs ~0)
MIN_EXTRACTED_TEXT_CHARS = int(os.getenv("MIN_EXTRACTED_TEXT_CHARS", "40"))
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    PayloadSchemaType, TextIndexParams,
    TokenizerType, OptimizersConfigDiff,
    HnswConfigDiff, ScalarQuantization,
    ScalarQuantizationConfig, ScalarType,
    Filter, FieldCondition, MatchValue,
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
    
    # Qdrant: prefer QDRANT_URL (e.g. host override) then VECTOR_DB_URL (.env / Docker)
    QDRANT_URL = os.getenv("QDRANT_URL") or os.getenv("VECTOR_DB_URL") or "http://localhost:6333"
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_TIMEOUT = 60
    QDRANT_COLLECTION_PREFIX = os.getenv("QDRANT_COLLECTION_PREFIX", "").strip()
    
    # Azure OpenAI
    AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")
    EMBEDDING_MODEL = "text-embedding-ada-002"
    EMBEDDING_DIMENSION = 1536
    
    # Chunking (from env for M1; fallback to defaults)
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
    PARENT_CHUNK_SIZE = CHUNK_SIZE
    CHILD_CHUNK_SIZE = min(500, CHUNK_SIZE)
    
    # Processing
    BATCH_SIZE = 100  # Qdrant upsert batch size
    
    # Embedding rate-limit (Azure TPM/RPM); use smaller batches + delay for large files
    EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "20"))
    EMBEDDING_BATCH_DELAY_SEC = float(os.getenv("EMBEDDING_BATCH_DELAY_SEC", "1.0"))
    EMBEDDING_MAX_RETRIES = int(os.getenv("EMBEDDING_MAX_RETRIES", "5"))
    EMBEDDING_RETRY_BASE_DELAY_SEC = float(os.getenv("EMBEDDING_RETRY_BASE_DELAY_SEC", "10.0"))
    
    # State directory (for reports); prefer STATE_ROOT then STATE_DIR
    STATE_DIR = os.getenv("STATE_ROOT") or os.getenv("STATE_DIR", "./state")


# ============================================================================
# QDRANT CLIENT INITIALIZATION
# ============================================================================

def get_qdrant_client() -> QdrantClient:
    """Initialize and return Qdrant client"""
    api_key = os.getenv("QDRANT_API_KEY")
    url = Config.QDRANT_URL
    if url and url.startswith("https://"):
        return QdrantClient(
            url=url,
            api_key=api_key,
            timeout=Config.QDRANT_TIMEOUT
        )
    else:
        return QdrantClient(
            url=url,
            port=Config.QDRANT_PORT,
            api_key=api_key,
            timeout=Config.QDRANT_TIMEOUT
        )


def delete_qdrant_collection(client: QdrantClient, collection_name: str) -> None:
    """
    Permanently delete a Qdrant collection and all its vectors.
    Does not remove files from disk (e.g. under data/).
    """
    if not collection_name or not str(collection_name).strip():
        raise ValueError("collection_name is required")
    client.delete_collection(collection_name=collection_name.strip())


def get_collection_names_for_dimension(client: QdrantClient, dimension: int) -> List[str]:
    """Return collection names whose vector size matches the given dimension (for current embedding).
    If get_collection() fails (e.g. client/server schema mismatch), returns all collection names
    so the UI folder list is not empty."""
    names = []
    fallback_names = []
    for c in client.get_collections().collections:
        try:
            info = client.get_collection(c.name)
            size = info.config.params.vectors.size
            if size == dimension:
                names.append(c.name)
        except Exception:
            # Schema mismatch or other error: collect for fallback so UI still shows folders
            fallback_names.append(c.name)
    if names:
        return sorted(names)
    # When no dimension-matched names (e.g. get_collection failed for all), show all collections
    return sorted(fallback_names) if fallback_names else []


def get_collection_vector_size(client: QdrantClient, collection_name: str) -> Optional[int]:
    """Return configured vector size for a collection, or None if unavailable."""
    try:
        info = client.get_collection(collection_name)
        return int(info.config.params.vectors.size)
    except Exception:
        return None


def _qdrant_collection_name(collection_name: str) -> str:
    """Return Qdrant collection name, with optional prefix from config."""
    prefix = (Config.QDRANT_COLLECTION_PREFIX or "").strip()
    if not prefix:
        return collection_name
    return f"{prefix.rstrip('_')}_{collection_name}" if prefix else collection_name


def get_embeddings_model(embedding_id: Optional[str] = None):
    """Initialize and return embeddings model (from embed_config; supports Azure and bge-m3)."""
    return embed_config.get_embeddings_model(embedding_id)


def _is_rate_limit_error(exc: BaseException) -> bool:
    """True if the exception indicates Azure/OpenAI rate limit (429)."""
    msg = str(exc).lower()
    if "429" in msg or "rate limit" in msg or "too many requests" in msg:
        return True
    if getattr(exc, "status_code", None) == 429:
        return True
    if getattr(exc, "response", None) is not None:
        status = getattr(exc.response, "status_code", None)
        if status == 429:
            return True
    return False


def _get_retry_after_sec(exc: BaseException, attempt: int) -> float:
    """Preferred wait time from Retry-After header, else exponential backoff."""
    # Prefer server hint (Retry-After)
    retry_after = getattr(exc, "retry_after", None)
    if retry_after is not None:
        try:
            return float(retry_after)
        except (TypeError, ValueError):
            pass
    response = getattr(exc, "response", None)
    if response is not None and hasattr(response, "headers"):
        ra = response.headers.get("retry-after") or response.headers.get("Retry-After")
        if ra is not None:
            try:
                return float(ra)
            except (TypeError, ValueError):
                pass
    # Exponential backoff
    return Config.EMBEDDING_RETRY_BASE_DELAY_SEC * (2 ** min(attempt, 4))


def embed_batch_with_retry(
    embeddings_model: AzureOpenAIEmbeddings,
    batch_texts: List[str],
) -> List[List[float]]:
    """
    Call embed_documents with retries on 429 (rate limit).
    Uses Retry-After when present, otherwise exponential backoff.
    """
    last_exc = None
    for attempt in range(Config.EMBEDDING_MAX_RETRIES):
        try:
            return embeddings_model.embed_documents(batch_texts)
        except Exception as e:
            last_exc = e
            if _is_rate_limit_error(e) and attempt < Config.EMBEDDING_MAX_RETRIES - 1:
                wait_sec = _get_retry_after_sec(e, attempt)
                logger.warning(
                    "Embedding rate limited (429), retry %s/%s after %.1fs: %s",
                    attempt + 1,
                    Config.EMBEDDING_MAX_RETRIES,
                    wait_sec,
                    str(e)[:200],
                )
                time.sleep(wait_sec)
            else:
                raise
    if last_exc is not None:
        raise last_exc
    return []  # unreachable


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


def total_extracted_text_length(pages: List[Dict[str, Any]]) -> int:
    """Sum of stripped text lengths across pages (detect empty / scanned PDFs)."""
    return sum(len((p.get("text") or "").strip()) for p in pages)


# ============================================================================
# DOCUMENT ID & METADATA GENERATION
# ============================================================================

def generate_doc_id(stable_key: str) -> str:
    """Generate unique document ID from a stable logical key (path or collection|name|size)."""
    return hashlib.md5(stable_key.encode()).hexdigest()


def generate_file_metadata(
    file_path: str,
    collection_name: str,
    logical_file_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate comprehensive metadata for a file
    
    Args:
        file_path: Path to the document
        collection_name: Name of the target collection
        
    Returns:
        Metadata dictionary
    """
    file_path = Path(file_path)
    display_name = logical_file_name or file_path.name
    size = file_path.stat().st_size if file_path.exists() else 0
    
    # Stable id: for uploads use collection+name+size; for direct file paths include content hash
    if logical_file_name:
        id_key = f"{collection_name}|{display_name}|{size}"
    else:
        # Include file content hash to remain stable across renames/moves
        try:
            with open(file_path, 'rb') as f:
                content_hash = hashlib.md5()
                for chunk in iter(lambda: f.read(8192), b''):
                    content_hash.update(chunk)
                content_md5 = content_hash.hexdigest()
            id_key = f"{file_path.resolve()}|{size}|{content_md5}"
        except Exception:
            # Fallback to path-only if file unreadable
            id_key = str(file_path.resolve())
    
    stem = Path(display_name).stem
    ext = Path(display_name).suffix

    return {
        'doc_id': generate_doc_id(id_key),
        'file_name': display_name,
        'file_stem': stem,
        'file_extension': ext,
        'file_size': size,
        'collection': collection_name,  # logical folder name; Qdrant upsert overwrites with full collection id
        'folder_name': file_path.parent.name,
        'full_path': str(file_path.absolute()),
        'ingest_source_path': f"{collection_name}/{display_name}",
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

    def _node_id(n):
        """Get node id from Node or RelatedNodeInfo."""
        return getattr(n, "id_", None) or getattr(n, "node_id", None)

    # Build parent text lookup
    parent_text_map = {}
    for node in nodes:
        if not node.parent_node:  # This is a parent
            parent_text_map[_node_id(node)] = node.text

    # Process nodes and match to pages
    chunks = []
    for node in nodes:
        # Determine which page this chunk belongs to
        page_number = _find_page_for_chunk(node.text, pages)
        parent_id = _node_id(node.parent_node) if node.parent_node else None
        chunk_data = {
            "id": _node_id(node),
            "text": node.text,
            "is_leaf": bool(node.parent_node),
            "parent_id": parent_id,
            "parent_text": parent_text_map.get(parent_id) if parent_id else None,
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
    recreate: bool = False,
    vector_size: Optional[int] = None,
) -> None:
    """
    Setup Qdrant collection with optimized configuration.
    Vector size must match the embedding model dimension (from embed_config).
    """
    size = vector_size if vector_size is not None else Config.EMBEDDING_DIMENSION
    collections = client.get_collections().collections
    exists = any(c.name == collection_name for c in collections)
    
    if exists:
        if recreate:
            logger.info(f"Deleting existing collection: {collection_name}")
            client.delete_collection(collection_name)
        else:
            logger.info(f"Collection '{collection_name}' already exists")
            return
    
    logger.info(f"Creating collection: {collection_name} (vector_size={size})")
    
    # Create collection with vector configuration
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=size,
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
        ("source_path", PayloadSchemaType.KEYWORD),
        ("file_name", PayloadSchemaType.KEYWORD),
        ("folder_name", PayloadSchemaType.KEYWORD),
        ("page_number", PayloadSchemaType.INTEGER),
        ("page_start", PayloadSchemaType.INTEGER),
        ("page_end", PayloadSchemaType.INTEGER),
        ("chunk_id", PayloadSchemaType.KEYWORD),
        ("is_leaf", PayloadSchemaType.BOOL),
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
            points_selector=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]),
        )
    
    # Generate embeddings in batches (rate-limit aware: smaller batches + delay + retry)
    batch_size = Config.EMBEDDING_BATCH_SIZE
    delay_sec = Config.EMBEDDING_BATCH_DELAY_SEC
    logger.info(
        "Generating embeddings for %s chunks (batch_size=%s, delay=%.1fs)...",
        len(chunks), batch_size, delay_sec,
    )
    texts = [chunk['text'] for chunk in chunks]
    
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = embed_batch_with_retry(embeddings_model, batch_texts)
        all_embeddings.extend(batch_embeddings)
        logger.info(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)}")
        if delay_sec > 0 and i + batch_size < len(texts):
            time.sleep(delay_sec)
    
    # Create points (required payload keys: collection, source_path, doc_id, page_start, page_end, chunk_index, chunk_id, text)
    points = []
    for idx, (chunk, embedding) in enumerate(zip(chunks, all_embeddings)):
        point_id = abs(hash(f"{doc_id}_{chunk['id']}")) % (10**15)
        page_num = chunk['page_number']
        page_start = page_end = page_num
        chunk_id = f"{doc_id}:{page_start}:{idx}"
        source_path = (
            file_metadata.get('ingest_source_path')
            or file_metadata.get('full_path')
            or file_metadata.get('file_name', '')
        )
        payload = {
            **file_metadata,
            'collection': collection_name,
            'source_path': source_path,
            'doc_id': doc_id,
            'page_start': page_start,
            'page_end': page_end,
            'chunk_index': idx,
            'chunk_id': chunk_id,
            'text': chunk['text'],
            'chunk_total': len(chunks),
            'is_leaf': chunk['is_leaf'],
            'parent_id': chunk['parent_id'],
            'parent_text': chunk['parent_text'],
            'page_number': page_num,
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
    embeddings_model,
    client: QdrantClient,
    embedding_id: Optional[str] = None,
    logical_file_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Complete pipeline: extract -> chunk -> embed -> upsert.
    embedding_id is used to set collection vector size (from embed_config); default from env EMBED_MODEL.
    logical_file_name: original upload filename (preserved in metadata when ingesting from a temp path).
    """
    embedding_id = embedding_id or os.getenv("EMBED_MODEL")
    vector_size = embed_config.get_embedding_dimension(embedding_id)
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing: {file_path}")
    logger.info(f"{'='*80}")
    
    qdrant_name = _qdrant_collection_name(collection_name)
    # Ensure collection exists (creates if missing); use dimension for selected embedding
    setup_collection(client, qdrant_name, recreate=False, vector_size=vector_size)
    
    # Extract text
    pages = extract_text_from_file(file_path)
    display = logical_file_name or Path(file_path).name
    if not pages:
        return {
            'file_name': display,
            'status': 'failed',
            'error': 'Text extraction failed',
            'ingest_flag': 'extraction_error',
        }

    if total_extracted_text_length(pages) < MIN_EXTRACTED_TEXT_CHARS:
        return {
            'file_name': display,
            'status': 'skipped',
            'error': (
                'No extractable text (likely scanned PDF or image-only). '
                'Enable OCR (e.g. Tesseract) or Azure Document Intelligence, or supply a text-based PDF.'
            ),
            'ingest_flag': 'zero_text_pdf',
        }
    
    # Generate metadata
    file_metadata = generate_file_metadata(file_path, collection_name, logical_file_name=logical_file_name)
    
    # Chunk
    chunks = hierarchical_chunk_pages(pages, file_metadata)
    if not chunks:
        return {
            'file_name': file_metadata['file_name'],
            'status': 'failed',
            'error': 'Chunking failed',
            'ingest_flag': 'chunking_failed',
        }
    
    # Embed and upsert
    stats = embed_and_upsert(chunks, file_metadata, embeddings_model, client, qdrant_name)
    
    logger.info(f"[SUCCESS] Successfully processed {file_metadata['file_name']}")
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
    
    qdrant_name = _qdrant_collection_name(collection_name)
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing Collection: {collection_name}")
    logger.info(f"{'='*80}")
    logger.info(f"Found {len(files)} files")
    
    embedding_id = os.getenv("EMBED_MODEL")
    vector_size = embed_config.get_embedding_dimension(embedding_id)
    setup_collection(client, qdrant_name, recreate=False, vector_size=vector_size)
    
    # Process each file (process_file will apply prefix to get qdrant name)
    results = []
    total_chunks = 0
    for file_path in files:
        result = process_file(str(file_path), collection_name, embeddings_model, client)
        results.append(result)
        if result.get('status') == 'success':
            total_chunks += result.get('chunks_upserted', 0)
    
    # Summary
    successful = sum(1 for r in results if r.get('status') == 'success')
    skipped = sum(1 for r in results if r.get('status') == 'skipped')
    failed = sum(1 for r in results if r.get('status') == 'failed')
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Collection Processing Complete")
    logger.info(f"{'='*80}")
    logger.info(f"Files processed: {successful}/{len(files)}")
    logger.info(f"Skipped (no text): {skipped}, Failed: {failed}")
    logger.info(f"Total chunks: {total_chunks}")
    
    # Save report to state
    save_ingestion_report(collection_name, results)
    
    return {
        'collection': collection_name,
        'status': 'success',
        'files_processed': successful,
        'files_skipped': skipped,
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
        'files_skipped': len([r for r in results if r.get('status') == 'skipped']),
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
        description='Ingest documents: extract text -> chunk -> embed -> upsert to Qdrant',
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
            logger.info("\n[SUCCESS] Ingestion complete!")
            sys.exit(0)
        else:
            logger.error(f"\n[FAILED] Ingestion failed: {result.get('error')}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        err_str = str(e).lower()
        if "refused" in err_str or "10061" in err_str or "connect" in err_str:
            logger.error(
                "Connection refused: Qdrant may not be running. Start it with:\n"
                "  docker compose -f docker_compose.yaml up -d qdrant\n"
                "If running on host (not in Docker), set: $env:QDRANT_URL = \"http://localhost:6333\""
            )
        sys.exit(1)


if __name__ == "__main__":
    main()