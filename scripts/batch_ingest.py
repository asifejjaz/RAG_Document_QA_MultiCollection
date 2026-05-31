#!/usr/bin/env python3
"""
Fast Batch Ingestion Script for Eva Research AI (Cloud Qdrant)
Stages files and uploads them to Cloud Qdrant using PyMuPDF and RecursiveCharacterTextSplitter.
Extremely fast, bypasses heavy CPU docling layout processing.
"""
import os
import sys
import time
import shutil
import hashlib
import logging
from pathlib import Path
from datetime import datetime

# Setup path to import backend modules
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("fast_ingest")

# Load environment
env_path = _project_root / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                val = v.split("#")[0].strip()
                os.environ[k.strip()] = val

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, HnswConfigDiff,
    ScalarQuantization, ScalarQuantizationConfig, ScalarType,
    OptimizersConfigDiff, PayloadSchemaType, TextIndexParams,
    TokenizerType, Filter, FieldCondition, MatchValue
)
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from scripts.embed_config import get_embeddings_model, get_embedding_dimension

HOST_BOOKS_DIR = Path(r"C:\Users\HP\Documents\Python Scripts\Python\ai_rag\Books\Books")
WORKSPACE_DATA_DIR = _project_root / "data"

COLLECTIONS_MAPPING = {
    "green_hydrogen": [
        "Green-Hydrogen-The-Race-to-Success.pdf",
        "Cheat Sheet Hydrogen (1).pdf",
        "Green Hydrogen the Race to Success-Members.pdf",
        "US National Clean Hydrogen Strategy Roadmap Full Report.pdf"
    ],
    "hydrogen_bunkering": [
        "Hydrogen Bunkering at Ports by Eliseo Curcio.pdf",
        "Hydrogen ports 2.pdf",
        "hydrogen bunkering.pdf"
    ],
    "ammonia_fuel": [
        "Ammonias-Double-Edged-Sword-Clean-Energy-or-Catastrophic-Risk.pdf",
        "Ammonia_ship_fuel_2020-11_web.pdf",
        "Risk assessment of ammonia bunkering.pdf"
    ]
}

def stage_files():
    """Copy files from host directory to local workspace directories."""
    logger.info("Staging book files into workspace...")
    WORKSPACE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    for folder, files in COLLECTIONS_MAPPING.items():
        folder_path = WORKSPACE_DATA_DIR / folder
        folder_path.mkdir(parents=True, exist_ok=True)
        
        for file_name in files:
            src_file = HOST_BOOKS_DIR / file_name
            dst_file = folder_path / file_name
            
            if not src_file.exists():
                logger.error(f"Source file not found: {src_file}")
                # Try finding case-insensitively if exact match fails
                matches = list(HOST_BOOKS_DIR.glob(f"*{file_name.split('.')[0]}*"))
                if matches:
                    src_file = matches[0]
                    dst_file = folder_path / matches[0].name
                    logger.info(f"Using case-insensitive match: {src_file.name}")
                else:
                    logger.error(f"Cannot find match for: {file_name}")
                    continue
                    
            if not dst_file.exists():
                logger.info(f"Copying {src_file.name} -> data/{folder}/")
                shutil.copy2(src_file, dst_file)
            else:
                logger.info(f"File already staged: data/{folder}/{dst_file.name}")

def setup_collection(client: QdrantClient, full_name: str, dimension: int):
    """Setup Qdrant collection with appropriate parameters and payload indexes."""
    exists = False
    try:
        collections = client.get_collections().collections
        exists = any(c.name == full_name for c in collections)
    except Exception as e:
        logger.warning(f"Could not list collections: {e}")
        
    if exists:
        logger.info(f"Collection '{full_name}' already exists.")
        return
        
    logger.info(f"Creating collection '{full_name}' (vector_size={dimension})")
    client.create_collection(
        collection_name=full_name,
        vectors_config=VectorParams(
            size=dimension,
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
        ("doc_type", PayloadSchemaType.KEYWORD),
        ("asset_path", PayloadSchemaType.KEYWORD),
    ]
    for field, schema in indexes:
        client.create_payload_index(collection_name=full_name, field_name=field, field_schema=schema)
        
    client.create_payload_index(
        collection_name=full_name,
        field_name="text",
        field_schema=TextIndexParams(
            type="text",
            tokenizer=TokenizerType.WORD,
            min_token_len=2,
            max_token_len=20,
        )
    )

def generate_file_metadata(file_path: Path, collection_name: str) -> dict:
    """Generate metadata for document mapping."""
    size = file_path.stat().st_size
    try:
        with open(file_path, 'rb') as f:
            content_hash = hashlib.md5()
            for chunk in iter(lambda: f.read(8192), b''):
                content_hash.update(chunk)
            content_md5 = content_hash.hexdigest()
        id_key = f"{file_path.resolve()}|{size}|{content_md5}"
    except Exception:
        id_key = str(file_path.resolve())

    doc_id = hashlib.md5(id_key.encode()).hexdigest()
    return {
        'doc_id': doc_id,
        'file_name': file_path.name,
        'file_stem': file_path.stem,
        'file_extension': file_path.suffix,
        'file_size': size,
        'collection': collection_name,
        'folder_name': file_path.parent.name,
        'full_path': str(file_path.absolute()),
        'ingest_source_path': f"{collection_name}/{file_path.name}",
        'created_at': datetime.utcnow().isoformat()
    }

def ingest_collections():
    """Run ingestion for all staged files into the Qdrant collections."""
    logger.info("Initializing connection to Cloud Qdrant...")
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    collection_prefix = os.getenv("QDRANT_COLLECTION_PREFIX", "").strip()
    
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=60)
    
    # Load Voyage Embeddings Model
    embed_model = get_embeddings_model()
    dimension = get_embedding_dimension()
    logger.info(f"Using embedding dimension: {dimension}")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150
    )
    
    for folder, files in COLLECTIONS_MAPPING.items():
        folder_path = WORKSPACE_DATA_DIR / folder
        full_collection_name = f"{collection_prefix.rstrip('_')}_{folder}" if collection_prefix else folder
        
        logger.info(f"\n=============================================================")
        logger.info(f"Targeting Cloud Collection: {full_collection_name}")
        logger.info(f"=============================================================")
        
        setup_collection(client, full_collection_name, dimension)
        
        for item in folder_path.iterdir():
            if item.is_file() and item.suffix.lower() == '.pdf':
                logger.info(f"Loading {item.name} with PyMuPDF...")
                
                # Check if document already exists
                file_metadata = generate_file_metadata(item, folder)
                doc_id = file_metadata['doc_id']
                
                # Scroll once to see if it exists
                res = client.scroll(
                    collection_name=full_collection_name,
                    scroll_filter=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]),
                    limit=1,
                    with_payload=False
                )
                if len(res[0]) > 0:
                    logger.info(f"  Document {item.name} already exists in collection. Skipping.")
                    continue
                
                try:
                    loader = PyMuPDFLoader(str(item))
                    docs = loader.load()
                except Exception as e:
                    logger.error(f"  Failed to load {item.name}: {e}")
                    continue
                    
                logger.info(f"  Extracted {len(docs)} pages.")
                
                # Perform chunking page-by-page to preserve accurate page number citations
                chunks = []
                for idx, page_doc in enumerate(docs):
                    page_num = idx + 1
                    page_text = page_doc.page_content
                    if not page_text.strip():
                        continue
                    
                    page_chunks = text_splitter.split_text(page_text)
                    for chunk_text in page_chunks:
                        chunks.append({
                            "text": chunk_text,
                            "page_number": page_num
                        })
                
                total_chunks = len(chunks)
                logger.info(f"  Generated {total_chunks} text chunks.")
                
                if total_chunks == 0:
                    logger.warning(f"  No text found in {item.name}. Skipping.")
                    continue
                
                # Embed chunks in batches to prevent Voyage API limits
                batch_size = 20
                logger.info(f"  Embedding and uploading points to Cloud Qdrant...")
                
                points = []
                for batch_idx in range(0, total_chunks, batch_size):
                    chunk_batch = chunks[batch_idx : batch_idx + batch_size]
                    texts = [c["text"] for c in chunk_batch]
                    
                    # Call API to embed
                    try:
                        embeddings_batch = embed_model.embed_documents(texts)
                    except Exception as e:
                        logger.error(f"  Embedding batch failure: {e}")
                        # Exponential backoff retry
                        time.sleep(10)
                        embeddings_batch = embed_model.embed_documents(texts)
                        
                    for idx_in_batch, (chunk, embedding) in enumerate(zip(chunk_batch, embeddings_batch)):
                        global_idx = batch_idx + idx_in_batch
                        point_id = abs(hash(f"{doc_id}_{global_idx}")) % (10**15)
                        page_num = chunk["page_number"]
                        chunk_id = f"{doc_id}:{page_num}:{global_idx}"
                        
                        payload = {
                            **file_metadata,
                            'collection': full_collection_name,
                            'source_path': file_metadata["ingest_source_path"],
                            'doc_id': doc_id,
                            'page_start': page_num,
                            'page_end': page_num,
                            'chunk_index': global_idx,
                            'chunk_id': chunk_id,
                            'text': chunk['text'],
                            'chunk_total': total_chunks,
                            'is_leaf': True,
                            'parent_id': None,
                            'parent_text': None,
                            'page_number': page_num,
                            'doc_type': "text",
                            'asset_path': None
                        }
                        points.append(PointStruct(id=point_id, vector=embedding, payload=payload))
                    
                    # Pause to stay within Voyage rate limits
                    time.sleep(1.5)
                
                # Upload points in batches of 100
                logger.info(f"  Uploading {len(points)} points...")
                for i in range(0, len(points), 100):
                    sub_batch = points[i : i + 100]
                    client.upsert(collection_name=full_collection_name, points=sub_batch)
                    
                logger.info(f"  [SUCCESS] {item.name} fully ingested and uploaded.")
                time.sleep(1.0)

if __name__ == "__main__":
    stage_files()
    ingest_collections()
