import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    PayloadSchemaType, TextIndexParams,
    TokenizerType, OptimizersConfigDiff,
    HnswConfigDiff, ScalarQuantization,
    ScalarQuantizationConfig, ScalarType,
    Filter, FieldCondition, MatchValue,
)
from backend.app.config import settings

logger = logging.getLogger(__name__)

class VectorDatabaseService:
    def __init__(self):
        if settings.qdrant_url and settings.qdrant_url.startswith("https://"):
            self.client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
                timeout=60
            )
        else:
            self.client = QdrantClient(
                url=settings.qdrant_url,
                port=settings.qdrant_port,
                api_key=settings.qdrant_api_key,
                timeout=60
            )
        self._prefix = settings.qdrant_collection_prefix or ""

    def get_full_collection_name(self, name: str) -> str:
        """Helper to prefix short collection names."""
        clean_name = name.strip()
        if not self._prefix:
            return clean_name
        if clean_name.startswith(self._prefix):
            return clean_name
        return f"{self._prefix.rstrip('_')}_{clean_name}"

    def get_short_collection_name(self, full_name: str) -> str:
        """Helper to get original display name from prefixed Qdrant collection name."""
        if self._prefix and full_name.startswith(self._prefix):
            return full_name[len(self._prefix):].lstrip("_")
        return full_name

    def setup_collection(self, collection_name: str, vector_size: int, recreate: bool = False) -> str:
        full_name = self.get_full_collection_name(collection_name)
        exists = False
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == full_name for c in collections)
        except Exception as e:
            logger.warning("Could not list collections, setting up: %s", e)
        
        if exists:
            if recreate:
                logger.info("Deleting existing collection: %s", full_name)
                self.client.delete_collection(full_name)
            else:
                logger.info("Collection '%s' already exists", full_name)
                return full_name

        logger.info("Creating collection '%s' (vector_size=%d)", full_name, vector_size)
        self.client.create_collection(
            collection_name=full_name,
            vectors_config=VectorParams(
                size=vector_size,
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
        self._create_payload_indexes(full_name)
        return full_name

    def _create_payload_indexes(self, full_name: str) -> None:
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
            try:
                self.client.create_payload_index(
                    collection_name=full_name,
                    field_name=field,
                    field_schema=schema
                )
            except Exception as e:
                logger.warning("Could not create payload index on %s: %s", field, e)

        try:
            self.client.create_payload_index(
                collection_name=full_name,
                field_name="text",
                field_schema=TextIndexParams(
                    type="text",
                    tokenizer=TokenizerType.WORD,
                    min_token_len=2,
                    max_token_len=20,
                )
            )
        except Exception as e:
            logger.warning("Could not create text index: %s", e)

    def delete_collection(self, collection_name: str) -> None:
        full_name = self.get_full_collection_name(collection_name)
        self.client.delete_collection(collection_name=full_name)

    def delete_document_points(self, collection_name: str, doc_id: str) -> None:
        full_name = self.get_full_collection_name(collection_name)
        self.client.delete(
            collection_name=full_name,
            points_selector=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])
        )

    def get_collection_vector_size(self, collection_name: str) -> Optional[int]:
        try:
            full_name = self.get_full_collection_name(collection_name)
            info = self.client.get_collection(full_name)
            return int(info.config.params.vectors.size)
        except Exception:
            return None

    def get_collections_for_dimension(self, dimension: int) -> List[str]:
        names = []
        try:
            for c in self.client.get_collections().collections:
                try:
                    info = self.client.get_collection(c.name)
                    if info.config.params.vectors.size == dimension:
                        # Return short representation for UI consumption
                        names.append(self.get_short_collection_name(c.name))
                except Exception:
                    continue
        except Exception as e:
            logger.error("Error listing collections: %s", e)
        return sorted(names)

    def document_exists(self, collection_name: str, doc_id: str) -> bool:
        full_name = self.get_full_collection_name(collection_name)
        try:
            res = self.client.scroll(
                collection_name=full_name,
                scroll_filter=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]),
                limit=1,
                with_payload=False
            )
            return len(res[0]) > 0
        except Exception:
            return False

    def upsert_points(self, collection_name: str, points: List[PointStruct], batch_size: int = 100) -> None:
        full_name = self.get_full_collection_name(collection_name)
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(collection_name=full_name, points=batch)

    def search_similarity(
        self,
        collection_name: Optional[str],
        query_vector: List[float],
        dimension: int,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search across one collection, or search across all collections matching dimension."""
        leaf_filter = Filter(must=[FieldCondition(key="is_leaf", match=MatchValue(value=True))])
        all_results = []

        try:
            if collection_name:
                # Single collection search
                full_name = self.get_full_collection_name(collection_name)
                # Verify dimension
                vs = self.get_collection_vector_size(full_name)
                if vs is not None and vs != dimension:
                    logger.warning("Dimension mismatch for %s. Expected %d, got %d", full_name, dimension, vs)
                    return []
                
                results = self.client.search(
                    collection_name=full_name,
                    query_vector=query_vector,
                    limit=top_k,
                    query_filter=leaf_filter,
                    with_payload=True
                )
                all_results = [(r, self.get_short_collection_name(full_name)) for r in results]
            else:
                # All collection search
                collections = self.client.get_collections().collections
                for c in collections:
                    vs = self.get_collection_vector_size(c.name)
                    if vs == dimension:
                        try:
                            results = self.client.search(
                                collection_name=c.name,
                                query_vector=query_vector,
                                limit=top_k,
                                query_filter=leaf_filter,
                                with_payload=True
                            )
                            all_results.extend([(r, self.get_short_collection_name(c.name)) for r in results])
                        except Exception as ex:
                            logger.error("Error searching in %s: %s", c.name, ex)
                            continue
                
                # Global sort and limit
                all_results.sort(key=lambda x: x[0].score, reverse=True)
                all_results = all_results[:top_k]

            # Format outputs
            contexts = []
            for result, short_coll in all_results:
                payload = result.payload or {}
                source_path = payload.get("source_path", "")
                file_name = payload.get("file_name") or "Unknown"
                page_num = payload.get("page_number") or payload.get("page_start") or 0
                contexts.append({
                    "text": payload.get("text", ""),
                    "source": source_path or file_name,
                    "page_number": page_num,
                    "score": result.score,
                    "doc_type": payload.get("doc_type", "document"),
                    "folder_name": short_coll,
                    "file_name": file_name,
                    "doc_id": payload.get("doc_id", ""),
                    "asset_path": payload.get("asset_path")
                })
            return contexts
        except Exception as e:
            logger.error("Error during similarity search: %s", e)
            return []

    def get_collection_documents(self, collection_name: str) -> List[Dict[str, Any]]:
        full_name = self.get_full_collection_name(collection_name)
        try:
            scroll_result = self.client.scroll(
                collection_name=full_name,
                limit=10000,
                with_payload=["file_name", "doc_id", "doc_type"]
            )
            docs = {}
            for point in scroll_result[0]:
                payload = point.payload or {}
                file_name = payload.get("file_name", "Unknown")
                doc_id = payload.get("doc_id")
                if not doc_id:
                    continue
                if file_name not in docs:
                    docs[file_name] = {
                        "file_name": file_name,
                        "doc_id": doc_id,
                        "doc_type": payload.get("doc_type", "document"),
                        "chunks": 0
                    }
                docs[file_name]["chunks"] += 1
            return list(docs.values())
        except Exception as e:
            logger.error("Error scrolling folder docs: %s", e)
            return []

    def get_collection_statistics(self, collection_name: str) -> Dict[str, Any]:
        full_name = self.get_full_collection_name(collection_name)
        try:
            scroll_result = self.client.scroll(
                collection_name=full_name,
                limit=10000,
                with_payload=True
            )
            points = scroll_result[0]
            unique_files = set()
            doc_types = {}
            for point in points:
                payload = point.payload or {}
                file_name = payload.get("file_name")
                doc_type = payload.get("doc_type", "document")
                if file_name:
                    unique_files.add(file_name)
                if doc_type:
                    doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
            return {
                "total_chunks": len(points),
                "total_files": len(unique_files),
                "doc_types": doc_types,
                "files": sorted(list(unique_files))
            }
        except Exception as e:
            logger.error("Error getting collection statistics: %s", e)
            return {}

    def get_aggregate_statistics(self) -> Dict[str, Any]:
        try:
            collections = self.client.get_collections().collections
            total_chunks = 0
            total_docs = 0
            unique_docs_all = set()
            for c in collections:
                try:
                    info = self.client.get_collection(c.name)
                    total_chunks += info.points_count
                    scroll = self.client.scroll(
                        collection_name=c.name,
                        limit=10000,
                        with_payload=["doc_id"]
                    )
                    for p in scroll[0]:
                        if p.payload and p.payload.get("doc_id"):
                            unique_docs_all.add(p.payload.get("doc_id"))
                except Exception:
                    continue
            return {
                "total_documents": len(unique_docs_all),
                "total_chunks": total_chunks,
                "total_folders": len(collections)
            }
        except Exception as e:
            logger.error("Error computing aggregate stats: %s", e)
            return {"total_documents": 0, "total_chunks": 0, "total_folders": 0}
