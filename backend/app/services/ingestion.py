import logging
import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from qdrant_client.models import PointStruct
from backend.app.config import settings
from backend.app.services.vector_db import VectorDatabaseService
from backend.app.services.embedding import BaseEmbeddingService

logger = logging.getLogger(__name__)

def caption_image_with_gpt4o_mini(image_path: Path) -> str:
    import base64
    from openai import OpenAI, AzureOpenAI
    from backend.app.config import settings
    
    prompt = (
        "You are a scientific expert analyzing figures/charts from academic literature. "
        "Provide a highly detailed, scholarly description of this image. "
        "For charts/graphs, describe the axes, the data points, legends, curves, and any visible trends or values. "
        "For diagrams, describe the components, flow direction, and structural details. "
        "Avoid introductory text like 'This image shows...' - jump straight into the dense academic description."
    )
    
    try:
        with open(image_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        logger.error("Failed to read image %s for captioning: %s", image_path.name, e)
        return "Image element extracted from document. Failed to read image file."

    # Try Azure OpenAI first if default LLM is azure
    use_azure = settings.default_llm_id == "azure" or not settings.openai_api_key
    
    if use_azure:
        if settings.azure_openai_api_key and settings.azure_openai_endpoint:
            try:
                logger.info("Captioning image using Azure OpenAI (deployment: %s)", settings.azure_openai_deployment)
                client = AzureOpenAI(
                    api_key=settings.azure_openai_api_key,
                    azure_endpoint=settings.azure_openai_endpoint,
                    api_version=settings.azure_openai_api_version
                )
                response = client.chat.completions.create(
                    model=settings.azure_openai_deployment,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=500,
                )
                caption = response.choices[0].message.content.strip()
                logger.info("Successfully captioned image %s with Azure OpenAI: %s...", image_path.name, caption[:50])
                return caption
            except Exception as e:
                logger.warning("Azure OpenAI captioning failed, falling back to OpenAI if configured: %s", e)
                
    # Fallback to OpenAI API
    if settings.openai_api_key:
        try:
            logger.info("Captioning image using OpenAI (gpt-4o-mini)")
            client = OpenAI(api_key=settings.openai_api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500,
            )
            caption = response.choices[0].message.content.strip()
            logger.info("Successfully captioned image %s with OpenAI: %s...", image_path.name, caption[:50])
            return caption
        except Exception as e:
            logger.error("OpenAI captioning failed: %s", e)
            
    # Final fallback if both failed or not configured
    # If we didn't try Azure but have it, try it now
    if not use_azure and settings.azure_openai_api_key and settings.azure_openai_endpoint:
        try:
            logger.info("Captioning image using Azure OpenAI fallback (deployment: %s)", settings.azure_openai_deployment)
            client = AzureOpenAI(
                api_key=settings.azure_openai_api_key,
                azure_endpoint=settings.azure_openai_endpoint,
                api_version=settings.azure_openai_api_version
            )
            response = client.chat.completions.create(
                model=settings.azure_openai_deployment,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500,
            )
            caption = response.choices[0].message.content.strip()
            logger.info("Successfully captioned image %s with Azure OpenAI fallback: %s...", image_path.name, caption[:50])
            return caption
        except Exception as e:
            logger.error("Azure OpenAI fallback captioning failed: %s", e)

    return "Image element extracted from document. Captioning is disabled or failed due to missing API keys."

class BaseDocumentLoader(ABC):
    @abstractmethod
    def load(self, file_path: Path) -> Any:
        """Extract pages/elements from document."""
        pass


class PDFLoader(BaseDocumentLoader):
    _converter = None

    def load(self, file_path: Path) -> Any:
        try:
            # Attempt to use Docling
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            
            logger.info("Using Docling for parsing PDF: %s", file_path)
            
            if PDFLoader._converter is None:
                # Configure options to generate pictures and tables
                pipeline_options = PdfPipelineOptions()
                pipeline_options.generate_page_images = False
                pipeline_options.generate_picture_images = True
                pipeline_options.images_scale = 2.0  # high-res for visual RAG
                
                PDFLoader._converter = DocumentConverter(
                    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
                )
            
            conv_res = PDFLoader._converter.convert(file_path)
            doc = conv_res.document
            
            # 1. Extract Pages/Paragraphs
            page_texts = {}
            for item in doc.texts:
                page_no = 1
                if item.prov:
                    page_no = item.prov[0].page_no
                page_texts.setdefault(page_no, []).append(item.text)
                
            pages = []
            for page_no in sorted(page_texts.keys()):
                pages.append({
                    'text': "\n".join(page_texts[page_no]),
                    'page_number': page_no,
                    'metadata': {'source': str(file_path), 'page': page_no - 1}
                })
                
            # 2. Extract Tables
            tables = []
            for table_idx, table in enumerate(doc.tables):
                page_no = 1
                if table.prov:
                    page_no = table.prov[0].page_no
                
                try:
                    df = table.export_to_dataframe(doc=doc)
                    markdown_table = df.to_markdown(index=False)
                except Exception as e:
                    logger.warning("Failed to export table to markdown: %s", e)
                    markdown_table = ""
                
                if markdown_table:
                    tables.append({
                        'text': markdown_table,
                        'page_number': page_no,
                        'table_index': table_idx
                    })
                    
            # 3. Extract Images
            images = []
            assets_dir = settings.data_root / "assets"
            assets_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate doc-specific prefix for assets
            doc_id = hashlib.md5(str(file_path.resolve()).encode()).hexdigest()
            
            for picture_idx, picture in enumerate(doc.pictures):
                page_no = 1
                if picture.prov:
                    page_no = picture.prov[0].page_no
                
                try:
                    img = picture.get_image(doc)
                    img_name = f"{doc_id}_img_{page_no}_{picture_idx}.png"
                    img_path = assets_dir / img_name
                    img.save(img_path, "PNG")
                    
                    images.append({
                        'asset_path': str(img_path.absolute()),
                        'page_number': page_no,
                        'image_index': picture_idx
                    })
                except Exception as e:
                    logger.warning("Failed to extract picture image: %s", e)
                    
            return {
                "pages": pages,
                "tables": tables,
                "images": images
            }
            
        except Exception as e:
            logger.error("Docling failed or not installed. Falling back to PyMuPDF: %s", e)
            # Fallback to PyMuPDF Loader
            from langchain_community.document_loaders import PyMuPDFLoader
            loader = PyMuPDFLoader(str(file_path))
            docs = loader.load()
            pages = []
            for doc in docs:
                pages.append({
                    'text': doc.page_content,
                    'page_number': doc.metadata.get('page', 0) + 1,
                    'metadata': doc.metadata
                })
            return pages


class DocxLoader(BaseDocumentLoader):
    def load(self, file_path: Path) -> List[Dict[str, Any]]:
        # Lazy import Docx2txt
        from langchain_community.document_loaders import Docx2txtLoader
        loader = Docx2txtLoader(str(file_path))
        docs = loader.load()
        pages = []
        for idx, doc in enumerate(docs):
            pages.append({
                'text': doc.page_content,
                'page_number': idx + 1,
                'metadata': doc.metadata
            })
        return pages


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, pages: List[Dict[str, Any]], metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk text pages into hierarchical chunks."""
        pass


class HierarchicalChunker(BaseChunker):
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, pages: List[Dict[str, Any]], metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Lazy imports for llama-index
        from llama_index.core import Document
        from llama_index.core.node_parser import HierarchicalNodeParser

        full_text = "\n\n".join([p['text'] for p in pages])
        doc = Document(text=full_text, metadata=metadata)
        
        # Build parser: parent (chunk_size) -> children (min(500, chunk_size))
        child_size = min(500, self.chunk_size)
        parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=[self.chunk_size, child_size],
            chunk_overlap=self.chunk_overlap
        )
        nodes = parser.get_nodes_from_documents([doc])

        def _node_id(n):
            return getattr(n, "id_", None) or getattr(n, "node_id", None)

        parent_text_map = {}
        for node in nodes:
            if not node.parent_node:
                parent_text_map[_node_id(node)] = node.text

        chunks = []
        for node in nodes:
            page_number = self._find_page_for_chunk(node.text, pages)
            parent_id = _node_id(node.parent_node) if node.parent_node else None
            chunks.append({
                "id": _node_id(node),
                "text": node.text,
                "is_leaf": bool(node.parent_node),
                "parent_id": parent_id,
                "parent_text": parent_text_map.get(parent_id) if parent_id else None,
                "page_number": page_number,
                "metadata": node.metadata
            })
        return chunks

    def _find_page_for_chunk(self, chunk_text: str, pages: List[Dict[str, Any]]) -> int:
        best_page = 1
        max_overlap = 0
        chunk_words = set(chunk_text.lower().split())
        for page in pages:
            page_words = set(page['text'].lower().split())
            overlap = len(chunk_words & page_words)
            if overlap > max_overlap:
                max_overlap = overlap
                best_page = page['page_number']
        return best_page


class IngestionPipeline:
    def __init__(self, db_service: VectorDatabaseService, embedding_service: BaseEmbeddingService):
        self.db_service = db_service
        self.embedding_service = embedding_service
        self.chunker = HierarchicalChunker(settings.chunk_size, settings.chunk_overlap)

    def generate_file_metadata(self, file_path: Path, collection_name: str, logical_file_name: Optional[str] = None) -> Dict[str, Any]:
        display_name = logical_file_name or file_path.name
        size = file_path.stat().st_size if file_path.exists() else 0
        
        if logical_file_name:
            id_key = f"{collection_name}|{display_name}|{size}"
        else:
            try:
                # Content MD5 hash for stability
                with open(file_path, 'rb') as f:
                    content_hash = hashlib.md5()
                    for chunk in iter(lambda: f.read(8192), b''):
                        content_hash.update(chunk)
                    content_md5 = content_hash.hexdigest()
                id_key = f"{file_path.resolve()}|{size}|{content_md5}"
            except Exception:
                id_key = str(file_path.resolve())

        doc_id = hashlib.md5(id_key.encode()).hexdigest()
        stem = Path(display_name).stem
        ext = Path(display_name).suffix

        return {
            'doc_id': doc_id,
            'file_name': display_name,
            'file_stem': stem,
            'file_extension': ext,
            'file_size': size,
            'collection': collection_name,
            'folder_name': file_path.parent.name if file_path.parent else collection_name,
            'full_path': str(file_path.absolute()),
            'ingest_source_path': f"{collection_name}/{display_name}",
            'created_at': datetime.utcnow().isoformat()
        }

    def process_document_generator(self, file_path: Path, collection_name: str, logical_file_name: Optional[str] = None):
        display = logical_file_name or file_path.name
        suffix = file_path.suffix.lower()
        
        # Loader Selection
        if suffix == '.pdf':
            loader = PDFLoader()
        elif suffix in ('.docx', '.doc'):
            loader = DocxLoader()
        else:
            yield {
                'type': 'progress',
                'status': 'failed',
                'percent': 0,
                'message': f'Unsupported file type: {suffix}'
            }
            yield {
                'file_name': display,
                'status': 'failed',
                'error': f'Unsupported file type: {suffix}'
            }
            return

        # Extract Pages
        yield {
            'type': 'progress',
            'status': 'parsing',
            'percent': 5,
            'message': f'Parsing document "{display}" and extracting text...'
        }
        try:
            pages = loader.load(file_path)
        except Exception as e:
            logger.error("Failed to load document %s: %s", display, e)
            yield {
                'type': 'progress',
                'status': 'failed',
                'percent': 5,
                'message': f'Load error: {e}'
            }
            yield {
                'file_name': display,
                'status': 'failed',
                'error': f'Load error: {e}'
            }
            return

        if not pages:
            yield {
                'type': 'progress',
                'status': 'failed',
                'percent': 5,
                'message': 'Text extraction yielded 0 pages.'
            }
            yield {
                'file_name': display,
                'status': 'failed',
                'error': 'Text extraction yielded 0 pages.'
            }
            return

        # Extract sub-elements if Docling dictionary was returned
        if isinstance(pages, dict):
            pages_list = pages.get("pages", [])
            tables_list = pages.get("tables", [])
            images_list = pages.get("images", [])
        else:
            pages_list = pages
            tables_list = []
            images_list = []

        # Check for scanned PDF
        total_text_len = sum(len((p.get("text") or "").strip()) for p in pages_list)
        if total_text_len < settings.min_extracted_text_chars:
            yield {
                'type': 'progress',
                'status': 'skipped',
                'percent': 100,
                'message': 'Document does not contain extractable text (likely scanned or image-only PDF).'
            }
            yield {
                'file_name': display,
                'status': 'skipped',
                'error': 'Document does not contain extractable text (likely scanned or image-only PDF).'
            }
            return

        yield {
            'type': 'progress',
            'status': 'parsed',
            'percent': 15,
            'message': f'Extracted {len(pages_list)} pages, {len(tables_list)} tables, and {len(images_list)} images ({total_text_len} characters).'
        }

        # Generate Metadata
        file_metadata = self.generate_file_metadata(file_path, collection_name, logical_file_name)
        doc_id = file_metadata['doc_id']

        # Setup Collection
        vs = self.db_service.get_collection_vector_size(collection_name)
        if vs is not None and vs != self.embedding_service.dimension:
            yield {
                'type': 'progress',
                'status': 'failed',
                'percent': 15,
                'message': f"Incompatible collection dimension: {vs} vs {self.embedding_service.dimension}"
            }
            yield {
                'file_name': display,
                'status': 'failed',
                'error': f"Target collection '{collection_name}' has vector dimension {vs}, which is incompatible with the selected embedding dimension {self.embedding_service.dimension}."
            }
            return

        yield {
            'type': 'progress',
            'status': 'setting_up',
            'percent': 20,
            'message': f"Setting up collection '{collection_name}'..."
        }
        full_collection = self.db_service.setup_collection(
            collection_name=collection_name,
            vector_size=self.embedding_service.dimension,
            recreate=False
        )

        # Handle Pre-existing Document
        if self.db_service.document_exists(collection_name, doc_id):
            logger.info("Document already exists, deleting older points for: %s", display)
            self.db_service.delete_document_points(collection_name, doc_id)

        # Chunk Pages
        yield {
            'type': 'progress',
            'status': 'chunking',
            'percent': 25,
            'message': 'Segmenting text into hierarchical chunks...'
        }
        try:
            chunks = self.chunker.chunk(pages_list, file_metadata)
        except Exception as e:
            logger.error("Chunking failed for %s: %s", display, e)
            yield {
                'type': 'progress',
                'status': 'failed',
                'percent': 25,
                'message': f'Chunking error: {e}'
            }
            yield {
                'file_name': display,
                'status': 'failed',
                'error': f'Chunking error: {e}'
            }
            return

        # Prepare text chunks
        for chunk in chunks:
            chunk["doc_type"] = "text"
            chunk["asset_path"] = None

        # Prepare table chunks
        table_chunks = []
        for idx, table in enumerate(tables_list):
            table_chunks.append({
                "id": f"table_{idx}",
                "text": table["text"],  # Markdown table string
                "is_leaf": True,
                "parent_id": None,
                "parent_text": None,
                "page_number": table["page_number"],
                "doc_type": "table",
                "asset_path": None,
                "metadata": {
                    "doc_type": "table",
                    "table_index": table["table_index"]
                }
            })

        # Process and caption image chunks
        image_chunks = []
        total_images = len(images_list)
        for idx, img in enumerate(images_list):
            yield {
                'type': 'progress',
                'status': 'captioning',
                'percent': 25 + int((idx / max(1, total_images)) * 10),
                'message': f'Captioning image {idx+1} of {total_images} using GPT-4o-mini...'
            }
            img_path = Path(img["asset_path"])
            caption = caption_image_with_gpt4o_mini(img_path)
            
            image_chunks.append({
                "id": f"image_{idx}",
                "text": caption,
                "is_leaf": True,
                "parent_id": None,
                "parent_text": None,
                "page_number": img["page_number"],
                "doc_type": "image",
                "asset_path": img["asset_path"],
                "metadata": {
                    "doc_type": "image",
                    "image_index": img["image_index"],
                    "asset_path": img["asset_path"]
                }
            })

        # Combine all elements
        all_chunks = chunks + table_chunks + image_chunks
        total_chunks = len(all_chunks)
        total_chunk_chars = sum(len(c['text']) for c in all_chunks)

        if not all_chunks:
            yield {
                'type': 'progress',
                'status': 'failed',
                'percent': 35,
                'message': 'Ingestion yielded 0 total segments.'
            }
            yield {
                'file_name': display,
                'status': 'failed',
                'error': 'Ingestion yielded 0 total segments.'
            }
            return

        yield {
            'type': 'progress',
            'status': 'chunked',
            'percent': 35,
            'message': f'Created {len(chunks)} text chunks, {len(table_chunks)} table chunks, and {len(image_chunks)} image chunks.'
        }

        # Embed & Upsert in batches
        batch_size = 20
        processed_chunks = 0
        processed_chars = 0

        for batch_idx in range(0, total_chunks, batch_size):
            chunk_batch = all_chunks[batch_idx : batch_idx + batch_size]
            
            # Embed
            try:
                texts = [c['text'] for c in chunk_batch]
                embeddings_batch = self.embedding_service.embed_documents(texts)
            except Exception as e:
                logger.error("Embedding generation failed for %s: %s", display, e)
                yield {
                    'type': 'progress',
                    'status': 'failed',
                    'percent': 35 + int((processed_chunks / total_chunks) * 65),
                    'message': f'Embedding error: {e}'
                }
                yield {
                    'file_name': display,
                    'status': 'failed',
                    'error': f'Embedding error: {e}'
                }
                return

            # Build Points
            points_batch = []
            for idx_in_batch, (chunk, embedding) in enumerate(zip(chunk_batch, embeddings_batch)):
                global_idx = batch_idx + idx_in_batch
                point_id = abs(hash(f"{doc_id}_{chunk['id']}")) % (10**15)
                page_num = chunk['page_number']
                chunk_id = f"{doc_id}:{page_num}:{global_idx}"
                source_path = file_metadata.get('ingest_source_path') or file_metadata.get('file_name', '')
                
                payload = {
                    **file_metadata,
                    'collection': full_collection,
                    'source_path': source_path,
                    'doc_id': doc_id,
                    'page_start': page_num,
                    'page_end': page_num,
                    'chunk_index': global_idx,
                    'chunk_id': chunk_id,
                    'text': chunk['text'],
                    'chunk_total': total_chunks,
                    'is_leaf': chunk['is_leaf'],
                    'parent_id': chunk['parent_id'],
                    'parent_text': chunk['parent_text'],
                    'page_number': page_num,
                    'doc_type': chunk.get("doc_type", "text"),
                    'asset_path': chunk.get("asset_path")
                }
                points_batch.append(PointStruct(id=point_id, vector=embedding, payload=payload))

            # Upsert
            try:
                self.db_service.upsert_points(collection_name, points_batch)
            except Exception as e:
                logger.error("Qdrant upsert failed for %s: %s", display, e)
                yield {
                    'type': 'progress',
                    'status': 'failed',
                    'percent': 35 + int((processed_chunks / total_chunks) * 65),
                    'message': f'Upsert error: {e}'
                }
                yield {
                    'file_name': display,
                    'status': 'failed',
                    'error': f'Upsert error: {e}'
                }
                return

            # Update stats
            processed_chunks += len(chunk_batch)
            processed_chars += sum(len(c['text']) for c in chunk_batch)
            
            # Progress calculation
            percent = 35 + int((processed_chunks / total_chunks) * 65)
            yield {
                'type': 'progress',
                'status': 'indexing',
                'percent': percent,
                'message': f'Embedded & indexed {processed_chunks} of {total_chunks} chunks ({processed_chars} of {total_chunk_chars} characters)...',
                'current_chunks': processed_chunks,
                'total_chunks': total_chunks,
                'current_characters': processed_chars,
                'total_characters': total_chunk_chars
            }

        # Yield Success
        yield {
            'doc_id': doc_id,
            'file_name': display,
            'chunks_created': total_chunks,
            'chunks_upserted': total_chunks,
            'status': 'success',
            'percent': 100,
            'message': f'Successfully ingested all {total_chunks} chunks.'
        }

    def process_document(self, file_path: Path, collection_name: str, logical_file_name: Optional[str] = None) -> Dict[str, Any]:
        generator = self.process_document_generator(file_path, collection_name, logical_file_name)
        last_res = {}
        for event in generator:
            if event.get("status") in ("success", "failed", "skipped"):
                last_res = event
        return last_res

