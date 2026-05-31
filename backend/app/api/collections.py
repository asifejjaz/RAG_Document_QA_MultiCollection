import logging
import shutil
import tempfile
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from backend.app.config import settings
from backend.app.services.vector_db import VectorDatabaseService
from backend.app.services.embedding import EmbeddingServiceFactory
from backend.app.services.ingestion import IngestionPipeline

logger = logging.getLogger(__name__)

router = APIRouter()
vector_db = VectorDatabaseService()

@router.get("")
@router.get("/")
def list_collections(embedding_id: Optional[str] = None):
    """
    List collections/folders that match the embedding dimension of the current active embedding model.
    Includes aggregate project stats.
    """
    try:
        embed_service = EmbeddingServiceFactory.get_service(embedding_id)
        collections = vector_db.get_collections_for_dimension(embed_service.dimension)
        stats = vector_db.get_aggregate_statistics()
        return {
            "collections": collections,
            "statistics": stats
        }
    except Exception as e:
        logger.error("Failed to list collections: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to list collections: {str(e)}")


@router.get("/{collection_name}/documents")
def list_documents(collection_name: str):
    """
    List unique documents stored in a specific collection/folder.
    """
    try:
        docs = vector_db.get_collection_documents(collection_name)
        return docs
    except Exception as e:
        logger.error("Failed to list documents for collection '%s': %s", collection_name, e)
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


@router.get("/{collection_name}/statistics")
def get_collection_statistics(collection_name: str):
    """
    Get detailed breakdown of statistics for a specific folder/collection.
    """
    try:
        stats = vector_db.get_collection_statistics(collection_name)
        return stats
    except Exception as e:
        logger.error("Failed to get statistics for collection '%s': %s", collection_name, e)
        raise HTTPException(status_code=500, detail=f"Failed to get collection statistics: {str(e)}")


@router.delete("/{collection_name}")
def delete_collection(collection_name: str):
    """
    Delete an entire collection/folder.
    """
    try:
        vector_db.delete_collection(collection_name)
        return {"status": "success", "message": f"Collection '{collection_name}' deleted."}
    except Exception as e:
        logger.error("Failed to delete collection '%s': %s", collection_name, e)
        raise HTTPException(status_code=500, detail=f"Failed to delete collection: {str(e)}")


@router.delete("/{collection_name}/documents/{doc_id}")
def delete_document(collection_name: str, doc_id: str):
    """
    Delete a single document (all its chunks) from a specific collection.
    """
    try:
        vector_db.delete_document_points(collection_name, doc_id)
        return {"status": "success", "message": f"Document '{doc_id}' deleted from collection '{collection_name}'."}
    except Exception as e:
        logger.error("Failed to delete document '%s' in collection '%s': %s", doc_id, collection_name, e)
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")


def save_ingestion_report(collection_name: str, result: dict) -> None:
    """Save document ingestion report for script/report inventory aggregator."""
    try:
        state_dir = settings.state_root
        state_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = state_dir / f"ingestion_{collection_name}_{timestamp_str}.json"
        
        status = result.get("status", "failed")
        
        report = {
            'collection': collection_name,
            'timestamp': datetime.utcnow().isoformat(),
            'files_processed': 1 if status == 'success' else 0,
            'files_failed': 1 if status == 'failed' else 0,
            'files_skipped': 1 if status == 'skipped' else 0,
            'total_chunks': result.get('chunks_upserted', 0),
            'results': [result]
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
    except Exception as e:
        logger.error("Failed to save ingestion report: %s", e)


@router.get("/inventory/report")
def get_inventory_report():
    """
    Generate and retrieve the overall ingestion/inventory report.
    """
    try:
        from scripts.report_inventory import generate_inventory_report
        report = generate_inventory_report(str(settings.state_root))
        return report
    except Exception as e:
        logger.error("Failed to generate inventory report: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")


@router.post("/upload")
def upload_file(
    file: UploadFile = File(...),
    collection_name: str = Form("default"),
    embedding_id: Optional[str] = Form(None)
):
    """
    Upload a document (PDF/DOCX) and run the hierarchical ingestion pipeline
    to parse, chunk, embed, and index into the chosen Qdrant collection.
    Streams progress updates back as newline-delimited JSON.
    """
    suffix = Path(file.filename).suffix.lower()
    if suffix not in (".pdf", ".docx", ".doc"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{suffix}'. Only PDF and DOCX files are supported."
        )
    
    # Save the uploaded file payload to a temp file in order to pass to our loaders
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = Path(temp_file.name)
    except Exception as e:
        logger.error("Failed to save uploaded file payload: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to save file payload: {str(e)}")

    def event_generator():
        try:
            # Initialize IngestionPipeline
            embed_service = EmbeddingServiceFactory.get_service(embedding_id)
            pipeline = IngestionPipeline(vector_db, embed_service)
            
            # Process document generator
            generator = pipeline.process_document_generator(
                file_path=temp_path,
                collection_name=collection_name,
                logical_file_name=file.filename
            )
            
            final_res = None
            for event in generator:
                if event.get("status") in ("success", "failed", "skipped"):
                    final_res = event
                yield json.dumps(event) + "\n"
            
            if final_res:
                save_ingestion_report(collection_name, final_res)
                
        except Exception as e:
            logger.error("Error during document upload/ingestion: %s", e)
            yield json.dumps({
                'type': 'progress',
                'status': 'failed',
                'percent': 0,
                'message': f"Upload processing failed: {str(e)}"
            }) + "\n"
            yield json.dumps({
                'file_name': file.filename,
                'status': 'failed',
                'error': str(e)
            }) + "\n"
        finally:
            # Clean up temp file
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception as ce:
                    logger.error("Failed to clean up temp file %s: %s", temp_path, ce)

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")
