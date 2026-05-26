import os
import sys
import logging
from pathlib import Path

# Setup path to import backend modules
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from backend.app.services.ingestion import PDFLoader, IngestionPipeline, caption_image_with_gpt4o_mini
from backend.app.services.vector_db import VectorDatabaseService
from backend.app.services.embedding import EmbeddingServiceFactory
from backend.app.agents.nodes import retrieve, generate
from backend.app.agents.state import AgentState

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    pdf_path = Path("data/hydrogen_books/Hydrogen Bunkering at Ports by Eliseo Curcio.pdf")
    if not pdf_path.exists():
        logger.error("Test PDF not found at %s", pdf_path)
        sys.exit(1)

    logger.info("=========================================")
    logger.info("1. TESTING PDF LOADER (DOCLING)")
    logger.info("=========================================")
    loader = PDFLoader()
    result = loader.load(pdf_path)

    if not isinstance(result, dict):
        logger.error("PDFLoader did not return a dictionary. Result type: %s", type(result))
        sys.exit(1)

    pages = result.get("pages", [])
    tables = result.get("tables", [])
    images = result.get("images", [])

    logger.info("Extracted %d pages", len(pages))
    logger.info("Extracted %d tables", len(tables))
    logger.info("Extracted %d images", len(images))

    if tables:
        logger.info("\n--- SAMPLE TABLE ---")
        logger.info(tables[0]["text"][:500])
        logger.info("---------------------\n")

    if images:
        logger.info("\n--- SAMPLE IMAGE PATH ---")
        logger.info(images[0]["asset_path"])
        logger.info("-------------------------\n")
        
        # Test image captioning with GPT-4o-mini
        logger.info("Testing image captioning on the first image...")
        caption = caption_image_with_gpt4o_mini(Path(images[0]["asset_path"]))
        logger.info("Generated Caption: %s", caption)

    logger.info("=========================================")
    logger.info("2. TESTING INGESTION PIPELINE & QDRANT")
    logger.info("=========================================")
    db_service = VectorDatabaseService()
    embed_service = EmbeddingServiceFactory.get_service() # Default service
    
    collection_name = "test_multimodal_collection"
    pipeline = IngestionPipeline(db_service, embed_service)
    
    logger.info("Running ingestion pipeline for collection: %s", collection_name)
    events = pipeline.process_document_generator(pdf_path, collection_name)
    for event in events:
        if event.get("type") == "progress":
            logger.info("[%s] %d%%: %s", event.get("status"), event.get("percent"), event.get("message"))
        elif event.get("status") in ("success", "failed"):
            logger.info("Ingestion completed with status: %s", event)

    logger.info("=========================================")
    logger.info("3. TESTING RAG RETRIEVAL AND GENERATION")
    logger.info("=========================================")
    
    # Query about tables or images if they were extracted, else standard query
    query = "What does the document say about safety and bunkering operations?"
    if tables:
        query = "Summarize the key data in the tables of the document."
    elif images:
        query = "Describe the charts or diagrams in the document."

    logger.info("Running query: '%s'", query)
    
    # Initialize AgentState
    state: AgentState = {
        "query": query,
        "original_query": query,
        "collection_name": collection_name,
        "messages": [{"role": "user", "content": query}],
        "retrieved_chunks": [],
        "generation": "",
        "grade": "yes",
        "loop_count": 0
    }
    
    # Run Retrieve node
    config = {"configurable": {"llm_id": "azure", "embedding_id": None}}
    logger.info("Executing retrieval...")
    retrieved_state = retrieve(state, config)
    logger.info("Retrieved %d chunks", len(retrieved_state.get("retrieved_chunks", [])))
    for idx, c in enumerate(retrieved_state["retrieved_chunks"]):
        logger.info("Chunk %d: type=%s, page=%d, score=%.3f, text_preview=%s", 
                    idx+1, c.get("doc_type"), c.get("page_number"), c.get("score"), c.get("text", "")[:100].replace('\n', ' '))

    # Run Generate node (using OpenAI)
    state.update(retrieved_state)
    logger.info("Executing response generation...")
    generate_res = generate(state, config)
    
    logger.info("\n=== GENERATED RESPONSE ===")
    logger.info(generate_res.get("generation"))
    logger.info("==========================\n")

    # Clean up test collection
    logger.info("Cleaning up: deleting test collection...")
    db_service.delete_collection(collection_name)
    logger.info("Cleanup completed successfully.")

if __name__ == "__main__":
    main()
