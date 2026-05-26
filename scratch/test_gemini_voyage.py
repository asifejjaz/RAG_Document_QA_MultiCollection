import os
import sys
import logging
from pathlib import Path

# Setup path to import backend modules
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from backend.app.services.ingestion import PDFLoader, IngestionPipeline
from backend.app.services.vector_db import VectorDatabaseService
from backend.app.services.embedding import EmbeddingServiceFactory
from backend.app.services.llm import LLMServiceFactory
from backend.app.agents.nodes import retrieve, generate
from backend.app.agents.state import AgentState

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("=========================================")
    logger.info("1. TESTING GEMINI AND VOYAGE APIS DIRECTLY")
    logger.info("=========================================")
    
    # 1a. Test Gemini LLM Service
    logger.info("Testing Gemini LLM Service...")
    try:
        gemini_service = LLMServiceFactory.get_service("gemini_2_5_flash")
        gemini_response = gemini_service.generate([
            {"role": "user", "content": "Say 'hello from gemini'"}
        ])
        logger.info("Gemini Output: %s", gemini_response)
    except Exception as e:
        logger.error("Gemini direct test failed: %s", e)
        sys.exit(1)
        
    # 1b. Test Voyage Embedding Service
    logger.info("Testing Voyage Embedding Service...")
    try:
        voyage_service = EmbeddingServiceFactory.get_service("voyage_3_lite")
        emb = voyage_service.embed_query("hello")
        logger.info("Voyage vector length: %d (expected 512)", len(emb))
        if len(emb) != 512:
            logger.error("Unexpected Voyage embedding dimension: %d", len(emb))
            sys.exit(1)
    except Exception as e:
        logger.error("Voyage direct test failed: %s", e)
        sys.exit(1)

    logger.info("=========================================")
    logger.info("2. TESTING MULTIMODAL INGESTION PIPELINE (VOYAGE + NEW QDRANT)")
    logger.info("=========================================")
    
    pdf_path = Path("data/hydrogen_books/Hydrogen Bunkering at Ports by Eliseo Curcio.pdf")
    if not pdf_path.exists():
        logger.warning("Test PDF not found at %s. Skipping full RAG test.", pdf_path)
        logger.info("Tests 1a & 1b passed successfully!")
        sys.exit(0)

    db_service = VectorDatabaseService()
    collection_name = "test_gemini_voyage_collection"
    
    pipeline = IngestionPipeline(db_service, voyage_service)
    
    logger.info("Running ingestion pipeline with Voyage embeddings on collection: %s", collection_name)
    events = pipeline.process_document_generator(pdf_path, collection_name)
    
    # Process and print ingestion events
    for event in events:
        if event.get("type") == "progress":
            logger.info("[%s] %d%%: %s", event.get("status"), event.get("percent"), event.get("message"))
        elif event.get("status") in ("success", "failed"):
            logger.info("Ingestion completed with status: %s", event)
            if event.get("status") == "failed":
                sys.exit(1)

    logger.info("=========================================")
    logger.info("3. TESTING RAG RETRIEVAL AND GEMINI VISUAL RESPONSE SYNTHESIS")
    logger.info("=========================================")
    
    query = "Describe the charts or diagrams in the document."
    logger.info("Executing retrieval for query: '%s'", query)
    
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
    
    config = {"configurable": {"llm_id": "gemini_2_5_flash", "embedding_id": "voyage_3_lite"}}
    
    # Run Retrieve node
    retrieved_state = retrieve(state, config)
    logger.info("Retrieved %d chunks", len(retrieved_state.get("retrieved_chunks", [])))
    
    # Run Generate node (using Gemini)
    state.update(retrieved_state)
    logger.info("Executing response generation via Gemini 2.5 Flash...")
    generate_res = generate(state, config)
    
    logger.info("\n=== GEMINI GENERATED RESPONSE ===")
    logger.info(generate_res.get("generation"))
    logger.info("=================================\n")

    # Clean up test collection
    logger.info("Cleaning up: deleting test collection...")
    db_service.delete_collection(collection_name)
    logger.info("Cleanup completed successfully.")
    logger.info("ALL TESTS COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
