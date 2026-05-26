from unittest.mock import MagicMock, patch
from backend.app.agents.graph import rag_graph

def test_graph_flow_direct_generate():
    """
    Assert that if the retrieved documents are graded relevant ('yes'),
    the workflow goes straight to generation without rewriting.
    """
    mock_embed = MagicMock()
    mock_embed.embed_query.return_value = [0.1] * 1536
    mock_embed.dimension = 1536
    
    mock_vector_db = MagicMock()
    mock_vector_db.search_similarity.return_value = [
        {"text": "Carbon dioxide is a greenhouse gas.", "source": "climate.pdf", "page_number": 4, "score": 0.88, "file_name": "climate.pdf"}
    ]
    
    mock_llm = MagicMock()
    # 1. Grader returns "yes"
    # 2. Generator returns explanation
    mock_llm.generate.side_effect = [
        "yes", 
        "Carbon dioxide is a greenhouse gas [Source: climate.pdf, page: 4]."
    ]
    
    with patch("backend.app.agents.nodes.EmbeddingServiceFactory.get_service", return_value=mock_embed), \
         patch("backend.app.agents.nodes.vector_db", mock_vector_db), \
         patch("backend.app.services.llm.LLMServiceFactory.get_service", return_value=mock_llm):
        
        inputs = {
            "messages": [{"role": "user", "content": "What is carbon dioxide?"}],
            "query": "What is carbon dioxide?",
            "original_query": "What is carbon dioxide?",
            "collection_name": "climate_docs",
            "retrieved_chunks": [],
            "generation": "",
            "grade": "",
            "loop_count": 0
        }
        
        result = rag_graph.invoke(inputs)
        
        assert result["generation"] == "Carbon dioxide is a greenhouse gas [Source: climate.pdf, page: 4]."
        assert result["grade"] == "yes"
        assert result["loop_count"] == 0
        assert mock_llm.generate.call_count == 2


def test_graph_flow_self_correction():
    """
    Assert that if the retrieved documents are graded irrelevant ('no'),
    the query is rewritten, a second retrieval is executed, and it proceeds to generate.
    """
    mock_embed = MagicMock()
    mock_embed.embed_query.return_value = [0.1] * 1536
    mock_embed.dimension = 1536
    
    mock_vector_db = MagicMock()
    # Mocking different return values for the first (irrelevant) vs second (relevant) searches
    mock_vector_db.search_similarity.side_effect = [
        [{"text": "Irrelevant rock structures.", "source": "geology.pdf", "page_number": 2, "score": 0.3, "file_name": "geology.pdf"}],  # 1st search
        [{"text": "Hydrogen gas burns cleanly with oxygen.", "source": "hydrogen.pdf", "page_number": 1, "score": 0.91, "file_name": "hydrogen.pdf"}] # 2nd search
    ]
    
    mock_llm = MagicMock()
    # 1. Grader returns "no"
    # 2. Query Rewriter returns rewritten query
    # 3. Grader returns "yes" for second search
    # 4. Generator returns response
    mock_llm.generate.side_effect = [
        "no",
        "hydrogen clean burning combustion properties",
        "yes",
        "Hydrogen gas burns cleanly [Source: hydrogen.pdf, page: 1]."
    ]
    
    with patch("backend.app.agents.nodes.EmbeddingServiceFactory.get_service", return_value=mock_embed), \
         patch("backend.app.agents.nodes.vector_db", mock_vector_db), \
         patch("backend.app.services.llm.LLMServiceFactory.get_service", return_value=mock_llm):
        
        inputs = {
            "messages": [{"role": "user", "content": "How clean does hydrogen burn?"}],
            "query": "How clean does hydrogen burn?",
            "original_query": "How clean does hydrogen burn?",
            "collection_name": "clean_energy",
            "retrieved_chunks": [],
            "generation": "",
            "grade": "",
            "loop_count": 0
        }
        
        result = rag_graph.invoke(inputs)
        
        assert result["generation"] == "Hydrogen gas burns cleanly [Source: hydrogen.pdf, page: 1]."
        assert result["grade"] == "yes"
        assert result["loop_count"] == 1
        assert result["query"] == "hydrogen clean burning combustion properties"
        assert mock_llm.generate.call_count == 4
