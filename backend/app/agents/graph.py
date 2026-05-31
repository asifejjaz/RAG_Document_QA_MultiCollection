import logging
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from backend.app.agents.state import AgentState
from backend.app.agents.nodes import retrieve, grade_documents, generate, rewrite_query

logger = logging.getLogger(__name__)

def decide_to_generate(state: AgentState) -> str:
    """
    Determines whether to generate an answer or rewrite the query.
    """
    grade = state.get("grade", "no")
    loop_count = state.get("loop_count", 0)
    
    if grade == "yes":
        logger.info("decide_to_generate edge: Found relevant chunks. Routing to 'generate'.")
        return "generate"
    
    if loop_count >= 1:
        logger.info("decide_to_generate edge: Low relevance but reached max rewrite limit (%d). Routing to 'generate'.", loop_count)
        return "generate"
        
    logger.info("decide_to_generate edge: Low relevance (grade='%s'). Routing to 'rewrite_query'.", grade)
    return "rewrite_query"

# Initialize StateGraph
workflow = StateGraph(AgentState)

# Add nodes to graph
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("rewrite_query", rewrite_query)

# Set workflow entrypoint
workflow.set_entry_point("retrieve")

# Add transition edges
workflow.add_edge("retrieve", "grade_documents")

# Add conditional routing
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "generate": "generate",
        "rewrite_query": "rewrite_query"
    }
)

# Loop back to retrieve after query rewrite
workflow.add_edge("rewrite_query", "retrieve")

# Complete after generation
workflow.add_edge("generate", END)

# Compile the LangGraph application
rag_graph = workflow.compile()
