from typing import List, Dict, Any, TypedDict, Annotated
import operator

class AgentState(TypedDict):
    """
    Represents the state of the RAG agent graph.
    """
    # The message history, accumulating messages
    messages: Annotated[List[Dict[str, str]], operator.add]
    # The current query being processed (could be rewritten)
    query: str
    # The original query from the user
    original_query: str
    # The name of the collection (folder) to search in (optional)
    collection_name: str
    # The list of chunks retrieved from the vector database
    retrieved_chunks: List[Dict[str, Any]]
    # The generated answer from the LLM
    generation: str
    # The relevance grade of the chunks: 'yes', 'no'
    grade: str
    # The number of retrieval loops/rewrites completed
    loop_count: int
