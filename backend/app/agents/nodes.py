import logging
from typing import List, Dict, Any, Optional
from backend.app.agents.state import AgentState
from backend.app.services.vector_db import VectorDatabaseService
from backend.app.services.embedding import EmbeddingServiceFactory
from backend.app.services.llm import LLMServiceFactory

logger = logging.getLogger(__name__)

# Initialize services
vector_db = VectorDatabaseService()

def retrieve(state: AgentState, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Retrieve relevant documents from the vector database.
    """
    query = state.get("query")
    collection_name = state.get("collection_name")
    logger.info("Retrieve node: Searching for '%s' in collection '%s'", query, collection_name)

    try:
        # Get embedding service and compute query vector
        configurable = config.get("configurable", {}) if config else {}
        embedding_id = configurable.get("embedding_id")
        embed_service = EmbeddingServiceFactory.get_service(embedding_id)
        query_vector = embed_service.embed_query(query)
        
        # Search Qdrant
        chunks = vector_db.search_similarity(
            collection_name=collection_name,
            query_vector=query_vector,
            dimension=embed_service.dimension,
            top_k=5
        )
        logger.info("Retrieve node: Retrieved %d chunks", len(chunks))
        
        # Notify callback if registered
        on_retrieval = configurable.get("on_retrieval", None)
        if on_retrieval:
            on_retrieval(chunks)
    except Exception as e:
        logger.error("Error in retrieve node: %s", e)
        chunks = []

    return {"retrieved_chunks": chunks}


def grade_documents(state: AgentState, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Grades the relevance of retrieved documents to the query.
    """
    query = state.get("query")
    chunks = state.get("retrieved_chunks", [])
    logger.info("Grade node: Evaluating %d chunks against query '%s'", len(chunks), query)

    if not chunks:
        logger.info("Grade node: No chunks to grade. Outcome = no")
        return {"grade": "no"}

    # Format chunks for LLM evaluator
    chunks_text = "\n\n".join([
        f"--- Chunk {i+1} (Source: {c['source']}) ---\n{c['text']}"
        for i, c in enumerate(chunks)
    ])

    prompt = f"""You are a grading assistant evaluating the relevance of a set of retrieved document chunks to a user question.

Question: {query}

Retrieved Chunks:
{chunks_text}

Is the retrieved information relevant and helpful to answer the question? Answer with 'yes' if there is any relevant info, or 'no' if none of the chunks are relevant. Reply with ONLY the word 'yes' or 'no' in lowercase. Do not include any punctuation or explanation."""

    messages = [
        {"role": "system", "content": "You are a precise binary classification grader."},
        {"role": "user", "content": prompt}
    ]

    try:
        configurable = config.get("configurable", {}) if config else {}
        llm_id = configurable.get("llm_id")
        llm = LLMServiceFactory.get_service(llm_id)
        raw_grade = llm.generate(messages).strip().lower()
        # Clean up output
        grade = "yes" if "yes" in raw_grade else "no"
        logger.info("Grade node: LLM Grader returned '%s' (raw: '%s')", grade, raw_grade)
    except Exception as e:
        logger.error("Error in grader node: %s. Defaulting to 'yes'", e)
        grade = "yes"  # Fallback

    return {"grade": grade}


def generate(state: AgentState, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Generate response based on retrieved chunks and chat history.
    Enforces the 95% RAG / 5% general knowledge citation rule.
    """
    messages = state.get("messages", [])
    chunks = state.get("retrieved_chunks", [])
    original_query = state.get("original_query")
    
    logger.info("Generate node: Generating response for original query: '%s'", original_query)

    from backend.app.config import settings
    configurable = config.get("configurable", {}) if config else {}
    llm_id = configurable.get("llm_id") or settings.default_llm_id
    is_multimodal = llm_id in ("openai", "azure", "gemini_2_5_flash")

    # Helper function to encode image to base64
    def encode_image_to_base64(image_path_str: str) -> Optional[str]:
        import base64
        from pathlib import Path
        try:
            path = Path(image_path_str)
            if path.exists():
                with open(path, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error("Failed to encode image %s: %s", image_path_str, e)
        return None

    # Format chunks with sources for citation
    chunks_text = ""
    images_to_send = []
    
    formatted_chunks = []
    for idx, c in enumerate(chunks):
        doc_type = c.get("doc_type", "text")
        source_info = f"Source: {c['file_name']} (Page {c['page_number']})"
        
        if doc_type == "table":
            formatted_chunks.append(f"{source_info} [Table Element]\nContent:\n{c['text']}")
        elif doc_type == "image":
            formatted_chunks.append(f"{source_info} [Image/Chart Element]\nVisual Caption/Description: {c['text']}")
            if is_multimodal and c.get("asset_path"):
                img_b64 = encode_image_to_base64(c["asset_path"])
                if img_b64:
                    images_to_send.append(img_b64)
        else:
            # Standard text
            formatted_chunks.append(f"{source_info}\nContent: {c['text']}")

    if formatted_chunks:
        chunks_text = "\n\n".join(formatted_chunks)
    else:
        chunks_text = "No document chunks retrieved."

    # Construct the instruction and query text
    system_prompt = f"""You are a distinguished academic research professor and leading scientific expert.
Your primary directive is to formulate a comprehensive, rigorously detailed, and highly structured academic response to the user's question, drawing analysis exclusively from the retrieved document chunks and conversation history.

Strict Content Constraints:
1. 95% of the factual assertions in your response MUST be directly supported by, and traceable to, the provided retrieved chunks and chat history.
2. The remaining 5% of the content is reserved strictly for academic synthesis, logical transitions, and grammatical coherence. Never introduce external facts, assumptions, or speculative information not directly supported by the retrieved texts.
3. If the retrieved chunks do not contain sufficient evidence to formulate a scholarly answer, explicitly state: "Based on the retrieved document corpus, there is insufficient evidence to address this question." Do not attempt to extrapolate or invent information.
4. Integrate precise in-text citations. Format citations as [Source: file_name, page: page_number] immediately following the cited claim.

Scholarly Structure & Formatting Guidelines:
- Under no circumstances should your response be a single consolidated paragraph.
- Write your response using clean, professional Markdown formatting.
- **Title / Theme Headings**: Structure your analysis with clear, descriptive section headings (using `##` and `###`) representing the core themes.
- **Multi-Paragraph Exposition**: Break down complex arguments into multiple, logically organized paragraphs. Start with a thematic overview and elaborate with findings from the texts.
- **Term Boldface**: Bold (`**term**`) key academic concepts, specialized terminology, variables, and critical data points.
- **Lists and Enumeration**: Use bulleted or numbered lists for comparing methods, listing key takeaways, summarizing factors, or detailing sequential steps.
- **Tone**: Maintain an objective, authoritative, precise, and analytical academic tone.

Retrieved Chunks:
{chunks_text}"""

    # Prepare chat history to feed the LLM
    # We want to replace or prepend the system prompt, and keep the user query.
    chat_messages = [{"role": "system", "content": system_prompt}]
    
    # Add previous messages, excluding system message
    for msg in messages:
        if msg.get("role") != "system":
            chat_messages.append({"role": msg.get("role"), "content": msg.get("content")})

    # If multimodal and we have images, inject them into the last user message
    if is_multimodal and images_to_send:
        last_user_msg = None
        for msg in reversed(chat_messages):
            if msg.get("role") == "user":
                last_user_msg = msg
                break
        
        if last_user_msg:
            original_content = last_user_msg.get("content", "")
            if isinstance(original_content, list):
                for img_b64 in images_to_send:
                    original_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_b64}"
                        }
                    })
            else:
                content_list = [{"type": "text", "text": str(original_content)}]
                for img_b64 in images_to_send:
                    content_list.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_b64}"
                        }
                    })
                last_user_msg["content"] = content_list

    # Call LLM service
    llm = LLMServiceFactory.get_service(llm_id)
    
    # Check if a streaming callback is registered
    on_token = configurable.get("on_token", None)
    
    generation_text = ""
    try:
        if on_token:
            logger.info("Generate node: Streaming tokens...")
            for token in llm.generate_stream(chat_messages):
                on_token(token)
                generation_text += token
        else:
            logger.info("Generate node: Generating response synchronously...")
            generation_text = llm.generate(chat_messages)
    except Exception as e:
        logger.error("Error in generate node: %s", e)
        err_msg = str(e)
        if "RESOURCE_EXHAUSTED" in err_msg or "429" in err_msg or "quota" in err_msg.lower() or "limit" in err_msg.lower():
            error_details = (
                "⚠️ **Gemini API Quota Reached (429 Resource Exhausted)**\n\n"
                "You have exceeded your current API quota limit for Google Gemini (e.g., the 20 requests/day limit on the free tier).\n\n"
                "**Details:**\n"
                f"```\n{err_msg}\n```\n\n"
                "Please check your billing details, API limits, or try switching to another model (like Azure OpenAI) in the settings."
            )
        else:
            error_details = f"An error occurred while generating the response: {err_msg}"
        
        generation_text = error_details
        if on_token:
            on_token(error_details)

    # Append assistant's response to the message history
    assistant_msg = {"role": "assistant", "content": generation_text}
    
    return {
        "generation": generation_text,
        "messages": [assistant_msg]
    }


def rewrite_query(state: AgentState, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Rewrite the query to optimize retrieval.
    """
    original_query = state.get("original_query")
    messages = state.get("messages", [])
    loop_count = state.get("loop_count", 0) + 1
    logger.info("Rewrite node: Rewriting query. Loop count: %d", loop_count)

    prompt = f"""You are a search query optimizer. Your task is to rewrite the user's original question into a search query that is optimized for vector-based semantic similarity search.
Analyze the user question and context, then output ONLY the optimized query. Do not add any explanation, quotation marks, or preamble.

Original Question: {original_query}"""

    chat_messages = [
        {"role": "system", "content": "You are a precise search query optimizer."},
    ]
    # Include recent context if available
    for msg in messages[-4:]:  # last 4 messages for context
        if msg.get("role") != "system":
            chat_messages.append({"role": msg.get("role"), "content": msg.get("content")})
            
    chat_messages.append({"role": "user", "content": prompt})

    try:
        configurable = config.get("configurable", {}) if config else {}
        llm_id = configurable.get("llm_id")
        llm = LLMServiceFactory.get_service(llm_id)
        rewritten_query = llm.generate(chat_messages).strip()
        # Clean any surrounding quotes
        rewritten_query = rewritten_query.strip('"').strip("'")
        logger.info("Rewrite node: Rewrote '%s' to '%s'", original_query, rewritten_query)
        
        # Notify callback if registered
        on_query_rewrite = configurable.get("on_query_rewrite", None)
        if on_query_rewrite:
            on_query_rewrite(rewritten_query)
    except Exception as e:
        logger.error("Error in rewrite node: %s. Using original query.", e)
        rewritten_query = original_query

    return {
        "query": rewritten_query,
        "loop_count": loop_count
    }
