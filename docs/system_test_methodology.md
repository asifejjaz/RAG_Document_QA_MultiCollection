# System Test Methodology

This document explains how the system was tested, which models and collections were used, and how answers were evaluated.

## Overview

The main objective was to validate the end-to-end RAG pipeline from ingestion to retrieval to answer generation.

The active system components are:

- `scripts/index_text.py` for document ingestion and Qdrant collection creation
- `frontend/app.py` for Streamlit retrieval and answer generation
- `scripts/eval_document_qa.py` for repeatable document-grounded QA evaluation
- `scripts/check_numbers.py` for numeric preservation validation

## Models and Embeddings

### Embedding model

- Model used: `text-embedding-ada-002`
- Managed through the environment variable `EMBED_MODEL`
- Embeddings are generated through `scripts/index_text.py` and used for similarity search in Qdrant

### Answer model

- Model used: Azure OpenAI deployment `my-gpt-4.1`
- Integrated through `scripts/azure_openai_env.py`
- Chat completion is performed by `chat_client.chat.completions.create(...)`

## Data collection

### Collection name

- Qdrant collection used for testing: `rag_hydrogen_books`

### Corpus

- Document ingested: `data/hydrogen_books/Hydrogen Bunkering at Ports by Eliseo Curcio.pdf`
- The collection contains dense chunks extracted from this PDF

## Test scripts and flow

### 1. Ingestion test

The ingestion workflow was validated with `scripts/index_text.py`:

- Load documents from `data/hydrogen_books`
- Extract text, chunk content, and metadata
- Create Qdrant points with payload fields such as `text`, `source_path`, `page_start`, `page_end`, `doc_id`, and `collection`
- Skip scanned/zero-text PDFs with explicit flags

### 2. Retrieval evaluation

Retrieval quality was validated in two ways:

- Direct Qdrant search via the collection with `scripts/query_chunks.py`
- Document-grounded evaluation using `scripts/eval_document_qa.py`

For each query, the pipeline:

- Generates an embedding for the query
- Searches `rag_hydrogen_books` for the top-k nearest chunks
- Builds a context string from retrieved payload entries
- Sends the context and question to the chat model

### 3. Answer evaluation

Answers were checked for correctness by comparing generated output to source facts.

The evaluation targeted three representative questions:

- What are the main forms of hydrogen discussed for bunkering?
- Which port is described as aiming to become a green hydrogen hub?
- What are the safety or storage challenges of hydrogen bunkering?

For each question, the model answer was evaluated against the source document and judged on:

- factual consistency with retrieved context
- correct citation of source page(s)
- completeness of the answer

### 4. Numeric preservation validation

Numeric integrity was validated with `scripts/check_numbers.py`:

- Compares numbers found in indexed chunks against the source document text
- Flags if a chunk contains no numeric tokens or if numeric values differ from the page source
- Writes results to `state/reports/numeric_issues.json`

## Evaluation criteria

The system was evaluated using these criteria:

- Retrieval relevance: top chunks should align with the query intent and source document page range
- Answer grounding: generated answers must use only provided retrieved context
- Source citations: answers should cite collection payload source information
- Numeric diagnostics: numeric preservation pipeline should detect number-related issues correctly

## Test artifacts

### Files created

- `docs/correctness_test_report.md` — detailed report with query, expected answer, actual answer, pass/fail status, and retrieval checks
- `scripts/eval_document_qa.py` — helper script to run document-grounded QA evaluation
- `state/reports/numeric_issues.json` — numeric preservation output from `scripts/check_numbers.py`

### Key results

- Verified that `rag_hydrogen_books` returns relevant chunks for hydrogen-bunkering queries
- Confirmed the answer generation path with Azure OpenAI chat is working
- Confirmed the numeric validation pipeline is active and produces diagnostics

## Notes

- The active ingestion/retrieval path is `scripts/index_text.py` -> `frontend/app.py`
- Legacy or inactive code such as `scripts/rg_pipeline.py` is not part of the current test flow

## How to rerun tests

Run the evaluation script:

```bash
python scripts/eval_document_qa.py
```

Run numeric preservation:

```bash
python scripts/check_numbers.py --collection rag_hydrogen_books --data-root ./data
```

If Qdrant is not running locally, set the URL in environment variables:

```bash
set QDRANT_URL=http://127.0.0.1:6333
set VECTOR_DB_URL=http://127.0.0.1:6333
```
