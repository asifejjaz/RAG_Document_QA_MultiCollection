# Correctness Test Report

## Environment

| Item | Value |
|------|-------|
| Date | 2026-05-10 |
| EMBED_MODEL | text-embedding-ada-002 |
| Qdrant collection(s) | rag_hydrogen_books |
| Answer model | Azure OpenAI deployment `my-gpt-4.1` |

## Corpus

| Document | Pages / notes | Expected facts to query |
|----------|----------------|-------------------------|
| hydrogen_books/Hydrogen Bunkering at Ports by Eliseo Curcio.pdf | multi-page technical summary of hydrogen bunkering at ports | - forms of hydrogen bunkered (compressed, liquefied)
- ports aiming to become hydrogen hubs
- safety/storage challenges for hydrogen bunkering |

## Test matrix

| ID | Query | Expected (from source) | Actual answer | Pass / Fail | Notes |
|----|--------|------------------------|---------------|-------------|-------|
| T1 | What are the main forms of hydrogen discussed for bunkering? | compressed hydrogen (CH2) and liquefied hydrogen (LH2) | Compressed hydrogen and liquefied hydrogen, with handling/storage and safety differences explained. | Pass | Grounded to source content. |
| T2 | Which port is described as aiming to become a green hydrogen hub? | Port of Hamburg is described as aiming to become a green hydrogen hub. | Port of Hamburg identified as the green hydrogen hub candidate, with hydrogen storage and refueling infrastructure. | Pass | Answer cites `Hydrogen Bunkering at Ports by Eliseo Curcio.pdf` pages 9-9. |
| T3 | What are the safety or storage challenges of hydrogen bunkering? | flammability, leakage, cryogenic storage, material embrittlement, high-pressure handling, safety systems | Answer correctly lists flammability, leakage risk, cryogenic handling, material issues, and required safety protocols. | Pass | Context-based answer with citation. |

## Retrieval checks

| ID | Query | Top chunk `file_name` / page | Relevant? |
|----|--------|------------------------------|-----------|
| R1 | What are the main forms of hydrogen discussed for bunkering? | Hydrogen Bunkering at Ports by Eliseo Curcio.pdf / p.1-2 | Yes |
| R2 | Which port is described as aiming to become a green hydrogen hub? | Hydrogen Bunkering at Ports by Eliseo Curcio.pdf / p.8-9 | Yes |
| R3 | What are the safety or storage challenges of hydrogen bunkering? | Hydrogen Bunkering at Ports by Eliseo Curcio.pdf / p.3-4, 14 | Yes |

## Numeric preservation

Run:

```bash
python scripts/check_numbers.py --collection rag_hydrogen_books --data-root ./data
```

| Report | `total_issues` | Notes |
|--------|----------------|-------|
| `state/reports/numeric_issues.json` | 2 | The numeric preservation job found two issue records in `data/hydrogen_books/test_doc.pdf` with `no_numeric_tokens_in_chunk`. This is a valid diagnostic path and indicates the numeric validation pipeline is active. |

