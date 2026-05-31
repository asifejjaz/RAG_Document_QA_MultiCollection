#!/usr/bin/env python3
"""
RAG System Performance Benchmark Runner
Queries the local FastAPI endpoint to test retrieval accuracy, factual precision,
visual description recall, and hallucination resistance.
"""
import os
import sys
import json
import time
import requests
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables
env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()

API_URL = "http://127.0.0.1:8000/api/chat/stream"
QUESTIONS_FILE = PROJECT_ROOT / "scratch" / "benchmark_questions.json"
REPORT_FILE = PROJECT_ROOT / "scratch" / "benchmark_report.json"

def run_query(query: str, collection_name: str = "hydrogen") -> dict:
    """Send query to the FastAPI chat stream endpoint and assemble response and citations."""
    payload = {
        "message": query,
        "history": [],
        "collection_name": collection_name,
        "llm_id": os.getenv("DEFAULT_LLM", "gemini_2_5_flash"),
        "embedding_id": os.getenv("EMBED_MODEL", "voyage_3_lite")
    }
    
    generated_text = []
    retrieved_pages = set()
    retrieved_types = set()
    retrieved_chunks = []
    
    try:
        response = requests.post(API_URL, json=payload, stream=True, timeout=30)
        if response.status_code != 200:
            return {"error": f"HTTP {response.status_code}: {response.text}"}
            
        for line in response.iter_lines():
            if not line:
                continue
            line_str = line.decode("utf-8").strip()
            if line_str.startswith("data:"):
                data_json = line_str[5:].strip()
                if data_json == "null":
                    break
                try:
                    event = json.loads(data_json)
                    if not event:
                        continue
                    if event.get("type") == "token":
                        generated_text.append(event.get("content", ""))
                    elif event.get("type") == "citations":
                        chunks = event.get("chunks", [])
                        retrieved_chunks.extend(chunks)
                        for c in chunks:
                            page = c.get("page_number")
                            if page is not None:
                                retrieved_pages.add(int(page))
                            doc_type = c.get("doc_type")
                            if doc_type:
                                retrieved_types.add(doc_type)
                except Exception as ex:
                    # Ignore parsing errors of partial content
                    pass
    except Exception as e:
        return {"error": str(e)}
        
    return {
        "answer": "".join(generated_text),
        "retrieved_pages": list(retrieved_pages),
        "retrieved_types": list(retrieved_types),
        "retrieved_chunks": retrieved_chunks
    }

def evaluate_test_case(case: dict, result: dict) -> dict:
    """Evaluate a single test case against its ground truth."""
    if "error" in result:
        return {
            "id": case["id"],
            "status": "failed",
            "error": result["error"],
            "retrieval_score": 0.0,
            "fact_score": 0.0,
            "hallucination_score": 0.0,
            "overall_pass": False
        }
        
    answer = result["answer"].lower()
    retrieved_pages = result["retrieved_pages"]
    
    # 1. Evaluate Retrieval Accuracy
    retrieval_pass = True
    expected_page = case.get("expected_page")
    if expected_page is not None:
        # Check if the expected page was retrieved in the citations
        retrieval_pass = int(expected_page) in retrieved_pages
        retrieval_score = 1.0 if retrieval_pass else 0.0
    else:
        retrieval_score = 1.0  # N/A for out of scope / negative tests
        
    # 2. Evaluate Factual Accuracy
    fact_matches = []
    fact_score = 1.0
    if case.get("key_facts"):
        for fact in case["key_facts"]:
            is_present = fact.lower() in answer
            fact_matches.append(is_present)
        fact_score = sum(fact_matches) / len(fact_matches) if fact_matches else 1.0
        
    # 3. Evaluate Hallucination / Negative Constraint Handling
    hallucination_pass = True
    refusal_score = 1.0
    
    if case.get("negative_test"):
        # For negative tests, we expect the system to refuse or state lack of info
        refusal_keywords = case.get("refusal_keywords", [
            "does not mention", "no information", "insufficient evidence", 
            "not specified", "not mentioned", "unable to find"
        ])
        refused = any(kw.lower() in answer for kw in refusal_keywords)
        hallucination_pass = refused
        refusal_score = 1.0 if refused else 0.0
        # If it's a negative test, factual score is inverse of whether it fabricated answers
        # A higher refusal score means it passed
        
    overall_pass = (retrieval_score >= 0.5) and (fact_score >= 0.5) and (refusal_score >= 0.5)
    
    return {
        "id": case["id"],
        "category": case["category"],
        "question": case["question"],
        "status": "success",
        "answer": result["answer"],
        "retrieved_pages": retrieved_pages,
        "retrieved_types": result["retrieved_types"],
        "expected_page": expected_page,
        "retrieval_score": retrieval_score,
        "fact_score": fact_score,
        "hallucination_score": refusal_score,
        "overall_pass": overall_pass,
        "details": {
            "retrieval_pass": retrieval_pass,
            "fact_matches": fact_matches if case.get("key_facts") else []
        }
    }

def main():
    print("=============================================================")
    print("        RAG SYSTEM PERFORMANCE BENCHMARK EVALUATOR")
    print("=============================================================")
    print(f"API Target: {API_URL}")
    
    if not QUESTIONS_FILE.exists():
        print(f"ERROR: Benchmark questions not found at {QUESTIONS_FILE}")
        sys.exit(1)
        
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        cases = json.load(f)
        
    print(f"Loaded {len(cases)} benchmark test cases.\n")
    
    # We will test against the collection named 'hydrogen' (prefixed as 'rag_hydrogen' locally)
    # Check if we should use 'hydrogen' or 'rag_hydrogen' based on listed collections
    collection_to_use = "hydrogen"
    try:
        collections_res = requests.get("http://127.0.0.1:8000/api/collections")
        if collections_res.status_code == 200:
            colls = collections_res.json()
            # If the backend returned a list of short names, check which matches
            print(f"Available Backend Collections: {colls}")
            if "hydrogen" in colls:
                collection_to_use = "hydrogen"
            elif "rag_hydrogen" in colls:
                collection_to_use = "rag_hydrogen"
            elif len(colls) > 0:
                collection_to_use = colls[0]
                print(f"Warning: 'hydrogen' collection not found. Using '{collection_to_use}' instead.")
    except Exception as e:
        print(f"Could not reach backend collections API: {e}. Defaulting to 'hydrogen'.")
        
    print(f"Using test collection: '{collection_to_use}'\n")
    
    results = []
    summary = {
        "total_cases": len(cases),
        "total_passed": 0,
        "retrieval_accuracy_pct": 0.0,
        "factual_accuracy_pct": 0.0,
        "hallucination_resistance_pct": 0.0,
        "categories": {}
    }
    
    total_retrieval = 0.0
    count_retrieval = 0
    total_fact = 0.0
    count_fact = 0
    total_refusal = 0.0
    count_refusal = 0
    
    for idx, case in enumerate(cases):
        print(f"[{idx+1}/{len(cases)}] Testing {case['id']} ({case['category']})...")
        print(f"  Q: {case['question']}")
        
        start_time = time.time()
        res = run_query(case["question"], collection_name=collection_to_use)
        elapsed = time.time() - start_time
        
        eval_res = evaluate_test_case(case, res)
        eval_res["latency_sec"] = round(elapsed, 2)
        results.append(eval_res)
        
        if eval_res["status"] == "success":
            print(f"  Result: Answer Length={len(eval_res['answer'])}, Latency={elapsed:.2f}s")
            print(f"  Scores: Retrieval={eval_res['retrieval_score']:.1f}, Fact={eval_res['fact_score']:.1f}, HallucinationRefusal={eval_res['hallucination_score']:.1f}")
            print(f"  Pass: {eval_res['overall_pass']}")
            
            if eval_res["overall_pass"]:
                summary["total_passed"] += 1
                
            if case.get("expected_page") is not None:
                total_retrieval += eval_res["retrieval_score"]
                count_retrieval += 1
                
            if case.get("key_facts"):
                total_fact += eval_res["fact_score"]
                count_fact += 1
                
            if case.get("negative_test"):
                total_refusal += eval_res["hallucination_score"]
                count_refusal += 1
        else:
            print(f"  Result: FAILED - {eval_res['error']}")
        print("-" * 60)
        
    # Calculate percentages
    summary["retrieval_accuracy_pct"] = round((total_retrieval / count_retrieval) * 100, 1) if count_retrieval else 100.0
    summary["factual_accuracy_pct"] = round((total_fact / count_fact) * 100, 1) if count_fact else 100.0
    summary["hallucination_resistance_pct"] = round((total_refusal / count_refusal) * 100, 1) if count_refusal else 100.0
    summary["pass_rate_pct"] = round((summary["total_passed"] / summary["total_cases"]) * 100, 1)
    
    # Save report
    report_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": summary,
        "results": results
    }
    
    with open(REPORT_FILE, "w", encoding="utf-8") as rf:
        json.dump(report_data, rf, indent=2)
        
    print("\n=============================================================")
    print("                    BENCHMARK RESULTS SUMMARY")
    print("=============================================================")
    print(f"Total Test Cases:            {summary['total_cases']}")
    print(f"Passed Cases:                {summary['total_passed']}")
    print(f"Overall Pass Rate:           {summary['pass_rate_pct']}%")
    print(f"Retrieval Recall:            {summary['retrieval_accuracy_pct']}%")
    print(f"Factual Precision:           {summary['factual_accuracy_pct']}%")
    print(f"Hallucination Resistance:    {summary['hallucination_resistance_pct']}%")
    print(f"Detailed report saved to:    {REPORT_FILE}")
    print("=============================================================")

if __name__ == "__main__":
    main()
