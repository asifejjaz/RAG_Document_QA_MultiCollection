#!/usr/bin/env python3
"""
Run all scripts in test order and append results to test_report_ollama.txt.
Order: (1) Ingestion  (2) Retrieval  (3) Inventory  (4) Preview  (5) Check numbers.

Usage:
  python scripts/run_script_tests_in_order.py
  python scripts/run_script_tests_in_order.py --data-root ./data --collection hydrogen_books

Optional: --data-root, --collection, --state, --report (path to report file).
If no data-root/collection, ingestion and retrieval tests are skipped (or use defaults).
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from typing import Tuple

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# Load env
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

REPORT_FILE = Path(__file__).parent / "test_report_ollama.txt"
STATE_DIR = os.getenv("STATE_ROOT", "state")
DATA_ROOT = os.getenv("DATA_ROOT", "data")
DEFAULT_COLLECTION = "hydrogen_books"


def run_cmd(cmd: list, env: dict = None, timeout: int = 300) -> Tuple[str, str, int]:
    """Run command; return (stdout, stderr, returncode)."""
    env = env or os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            cwd=PROJECT_ROOT,
            env=env,
        )
        return (r.stdout or "", r.stderr or "", r.returncode)
    except subprocess.TimeoutExpired:
        return ("", "Timeout", -1)
    except Exception as e:
        return ("", str(e), -1)


def append_section(report_path: Path, title: str, command: str, stdout: str, stderr: str, returncode: int):
    """Append a results section to the report file."""
    with open(report_path, "a", encoding="utf-8") as f:
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write(f"RUN: {title}\n")
        f.write(f"Time: {datetime.now().isoformat()}\n")
        f.write("-" * 80 + "\n")
        f.write(f"Command: {command}\n")
        f.write(f"Return code: {returncode}\n")
        f.write("-" * 80 + "\n")
        if stdout:
            f.write("STDOUT:\n")
            f.write(stdout)
            if not stdout.endswith("\n"):
                f.write("\n")
        if stderr:
            f.write("STDERR:\n")
            f.write(stderr)
            if not stderr.endswith("\n"):
                f.write("\n")
        f.write("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run script tests in order and append to report")
    parser.add_argument("--data-root", type=str, default=DATA_ROOT, help="Data root for ingestion")
    parser.add_argument("--collection", type=str, default=DEFAULT_COLLECTION, help="Collection name")
    parser.add_argument("--state", type=str, default=STATE_DIR, help="State directory")
    parser.add_argument("--report", type=str, default=str(REPORT_FILE), help="Report file path")
    parser.add_argument("--skip-ingestion", action="store_true", help="Skip ingestion (use existing collection)")
    parser.add_argument("--skip-ollama", action="store_true", help="Skip answer_local (Ollama)")
    args = parser.parse_args()
    report_path = Path(args.report)

    data_root = Path(args.data_root)
    state_dir = Path(args.state)
    collection = args.collection
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "reports").mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env["QDRANT_URL"] = env.get("VECTOR_DB_URL") or env.get("QDRANT_URL", "http://localhost:6333")
    env["OLLAMA_BASE_URL"] = env.get("OLLAMA_BASE_URL", "http://localhost:11434")
    env["STATE_ROOT"] = str(state_dir)
    env["PYTHONIOENCODING"] = "utf-8"

    results: list[Tuple[str, int]] = []  # (step_name, return_code)

    # 1) INGESTION
    if not args.skip_ingestion and data_root.exists():
        cmd = [sys.executable, "scripts/index_text.py", "--data-root", str(data_root), "--collection", collection]
        out, err, code = run_cmd(cmd, env=env, timeout=600)
        append_section(report_path, "1. Ingestion (index_text.py)", " ".join(cmd), out, err, code)
        results.append(("1. Ingestion (index_text.py)", code))
    else:
        if not data_root.exists():
            append_section(report_path, "1. Ingestion (index_text.py) [SKIPPED - no data root]", "", "", f"Data root not found: {data_root}", 0)
        else:
            append_section(report_path, "1. Ingestion (index_text.py) [SKIPPED]", "", "", "--skip-ingestion", 0)
        results.append(("1. Ingestion (index_text.py)", 0))

    # 2) RETRIEVAL - query_chunks
    cmd = [sys.executable, "scripts/query_chunks.py", "--q", "electrolyzer efficiency", "--collection", collection, "--topk", "5"]
    out, err, code = run_cmd(cmd, env=env, timeout=120)
    append_section(report_path, "2a. Retrieval (query_chunks.py)", " ".join(cmd), out, err, code)
    results.append(("2a. Retrieval (query_chunks.py)", code))

    # 2) RETRIEVAL - answer_local (Ollama)
    if not args.skip_ollama:
        cmd = [sys.executable, "scripts/answer_local.py", "--q", "What is hydrogen bunkering?", "--collection", collection, "--model", "qwen2.5:7b-instruct"]
        out, err, code = run_cmd(cmd, env=env, timeout=180)
        append_section(report_path, "2b. Retrieval (answer_local.py - Ollama)", " ".join(cmd), out, err, code)
        results.append(("2b. Retrieval (answer_local.py - Ollama)", code))
    else:
        append_section(report_path, "2b. Retrieval (answer_local.py) [SKIPPED]", "", "", "--skip-ollama", 0)
        results.append(("2b. Retrieval (answer_local.py)", 0))

    # 3) INVENTORY
    cmd = [sys.executable, "scripts/report_inventory.py", "--state", str(state_dir)]
    out, err, code = run_cmd(cmd, env=env, timeout=30)
    append_section(report_path, "3. Inventory (report_inventory.py)", " ".join(cmd), out, err, code)
    results.append(("3. Inventory (report_inventory.py)", code))

    # 4) PREVIEW - try first PDF or DOCX under data_root/collection
    preview_done = False
    if data_root.exists():
        coll_path = data_root / collection
        if coll_path.exists():
            for f in list(coll_path.rglob("*.pdf")) + list(coll_path.rglob("*.docx")) + list(coll_path.rglob("*.doc")):
                if f.suffix.lower() == ".pdf":
                    cmd = [sys.executable, "scripts/preview_extract.py", "--file", str(f), "--pages", "1"]
                else:
                    cmd = [sys.executable, "scripts/preview_extract.py", "--file", str(f), "--head", "500"]
                out, err, code = run_cmd(cmd, env=env, timeout=30)
                append_section(report_path, "4. Preview (preview_extract.py)", " ".join(cmd), out[:2000], err, code)
                results.append(("4. Preview (preview_extract.py)", code))
                preview_done = True
                break
    if not preview_done:
        append_section(report_path, "4. Preview (preview_extract.py) [SKIPPED - no file]", "", "", "No PDF/DOCX found under data_root/collection", 0)
        results.append(("4. Preview (preview_extract.py)", 0))

    # 5) CHECK NUMBERS
    cmd = [sys.executable, "scripts/check_numbers.py", "--collection", collection, "--state", str(state_dir)]
    out, err, code = run_cmd(cmd, env=env, timeout=60)
    append_section(report_path, "5. Check numbers (check_numbers.py)", " ".join(cmd), out, err, code)
    results.append(("5. Check numbers (check_numbers.py)", code))

    # Write FINAL TEST REPORT summary
    all_pass = all(r[1] == 0 for r in results)
    with open(report_path, "a", encoding="utf-8") as f:
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("FINAL TEST REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Date: {datetime.now().isoformat()}\n")
        f.write(f"Environment: QDRANT_URL={env.get('QDRANT_URL')}, OLLAMA_BASE_URL={env.get('OLLAMA_BASE_URL')}, STATE_ROOT={env.get('STATE_ROOT')}\n")
        f.write("-" * 80 + "\n")
        for name, ret in results:
            status = "PASS" if ret == 0 else "FAIL"
            f.write(f"  {status}  {name} (return code {ret})\n")
        f.write("-" * 80 + "\n")
        if all_pass:
            f.write("All scripts completed successfully.\n")
        else:
            f.write("One or more scripts failed. See sections above for details.\n")
        f.write("=" * 80 + "\n")

    print(f"Results appended to {report_path}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
