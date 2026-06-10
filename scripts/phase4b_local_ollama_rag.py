#In this stage of the project, Composer through a Cursor IDE supported the boiler plate generation for this script

"""Local Ollama baseline and hybrid RAG over Chroma (Cohere embeddings) + optional cross-encoder rerank.

Reads stratified questions from ``pipeline_constants.QUESTIONS_JSONL``, writes baseline and hybrid JSONL,
and may sync to PostgreSQL when ``db_utils`` is available.
"""

import argparse
import json
import os
import re
import platform
import random
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

#imports the necessary libraries
import chromadb
import cohere
from cohere.errors import (
    #imports the necessary errors
    GatewayTimeoutError,
    InternalServerError,
    ServiceUnavailableError,
    TooManyRequestsError,
)
import numpy as np
import requests
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
#imports the necessary constants
from pipeline_constants import (
    BASE_DIR,
    BASELINE_JSONL,
    EVAL_PROMPT_COUNT,
    HYBRID_JSONL,
    INFERENCE_TAG,
    QUESTIONS_JSONL,
)
#sets the chroma path for the script
CHROMA_PATH = BASE_DIR / "data" / "04_chroma_db_cohere"
COLLECTION_NAME = "send_statutory_cohere"
#sets the baseline output for the script
BASELINE_OUTPUT = BASELINE_JSONL
RAG_OUTPUT = HYBRID_JSONL
#sets the prompt version for the script
#sets the default baseline model for the script
PROMPT_VERSION = INFERENCE_TAG
#sets the default hybrid model for the script
DEFAULT_BASELINE_MODEL = "llama3.1:8b-instruct-q4_K_M"
DEFAULT_HYBRID_MODEL = "senco-ft:latest"
#sets the top k for the script
TOP_K = 3
#sets the retrieve k for the script
RETRIEVE_K = 24
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
#sets the max tokens for the script
MAX_TOKENS = 150
#sets the temperature for the script
TEMPERATURE = 0.0
#sets the top p for the script
TOP_P = 1.0
#sets the max answer sentences for the script
MAX_ANSWER_SENTENCES = 3

#sets the shared system rules for the script
SHARED_SYSTEM_RULES = (
    "You are an expert UK SENCo writing brief guidance for mainstream school colleagues. "
    "RULES: "
    "1) Third-person, staff-facing voice (e.g. 'Staff may…', 'The school can…', 'The pupil may…'). "
    "Do not use first person ('I would…'). "
    f"2) Maximum {MAX_ANSWER_SENTENCES} sentences in the Answer. "
    "3) Be direct and practical: lead with the most useful staff-facing guidance. "
    "4) Answer the QUESTION first in sentences 1–2. Use sentence 3 only for a brief caveat if needed. "
    "5) Do not open with literature-review phrasing (e.g. 'The review does not address…', "
    "'This is an area where further research…'). Prefer collegial, actionable wording."
)

#sets the baseline system rules for the script
BASELINE_SYSTEM_RULES = (
    f"{SHARED_SYSTEM_RULES} "
    "6) Use your pre-trained knowledge only. Do not invent citations or statutory references."
)

#sets the hybrid system rules for the script
HYBRID_SYSTEM_RULES = (
    f"{SHARED_SYSTEM_RULES} "
    "6) Use only the provided CONTEXT for facts. Do not add legal or policy detail not supported by CONTEXT. "
    "7) If CONTEXT is partial, still answer with the best supported guidance plus one safe next step "
    "grounded in CONTEXT; name any gap briefly at the end, not as the opening."
)

#loads the questions from the path
def load_questions(path: Path) -> list[dict]:
    questions = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            questions.append(json.loads(line))
    return questions

#gets the cohere api key from the environment
def _get_cohere_api_key() -> str:
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise ValueError("COHERE_API_KEY is missing. Add it to .env before running.")
    return api_key

#builds the chroma collection
def build_chroma_collection():
    api_key = _get_cohere_api_key()
    if not CHROMA_PATH.exists():
        raise FileNotFoundError(f"Chroma path not found: {CHROMA_PATH}")

    embed_fn = embedding_functions.CohereEmbeddingFunction(
        api_key=api_key,
        model_name="embed-english-v3.0",
    )
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return client.get_collection(name=COLLECTION_NAME, embedding_function=embed_fn)

#builds the cohere client   
def build_cohere_client() -> cohere.Client:
    return cohere.Client(_get_cohere_api_key())

#normalizes the embeddings
#handles the unsupported embedding response shape
def normalize_embeddings(raw_response):
    if hasattr(raw_response, "embeddings"):
        emb_obj = raw_response.embeddings
        if hasattr(emb_obj, "float") and emb_obj.float:
            return emb_obj.float
        if isinstance(emb_obj, list):
            return emb_obj
    if isinstance(raw_response, dict) and "embeddings" in raw_response:
        emb = raw_response["embeddings"]
        if isinstance(emb, list):
            return emb
        if hasattr(emb, "float") and emb.float:
            return emb.float
    raise ValueError(f"Unsupported embedding response shape: {raw_response}")

#sets the cohere embed retry errors for the script
_COHERE_EMBED_RETRY_ERRORS = (
    InternalServerError,
    ServiceUnavailableError,
    GatewayTimeoutError,
    TooManyRequestsError,
)

#embeds the query
def embed_query(
    co_client: cohere.Client,
    query_text: str,
    max_retries: int = 6,
) -> list[list[float]]:
    delay_seconds = 2.0
    for attempt in range(max_retries):
        try:
            response = co_client.embed(
                texts=[query_text],
                model="embed-english-v3.0",
                input_type="search_query",
            )
            return normalize_embeddings(response)
        except _COHERE_EMBED_RETRY_ERRORS as exc:
            if attempt >= max_retries - 1:
                raise
            print(
                f"Cohere embed failed (attempt {attempt + 1}/{max_retries}): {exc}; "
                f"retrying in {delay_seconds:.0f}s"
            )
            time.sleep(delay_seconds)
            delay_seconds = min(delay_seconds * 2, 60.0)

#trims the document
def trim_doc(text: str, max_chars: int = 1400) -> str:
    return text if len(text) <= max_chars else text[:max_chars] + " ..."

#sets the cross encoder
_cross_encoder = None

#gets the cross encoder
def get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    return _cross_encoder

#reranks the documents with the cross encoder
def rerank_with_cross_encoder(query: str, documents: list[str], top_n: int) -> list[str]:
    documents = [d for d in documents if isinstance(d, str) and d.strip()]
    if not documents:
        return []
    if len(documents) <= top_n:
        return documents

    model = get_cross_encoder()
    pairs = [[query, doc] for doc in documents]
    scores = model.predict(pairs)
    ranked_indices = sorted(range(len(documents)), key=lambda i: scores[i], reverse=True)
    return [documents[i] for i in ranked_indices[:top_n]]

#calls the ollama model
def call_ollama(model_name: str, system_prompt: str, user_prompt: str) -> str:
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "options": {
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "num_predict": MAX_TOKENS,
        },
        "stream": False,
    }
    resp = requests.post("http://127.0.0.1:11434/api/chat", json=payload, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    return data["message"]["content"].strip()

#normalizes the answer format
def normalize_answer_format(raw_answer: str) -> str:
    """Ensure a single Answer: line (no Evidence line in either condition)."""
    text = (raw_answer or "").strip()
    if not text:
        return "Answer: The model could not generate a response."
    if text.lower().startswith("answer:"):
        return text
    return f"Answer: {text}"

#trims the answer to the max sentences
def trim_answer_to_max_sentences(
    formatted_answer: str, max_sentences: int = MAX_ANSWER_SENTENCES
) -> str:
    """Shared post-generation guardrail: cap Answer body to max_sentences (both arms)."""
    text = normalize_answer_format(formatted_answer)
    if not text.lower().startswith("answer:"):
        return text
    body = text.split(":", 1)[1].strip()
    if not body:
        return text
    parts = re.split(r"(?<=[.!?])\s+", body.replace("\n", " "))
    parts = [p.strip() for p in parts if p.strip()]
    if not parts:
        return text
    trimmed = " ".join(parts[:max_sentences])
    if trimmed and trimmed[-1] not in ".!?":
        trimmed += "."
    return f"Answer: {trimmed}"

#finalizes the answer
def finalize_answer(raw_answer: str) -> str:
    return trim_answer_to_max_sentences(raw_answer)

#formats the user prompt
def format_user_prompt(question: str, context: str | None = None) -> str:
    """Shared user template; hybrid adds CONTEXT above the question."""
    output_fmt = (
        "OUTPUT FORMAT:\n"
        f"Answer: <staff-facing third-person response, maximum {MAX_ANSWER_SENTENCES} sentences; "
        "sentences 1–2 answer the QUESTION directly>"
    )
    if context is not None:
        return (
            f"CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\n"
            "Answer using only CONTEXT. Prefer practical staff-facing wording over review summaries.\n\n"
            f"{output_fmt}"
        )
    return f"QUESTION:\n{question}\n\n{output_fmt}"

#runs the baseline
def run_baseline(questions: list[dict], model_name: str, baseline_output: Path):
    baseline_output.parent.mkdir(parents=True, exist_ok=True)
    with baseline_output.open("w", encoding="utf-8") as out_f:
        for i, item in enumerate(questions, start=1):
            q = item["question"]
            answer = call_ollama(
                model_name, BASELINE_SYSTEM_RULES, format_user_prompt(q)
            )
            answer = finalize_answer(answer)
            row = {
                "id": item.get("id", i),
                "category": item.get("category", "Unknown"),
                "model": f"{model_name}-baseline-local-{PROMPT_VERSION}",
                "prompt": q,
                "retrieved_context": "NONE - BASELINE",
                "ai_response": answer,
            }
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[baseline {i}/{len(questions)}] done")
    print(f"Baseline output saved to: {baseline_output}")

#runs the rag
def run_rag(
    questions: list[dict],
    model_name: str,
    rag_output: Path,
    start_index: int = 1,
):
    """Run hybrid RAG. start_index is 1-based; use >1 to append after a partial run."""
    if start_index < 1:
        raise ValueError("start_index must be >= 1")
    total = len(questions)
    if start_index > total:
        raise ValueError(f"start_index {start_index} exceeds question count {total}")

    collection = build_chroma_collection()
    co_client = build_cohere_client()
    rag_output.parent.mkdir(parents=True, exist_ok=True)

    existing_rows = _count_jsonl_rows(rag_output)
    if start_index == 1:
        file_mode = "w"
    else:
        if existing_rows != start_index - 1:
            raise ValueError(
                f"Resume expects {start_index - 1} existing hybrid rows in {rag_output}, "
                f"found {existing_rows}"
            )
        file_mode = "a"
        print(f"Resuming hybrid from question {start_index}/{total} (appending to {rag_output})")

    with rag_output.open(file_mode, encoding="utf-8") as out_f:
        for i, item in enumerate(questions[start_index - 1 :], start=start_index):
            q = item["question"]
            category = item.get("category", "Unknown")
            q_emb = embed_query(co_client, q)
            result = collection.query(
                query_embeddings=q_emb,
                n_results=min(RETRIEVE_K, 200)
            )
            docs = result.get("documents", [[]])[0]
            docs_trimmed = [trim_doc(d) for d in docs]
            docs_for_context = rerank_with_cross_encoder(q, docs_trimmed, top_n=TOP_K)
            context = "\n\n".join(docs_for_context)
            user_prompt = format_user_prompt(q, context=context)
            answer = call_ollama(model_name, HYBRID_SYSTEM_RULES, user_prompt)
            answer = finalize_answer(answer)

            row = {
                "id": item.get("id", i),
                "category": category,
                "model": f"{model_name}-rag-local-{PROMPT_VERSION}",
                "prompt": q,
                "retrieved_context": context,
                "ai_response": answer,
            }
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[rag {i}/{total}] done")
    print(f"RAG output saved to: {rag_output}")

#parses the arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description=f"Run local Ollama baseline and/or hybrid RAG for the {EVAL_PROMPT_COUNT}-question set."
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "rag", "both"],
        default="both",
        help="Which run mode to execute.",
    )
    parser.add_argument(
        "--baseline-model",
        default=DEFAULT_BASELINE_MODEL,
        help="Baseline model (parametric-only). Example: llama3.1:8b-instruct-q4_K_M",
    )
    parser.add_argument(
        "--hybrid-model",
        default=DEFAULT_HYBRID_MODEL,
        help="Hybrid RAG model (typically fine-tuned). Example: senco-ft:latest",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Deprecated alias. If provided, overrides both baseline and hybrid models.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed for reproducibility where supported.",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=0,
        help="Optional cap for question count (0 = all questions).",
    )
    parser.add_argument(
        "--baseline-output",
        default="",
        help="Optional override path for baseline JSONL output.",
    )
    parser.add_argument(
        "--hybrid-output",
        default="",
        help="Optional override path for hybrid RAG JSONL output.",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Optional run identifier for reproducibility artifacts.",
    )
    parser.add_argument(
        "--run-log-root",
        default=str(BASE_DIR / "data" / "07_run_logs"),
        help="Directory where run manifest/summary files are written.",
    )
    parser.add_argument(
        "--rag-start-index",
        type=int,
        default=1,
        help="1-based question index to start hybrid RAG (default 1). Use >1 to append to existing JSONL.",
    )
    return parser.parse_args()

#makes the run id
def _make_run_id(explicit: str) -> str:
    if explicit:
        return explicit
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    return f"phase4b_{stamp}"

#writes the run manifest - creates the directory and writes the payload to the file
def _write_run_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

#counts the jsonl rows
def _count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())

#sets the global seed
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

#main function to run the script
def main():
    load_dotenv()
    args = parse_args()
    run_started = datetime.now(timezone.utc)
    run_start_monotonic = time.perf_counter()
    run_id = _make_run_id(args.run_id)
    run_dir = Path(args.run_log_root) / run_id
    manifest_path = run_dir / "run_manifest.json"
    summary_path = run_dir / "run_summary.json"
    #sets the baseline model and hybrid model   
    if args.model:
        baseline_model = args.model
        hybrid_model = args.model
    else:
        baseline_model = args.baseline_model
        hybrid_model = args.hybrid_model
    #sets the global seed
    set_global_seed(args.seed)
    baseline_output = Path(args.baseline_output) if args.baseline_output else BASELINE_OUTPUT
    #sets the rag output
    rag_output = Path(args.hybrid_output) if args.hybrid_output else RAG_OUTPUT
    #writes the run manifest
    _write_run_manifest(
        manifest_path,
        {
            "run_id": run_id,
            "phase": "phase4b_local_ollama_rag",
            "started_at_utc": run_started.isoformat(),
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "mode": args.mode,
            "baseline_model": baseline_model,
            "hybrid_model": hybrid_model,
            "prompt_version": PROMPT_VERSION,
            "inference_tag": INFERENCE_TAG,
            "seed": args.seed,
            "retrieval_strategy": "dense_retrieve_then_cross_encoder_rerank",
            "retrieve_k": RETRIEVE_K,
            "max_answer_sentences": MAX_ANSWER_SENTENCES,
            "answer_sentence_guardrail": "trim_answer_to_max_sentences",
            "cross_encoder_model": CROSS_ENCODER_MODEL,
            "questions_file": str(QUESTIONS_JSONL),
            "baseline_output": str(baseline_output),
            "rag_output": str(rag_output),
            "chroma_path": str(CHROMA_PATH),
            "collection_name": COLLECTION_NAME,
            "top_k": TOP_K,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_tokens": MAX_TOKENS,
            "rag_start_index": args.rag_start_index,
        },
    )

    #loads the questions
    questions = load_questions(QUESTIONS_JSONL)
    #sets the max questions
    if args.max_questions > 0:
        questions = questions[: args.max_questions]
    print(f"Loaded {len(questions)} questions from {QUESTIONS_JSONL}")
    print(f"Baseline model: {baseline_model}")
    print(f"Hybrid model: {hybrid_model}")
    print(f"Random seed: {args.seed}")
    print(f"Run artifacts directory: {run_dir}")
    print(f"Run manifest saved to: {manifest_path}")

    if args.mode in ("baseline", "both"):
        run_baseline(questions, baseline_model, baseline_output)
    if args.mode in ("rag", "both"):
        run_rag(questions, hybrid_model, rag_output, start_index=args.rag_start_index)

    #tries to sync the phase4 outputs
    try:
        from db_utils import sync_phase4_outputs
        #syncs the phase4 outputs
        sync_phase4_outputs(questions, baseline_output, rag_output, PROMPT_VERSION)
        print("PostgreSQL: synced questions and responses (if DB is running).")
    except Exception as exc:
        print(f"PostgreSQL sync skipped (optional): {exc}")

    #sets the finished at time
    finished_at = datetime.now(timezone.utc)
    #sets the elapsed seconds
    elapsed_seconds = round(time.perf_counter() - run_start_monotonic, 3)
    #sets the summary
    summary = {
        "run_id": run_id,
        "finished_at_utc": finished_at.isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "mode": args.mode,
        "baseline_model": baseline_model,
        "hybrid_model": hybrid_model,
        "seed": args.seed,
        "questions_count": len(questions),
        "baseline_rows": _count_jsonl_rows(baseline_output),
        "hybrid_rag_rows": _count_jsonl_rows(rag_output),
        "baseline_output_exists": baseline_output.exists(),
        "hybrid_output_exists": rag_output.exists(),
    }
    #writes the run summary
    _write_run_manifest(summary_path, summary)
    print(f"Run summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
