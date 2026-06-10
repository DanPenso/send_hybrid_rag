# composer ide was used to support the researcher in the development of this script

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Final dissertation benchmark: stratified evaluation prompts (baseline + hybrid = 2 × this many rows)
EVAL_PROMPT_COUNT = 250
EVAL_PROMPT_VERSION = "v10"
# Output format / retrieval stack version (v14 = staff-facing prompts, question-first, 3.1 baseline)
INFERENCE_TAG = "v14"

# Evaluation artifact tags (must match phase5 notebooks / 5d when using PostgreSQL)
RAGAS_EVAL_TAG = "v14"
LLM_JUDGE_EVAL_TAG = "v14_llm_judge_notebook"
# Stream C drift artifacts: data/06_evaluation/stream_c_r/{STREAM_C_TAG}/
STREAM_C_TAG = INFERENCE_TAG


def _results_dir() -> Path:
    return BASE_DIR / "data" / "05_results"


RESULTS_DIR = _results_dir()

QUESTIONS_JSONL = RESULTS_DIR / f"questions_{EVAL_PROMPT_COUNT}_{EVAL_PROMPT_VERSION}.jsonl"
BASELINE_JSONL = RESULTS_DIR / f"baseline_results_{EVAL_PROMPT_COUNT}_local_ollama_{INFERENCE_TAG}.jsonl"
HYBRID_JSONL = RESULTS_DIR / f"hybrid_rag_results_{EVAL_PROMPT_COUNT}_local_ollama_{INFERENCE_TAG}.jsonl"
