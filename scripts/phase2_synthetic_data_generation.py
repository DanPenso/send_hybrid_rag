#In this stage of the project, Composer through a Cursor IDE supported generation of this script.

#script to generate synthetic SEND instruction-response training pairs using Anthropic Claude models

import argparse
import json
import os
import time
from typing import List

from anthropic import Anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


INPUT_DIR = "../data/02_extracted_text/"

#sets the checkpoint file for the synthetic data generation
CHECKPOINT_FILE = "../data/03_synthetic_eval/phase2_checkpoint.json"

# sets the anthropic model correctly for the script
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"

#sets the files to process and the target number of instruction-response pairs to generate
FILES_TO_PROCESS = None
TARGET_INSTRUCTION_PAIRS = 1000
#sets the output file for the synthetic data generation
OUTPUT_FILE = "../data/03_synthetic_eval/bedrock_ready_llama.jsonl"

#validates the training pairs through Pydantic
#defines the QA pair schema for the synthetic data generation and sets the ideal response rules
class QAPair(BaseModel):
    user_query: str = Field(description="The question a teacher would ask")
    condition_tag: str = Field(description="The SEND condition (e.g., Autism)")
    query_type: str = Field(description="Macro or Micro")
    statutory_reference: str = Field(description="The source document section")
    ideal_response: str = Field(
        description=(
            "Third person only. Warm, supportive UK SENCo tone; non-directive. "
            "At most 3 short sentences (roughly 12–22 words each; ~45–70 words total). "
            "Succinct: one clear point per sentence, no filler."
        )
    )

#wrapper class for the training pairs
class QADataset(BaseModel):
    training_pairs: List[QAPair]

#parses the arguments for the script
def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic SEND QA training pairs.")
    parser.add_argument(
        "--teacher-model",
        default=DEFAULT_ANTHROPIC_MODEL,
        help="Anthropic Claude model name.",
    )
    return parser.parse_args()

#Removes the fence lines from the text - will cause an error downstream if not removed
def strip_code_fences(text: str) -> str:
    content = text.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        # Drop first and last fence lines when present.
        if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].startswith("```"):
            content = "\n".join(lines[1:-1]).strip()
    return content

#Ensures a fair split of the training pairs across the files so that larger files don't dominate the QA dataset
def _questions_for_file(file_index: int, total_files: int, target_total: int) -> int:
    """Spread target_total across total_files: first (remainder) files get one extra pair."""
    if total_files <= 0:
        return 0
    base = target_total // total_files
    extra = target_total % total_files
    return base + (1 if file_index < extra else 0)

#Counts the number of lines in the JSONL file to ensure one line per training pair
def _count_jsonl_lines(path: str) -> int:
    if not os.path.isfile(path):
        return 0
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def _bootstrap_pairs_from_line_count(
    line_count: int, sorted_files: list[str], target_total: int
) -> dict[str, int]:
    """
    Infer how many JSONL lines were written per source file (in sorted order) from line count only.
    Used when resuming after a crash with no checkpoint file.
    """
    n = len(sorted_files)
    out: dict[str, int] = {}
    remaining = line_count
    for i, fname in enumerate(sorted_files):
        expected = _questions_for_file(i, n, target_total)
        if remaining >= expected:
            out[fname] = expected
            remaining -= expected
        elif remaining > 0:
            out[fname] = remaining
            break
        else:
            break
    return out

#Loads the checkpoint file for the synthetic data generation
def _load_checkpoint(path: str) -> dict | None:
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)

#Saves the checkpoint file for the synthetic data generation
def _save_checkpoint(path: str, pairs_by_file: dict[str, int], target: int) -> None:
    payload = {
        "target_instruction_pairs": target,
        "pairs_by_file": pairs_by_file,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

#Resolves the pairs by file for the synthetic data generation
def _resolve_pairs_by_file(
    checkpoint_path: str,
    output_path: str,
    sorted_files: list[str],
    target_total: int,
) -> dict[str, int]:
    """Merge checkpoint with JSONL line count; re-bootstrap if inconsistent."""
    line_count = _count_jsonl_lines(output_path)
    data = _load_checkpoint(checkpoint_path)
    if data and isinstance(data.get("pairs_by_file"), dict):
        pb = {k: int(v) for k, v in data["pairs_by_file"].items()}
        if sum(pb.values()) == line_count:
            return pb
        print(
            f"Checkpoint line sum ({sum(pb.values())}) != JSONL lines ({line_count}); "
            "re-inferring progress from JSONL line count."
        )
    if line_count == 0:
        return {}
    return _bootstrap_pairs_from_line_count(line_count, sorted_files, target_total)

#Appends the training pairs to the JSONL file
def _append_jsonl(path: str, pairs: List[QAPair]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for item in pairs:
            row = {"prompt": item.user_query, "completion": item.ideal_response}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


#Rules to develop the training data through the Anthropic Claude model
def generate_training_data(
    text_content: str,
    filename: str,
    model_name: str,
    num_questions: int,
) -> List[QAPair]:
    client = Anthropic()
    system_prompt = (
        "You are an expert UK SENCo. You produce training data for a language model. "
        "Each ideal_response stays in the third person, sounds supportive and understanding, "
        "and explains how things work in plain language. Prefer descriptive, invitational wording "
        "over commands or checklists: avoid telling the teacher or parent what they 'must', "
        "'should', or 'need to' do unless the source text itself is duty-based; even then, "
        "keep the tone gentle and factual. "
        "Be succinct: at most 3 sentences, each short and direct (avoid long chains of clauses). "
        "Aim for roughly 45–70 words total per ideal_response; prioritise the minimum needed to "
        "answer the query faithfully—trim repetition and secondary detail if space is tight. "
        "Return strictly valid JSON only (no markdown, no prose outside the JSON)."
    )
    user_prompt = f"""
Source: {filename}
Text: {text_content[:12000]}

Generate exactly {num_questions} training pairs in this JSON schema:
{{
  "training_pairs": [
    {{
      "user_query": "question a teacher would ask",
      "condition_tag": "SEND condition label",
      "query_type": "Macro or Micro",
      "statutory_reference": "source section or evidence hint",
      "ideal_response": "third person; succinct; max 3 short sentences; ~45–70 words total"
    }}
  ]
}}

Rules for ideal_response:
- Third person only (no \"I\" / \"we\" as the SENCo speaking).
- At most three sentences. Each sentence should be compact (about 12–22 words). Total length about 45–70 words; if the topic is wide, cover essentials only—one main idea per sentence.
- Avoid very long sentences: no stuffing multiple facts into one line; split ideas across sentences or drop the least important detail.
- Less directive language: avoid imperatives (e.g. \"Share…\", \"Ensure…\", \"Make sure…\") and avoid lecturing the reader; prefer neutral explanation (e.g. \"The guidance describes…\", \"The inspection focuses on…\"). Minimise chains of \"must\" / \"should\" / \"need to\"; when duties exist, state them factually and sparingly.
- Tone: supportive and understanding—brief acknowledgement of worry is fine; avoid padded reassurance.
- Stay faithful to the excerpt; do not invent law or policy not implied by the text.
"""
    #sets the maximum tokens for the response
    max_tokens = min(16384, max(3072, num_questions * 320))
    resp = client.messages.create(
        model=model_name,
        max_tokens=max_tokens,
        temperature=0.0,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    #extracts the response text from the response
    text = "".join(block.text for block in resp.content if hasattr(block, "text"))
    payload = json.loads(strip_code_fences(text))
    validated = QADataset.model_validate(payload)
    return validated.training_pairs

#main function to run the script
def main():
    args = parse_args()
    model_name = args.teacher_model

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise ValueError("ANTHROPIC_API_KEY is missing in environment/.env.")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    available_files = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".txt"))
    if not available_files:
        print("No files found!")
        return

    if FILES_TO_PROCESS is None:
        files_to_read = available_files
    else:
        files_to_read = available_files[:FILES_TO_PROCESS]
    n_files = len(files_to_read)
    planned_total = sum(
        _questions_for_file(i, n_files, TARGET_INSTRUCTION_PAIRS) for i in range(n_files)
    )

#resolves the pairs by file for the synthetic data generation
    pairs_by_file: dict[str, int] = _resolve_pairs_by_file(
        CHECKPOINT_FILE, OUTPUT_FILE, files_to_read, TARGET_INSTRUCTION_PAIRS
    )
    existing_lines = _count_jsonl_lines(OUTPUT_FILE)
    if existing_lines:
        print(
            f"Resume: {existing_lines} line(s) on disk; "
            f"will generate missing pairs up to {planned_total} total."
        )

    print(
        f"Starting Generation with provider=anthropic, model={model_name}... "
        f"target={TARGET_INSTRUCTION_PAIRS} pairs across {n_files} file(s) (planned total {planned_total})."
    )

#loops through the files and generates the training pairs
    for idx, filename in enumerate(files_to_read):
        expected = _questions_for_file(idx, n_files, TARGET_INSTRUCTION_PAIRS)
        done = pairs_by_file.get(filename, 0)
        if done >= expected:
            print(f"Skip (done): {filename} ({done}/{expected} pairs)")
            continue
        need = expected - done
        print(f"Processing: {filename} ({need} pairs to generate, {done}/{expected} already)...")
        with open(os.path.join(INPUT_DIR, filename), encoding="utf-8") as f:
            text = f.read()
        try:
            data = generate_training_data(text, filename, model_name, need)
            if len(data) != need:
                print(f"   [WARN] Expected {need} pairs, got {len(data)}; saving what was returned.")
            _append_jsonl(OUTPUT_FILE, data)
            pairs_by_file[filename] = done + len(data)
            _save_checkpoint(CHECKPOINT_FILE, pairs_by_file, TARGET_INSTRUCTION_PAIRS)
            print(f"   [OK] Success: {len(data)} pairs generated.")
            time.sleep(1)
        except Exception as e:
            print(f"   [ERROR] {e}")

#counts the total number of training pairs in the output file
    total_out = _count_jsonl_lines(OUTPUT_FILE)
    print(f"\nTotal Training Pairs in output file: {total_out}")
    print(f"Output path: {OUTPUT_FILE}")
    print(f"Checkpoint: {CHECKPOINT_FILE}")

#main function to run the script
if __name__ == "__main__":
    main()