#In the Supervised-Fine-Tuning stage of the project, Composer through a Cursor IDE supported generation of this script.

"""
Phase 3c: GGUF conversion and Ollama registration (manual steps helper).

This project does not bundle llama.cpp. After Phase 3b you have merged HF weights.

Typical flow:
  1) Clone/update llama.cpp and run convert_hf_to_gguf.py on the merged HF directory.
  2) Point a Modelfile at the .gguf and run `ollama create`.

Run this script with no args (or --merged-hf-dir / --gguf-out) to print concrete commands
for your machine. Optional: --write-modelfile PATH
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
#sets the root directory for the script
_ROOT = Path(__file__).resolve().parent.parent
#builds the modelfile for the script
#takes the gguf path, temperature, top p, and num predict and builds the modelfile  
def build_modelfile(
    gguf_path: str,
    temperature: str = "0",
    top_p: str = "1",
    num_predict: str = "256",
) -> str:
    p = Path(gguf_path)
    from_line = str(p.as_posix()) if p.is_absolute() else f"./{p.as_posix()}"
    return (
        f"FROM {from_line}\n"
        f"PARAMETER temperature {temperature}\n"
        f"PARAMETER top_p {top_p}\n"
        f"PARAMETER num_predict {num_predict}\n"
    )

#main function to run the script
def main() -> None:
    default_merged = _ROOT / "data" / "03_model_merged" / "llama31_senco_merged"
    default_gguf = _ROOT / "local_models" / "senco-ft-f16.gguf"
    #sets the parser for the script
    parser = argparse.ArgumentParser(description="Phase 3c: GGUF + Ollama helper.")
    #sets the merged hf directory for the script
    parser.add_argument(
        "--merged-hf-dir",
        type=Path,
        default=default_merged,
        help="Merged HF checkpoint from Phase 3b (directory with config.json, tokenizer).",
    )
    #sets the gguf output directory for the script
    parser.add_argument(
        "--gguf-out",
        type=Path,
        default=default_gguf,
        help="Where to write the .gguf after conversion (used in Ollama creation).",
    )
    #sets the gguf path for the script
    parser.add_argument(
        "--gguf-path",
        default="",
        help="Override path for Modelfile FROM line only (default: same as --gguf-out).",
    )
    #sets the ollama model name for the script
    parser.add_argument(
        "--ollama-model-name",
        default="senco-ft",
        help="Name for: ollama create <name> -f Modelfile",
    )
    #sets the write modelfile for the script
    parser.add_argument(
        "--write-modelfile",
        default="",
        help="If set, write Modelfile to this path (e.g. local_models/Modelfile).",
    )
    args = parser.parse_args()
    #sets the merged hf directory for the script
    merged = args.merged_hf_dir.resolve()
    #sets the gguf output directory for the script
    gguf_out = args.gguf_out.resolve()
    #sets the gguf path for the script
    gguf_for_modelfile = Path(args.gguf_path).resolve() if args.gguf_path else gguf_out

    #checks if the merged hf directory exists
    if not merged.is_dir():
        print(f"Warning: merged HF dir not found (run Phase 3b first): {merged}", file=sys.stderr)

    #creates the gguf output directory
    gguf_out.parent.mkdir(parents=True, exist_ok=True)
    #builds the modelfile for the script
    #prints the phase 3c message
    mf = build_modelfile(str(gguf_for_modelfile))
    #prints the phase 3c message
    print("=== Phase 3c: GGUF + Ollama ===\n")
    #prints the install llama.cpp message
    print("1) Install / update llama.cpp (you need convert_hf_to_gguf.py from a recent main).")
    #prints the install llama.cpp message
    print("   https://github.com/ggerganov/llama.cpp\n")
    #prints the convert merged hf to gguf message
    print("2) Convert merged HF -> GGUF (run from your llama.cpp clone; Python 3.10+ with")
    #prints the convert merged hf to gguf message
    print("   `pip install -r requirements.txt` inside llama.cpp if the script asks).\n")
    print("1) Install / update llama.cpp (you need convert_hf_to_gguf.py from a recent main).")
    print("   https://github.com/ggerganov/llama.cpp\n")
    print("2) Convert merged HF -> GGUF (run from your llama.cpp clone; Python 3.10+ with")
    print("   `pip install -r requirements.txt` inside llama.cpp if the script asks).\n")
    # Use quoted paths for Windows shells
    print("   Example (adjust LLAMA_CPP_ROOT to your clone):\n")
    print(
        f'   python "%LLAMA_CPP_ROOT%\\convert_hf_to_gguf.py" '
        f'"{merged}" --outfile "{gguf_out}" --outtype f16'
    )
    print("\n   PowerShell one-liner if $env:LLAMA_CPP_ROOT is set:\n")
    print(
        f'   python "$env:LLAMA_CPP_ROOT\\convert_hf_to_gguf.py" '
        f'"{merged}" --outfile "{gguf_out}" --outtype f16'
    )
    print("\n   If conversion fails, update llama.cpp; Llama 3.x support moves quickly.\n")
    print("--- Modelfile (after GGUF exists) ---")
    print(mf)
    print("--- Ollama ---")
    print(f"  cd \"{gguf_out.parent}\"")
    print("  # Save the Modelfile text above as Modelfile in this folder (or use --write-modelfile).")
    print(f"  ollama create {args.ollama_model_name} -f Modelfile")
    print(f"  ollama run {args.ollama_model_name}")
    print("\nDocs: https://github.com/ggerganov/llama.cpp/blob/master/convert_hf_to_gguf.py")
    #writes the modelfile to the output directory
    if args.write_modelfile:
        out = Path(args.write_modelfile)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(mf, encoding="utf-8")
        print(f"\nWrote Modelfile: {out.resolve()}")

#main function to run the script
if __name__ == "__main__":
    main()
