#In the Supervised-Fie-Tuning stage of the project, Composer through a Cursor IDE supported generation of this script.

"""
Phase 3b: Merge a trained LoRA adapter into the base Hugging Face model.

https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct is the base model for the script.

Run after Phase 3a. Env: BASE_MODEL_ID (must match Phase 3a base), ADAPTER_DIR (3a output), OUT_DIR (merged output).
Also set HF_TOKEN in .env if the base model is gated.

Loads the full base weights on CPU; ensure enough free RAM (~16GB+ for 8B in fp16).

Next: Phase 3c (GGUF + Ollama) before using the tuned model in Phase 4b.
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

#sets the root directory for the script
_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")

#main function to run the script    
def main():
    #sets the base model id for the script
    base_model_id = os.environ.get("BASE_MODEL_ID")
    #sets the adapter directory for the script
    adapter_dir = os.environ.get("ADAPTER_DIR")
    #sets the output directory for the script
    out_dir = os.environ.get("OUT_DIR")
    #checks if the base model id is set
    if not base_model_id:
        raise ValueError("BASE_MODEL_ID env var is required.")
    #checks if the adapter directory is set
    if not adapter_dir:
        raise ValueError("ADAPTER_DIR env var is required.")
    #checks if the output directory is set
    if not out_dir:
        raise ValueError("OUT_DIR env var is required.")

#sets the adapter path for the script
    adapter_path = Path(adapter_dir)
    #sets the output path for the script
    output_path = Path(out_dir)
    #checks if the adapter path exists
    if not adapter_path.exists():
        raise FileNotFoundError(f"ADAPTER_DIR not found: {adapter_path}")
    #loads the Hugging Face token for the model
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    hf_kw = {"token": hf_token} if hf_token else {}
    #prints the base model id
    print(f"Loading base model: {base_model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype="auto",
        device_map="cpu",
        **hf_kw,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, **hf_kw)
    #prints the adapter path
    print(f"Loading LoRA adapter: {adapter_path}")
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_path))
    #prints the merging message
    print("Merging adapter into base model...")
    merged_model = peft_model.merge_and_unload()
    #creates the output directory
    output_path.mkdir(parents=True, exist_ok=True)
    #prints the output path
    print(f"Saving merged model to: {output_path}")
    #saves the merged model
    merged_model.save_pretrained(str(output_path), safe_serialization=False)
    #saves the tokenizer
    tokenizer.save_pretrained(str(output_path))
    #prints the merge complete message

    print("Merge complete.")

#main function to run the script
if __name__ == "__main__":
    main()
