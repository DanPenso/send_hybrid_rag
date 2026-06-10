#In the Supervised-Fie-Tuning stage of the project, Composer through a Cursor IDE supported generation of this script.

"""
Phase 3a: Supervised fine-tuning (LoRA) on synthetic prompt/completion pairs.

Reads: data/03_synthetic_eval/bedrock_ready_llama.jsonl (Phase 2 output).
Writes: PEFT adapter weights to `--output` (default `data/03_model_adapters/lora_send`) for Phase 3b merge.

Requires: GPU strongly recommended, Hugging Face access to the base model, trl/datasets/accelerate.
Env: HF_TOKEN if the base model is gated; BASE_MODEL_ID optional override.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer

#sets the root directory for the script
_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")

#formats the dataset into the JSONL file
def _format_dataset(tokenizer, examples: dict) -> dict:
    texts = []
    for prompt, completion in zip(examples["prompt"], examples["completion"]):
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ]
        texts.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        )
    return {"text": texts}
    

#main function to run the script
def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3a: LoRA SFT on bedrock_ready_llama.jsonl")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("../data/03_synthetic_eval/bedrock_ready_llama.jsonl"),
        help="JSONL with prompt/completion keys.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("../data/03_model_adapters/lora_send"),
        help="Directory to save PEFT adapter.",
    )
    parser.add_argument(
        "--base-model",
        default=os.environ.get(
            "BASE_MODEL_ID", "meta-llama/Meta-Llama-3.1-8B-Instruct"
        ),
        help="Hugging Face model id for the student.",
    )
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="QLoRA: 4-bit NF4 base weights (recommended for 8B on ~8–12GB VRAM). Requires CUDA + bitsandbytes.",
    )
    args = parser.parse_args()

    if not args.data.is_file():
        raise FileNotFoundError(f"Training data not found: {args.data.resolve()}")

    args.output.mkdir(parents=True, exist_ok=True)

#loads the Hugging Face token for the model
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    hf_kw = {"token": hf_token} if hf_token else {}

#loads the tokenizer for the model
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, **hf_kw)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

#loads the dataset for the model
    ds = load_dataset("json", data_files=str(args.data), split="train")
    ds = ds.map(
        lambda batch: _format_dataset(tokenizer, batch),
        batched=True,
    )
    ds = ds.remove_columns(
        [c for c in ds.column_names if c != "text"]
    )

#loads the model for the model
    if args.load_in_4bit:
        #checks if the CUDA GPU is available
        if not torch.cuda.is_available():
            raise ValueError("--load-in-4bit requires a CUDA GPU.")
        #loads the BitsAndBytesConfig for the model
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
       #loads the model for the model       
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            max_memory={0: "7000MiB", "cpu": "48GiB"},
            **hf_kw,
        )
        model = prepare_model_for_kbit_training(model)
        use_bf16 = True
        use_fp16 = False
    else:
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        use_bf16 = torch.cuda.is_available() and torch_dtype == torch.bfloat16
        use_fp16 = torch.cuda.is_available() and not use_bf16
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch_dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            **hf_kw,
        )
#loads the PEFT config for the model
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
#loads the training arguments for the model
    training_args = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="epoch",
        fp16=use_fp16,
        bf16=use_bf16,
        gradient_checkpointing=args.load_in_4bit,
        report_to="none",
        optim="adamw_torch",
    )

    #loads the trainer for the model
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        peft_config=peft_config,
        args=training_args,
        packing=False,
    )

    #trains the model
    trainer.train()
    #saves the model
    trainer.model.save_pretrained(str(args.output))
    #saves the tokenizer
    tokenizer.save_pretrained(str(args.output))
    #prints the path to the saved model
    print(f"Adapter saved to: {args.output.resolve()}")

#main function to run the script
if __name__ == "__main__":
    main()
