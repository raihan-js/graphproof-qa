#!/usr/bin/env python3
"""LoRA fine-tune Qwen2.5-1.5B-Instruct on GraphProof training pairs.

Usage:
  PYTHONPATH=src python scripts/train_lora.py --data data/train/dsl.jsonl --out data/adapters/dsl
  PYTHONPATH=src python scripts/train_lora.py --data data/train/direct.jsonl --out data/adapters/direct
Hyperparameters per manifest: r=16, alpha=32, dropout=0.05, 3 epochs, lr=2e-4.
"""
import argparse
import json
import os

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def load_rows(path: str) -> Dataset:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return Dataset.from_list(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="data/models/Qwen--Qwen2.5-1.5B-Instruct")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--resume", default=None,
                    help="Resume from checkpoint dir (e.g. data/adapters/dsl/checkpoint-1500)")
    ap.add_argument("--no-grad-ckpt", action="store_true",
                    help="Disable gradient checkpointing (faster, more VRAM)")
    args = ap.parse_args()

    print(f"Loading tokenizer + model from {args.model}...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="right")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto",
    )
    model.config.use_cache = False

    print("Applying chat template...", flush=True)
    ds = load_rows(args.data)

    def tokenize(row):
        text = tok.apply_chat_template(row["messages"], tokenize=False)
        # NOTE: no manual "labels" — DataCollatorForLanguageModeling(mlm=False)
        # clones padded input_ids into labels itself. Setting labels here
        # breaks dynamic padding (ragged lists can't tensorize).
        return tok(text, truncation=True, max_length=1024)

    ds = ds.map(tokenize, remove_columns=["messages", "hop"])

    print("Attaching LoRA (r=16, alpha=32)...", flush=True)
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj"],
                     task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=not args.no_grad_ckpt,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        bf16=True,
        optim="adamw_torch",
        seed=42,
        report_to="none",
    )
    Trainer(model=model, args=targs, train_dataset=ds,
            data_collator=DataCollatorForLanguageModeling(tok, mlm=False)
            ).train(resume_from_checkpoint=args.resume)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"Saved adapter -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
