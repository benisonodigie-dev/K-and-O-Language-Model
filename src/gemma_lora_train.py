#!/usr/bin/env python3
"""
Fine-tune a small Gemma model with LoRA for Urgent vs Routine SMS classification.
Matches the K and O Language Model capstone proposal:
  - Gemma (small) + LoRA
  - High-recall priority for Urgent
  - Multilingual synthetic SMS (EN, Pidgin, Hausa, Yoruba, Igbo, Edo)
  - No PII in training data

Usage:
  python src/gemma_lora_train.py
  python src/gemma_lora_train.py --model google/gemma-2-2b-it --epochs 2

Requires (optional GPU):
  pip install torch transformers peft datasets accelerate bitsandbytes trl
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("gemma_lora")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
ADAPTER_DIR = MODEL_DIR / "gemma_lora_adapter"

LABEL2ID = {"Routine": 0, "Urgent": 1}
ID2LABEL = {0: "Routine", 1: "Urgent"}

# Instruction template (works for causal LM classification via generation or last-token)
PROMPT_TEMPLATE = (
    "Classify this maternal/infant health SMS as Urgent or Routine.\n"
    "SMS: {text}\n"
    "Label:"
)


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_hf_dataset(train_path: Path, test_path: Path):
    from datasets import Dataset, DatasetDict

    def to_records(rows):
        return {
            "text": [r["text"] for r in rows],
            "label": [LABEL2ID[r["label"]] for r in rows],
            "prompt": [PROMPT_TEMPLATE.format(text=r["text"]) for r in rows],
        }

    train_rows = load_jsonl(train_path)
    test_rows = load_jsonl(test_path)
    return DatasetDict({
        "train": Dataset.from_dict(to_records(train_rows)),
        "test": Dataset.from_dict(to_records(test_rows)),
    })


def train(
    model_name: str = "google/gemma-2-2b-it",
    epochs: int = 2,
    batch_size: int = 4,
    lr: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    max_length: int = 192,
    max_train_samples: Optional[int] = None,
) -> Path:
    """
    Fine-tune Gemma with LoRA for binary urgency classification.
    Saves adapter weights to models/gemma_lora_adapter.
    """
    try:
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            TrainingArguments,
            Trainer,
            DataCollatorWithPadding,
        )
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError as e:
        raise SystemExit(
            "Missing dependencies. Install with:\n"
            "  pip install torch transformers peft datasets accelerate\n"
            f"Original error: {e}"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s | Model: %s", device, model_name)

    ds = build_hf_dataset(DATA_DIR / "train.jsonl", DATA_DIR / "test.jsonl")
    if max_train_samples and max_train_samples < len(ds["train"]):
        ds["train"] = ds["train"].shuffle(seed=42).select(range(max_train_samples))
    logger.info("Train size=%d  Test size=%d", len(ds["train"]), len(ds["test"]))

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    tokenized = ds.map(tokenize, batched=True, remove_columns=["text", "prompt"])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        import numpy as np
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        # Urgent = class 1
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        acc = float((preds == labels).mean())
        return {
            "accuracy": acc,
            "urgent_precision": prec,
            "urgent_recall": rec,
            "urgent_f1": f1,
        }

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(MODEL_DIR / "gemma_lora_runs"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="urgent_recall",
        greater_is_better=True,
        logging_steps=20,
        report_to=[],
        fp16=(device == "cuda"),
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting LoRA fine-tuning...")
    trainer.train()
    metrics = trainer.evaluate()
    logger.info("Eval metrics: %s", metrics)

    # Save adapter + tokenizer + label map
    model.save_pretrained(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))
    meta = {
        "base_model": model_name,
        "label2id": LABEL2ID,
        "id2label": {str(k): v for k, v in ID2LABEL.items()},
        "max_length": max_length,
        "metrics": {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
    }
    with open(ADAPTER_DIR / "triage_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info("Adapter saved to %s", ADAPTER_DIR)
    return ADAPTER_DIR


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Gemma with LoRA for SMS triage")
    parser.add_argument("--model", default="google/gemma-2-2b-it", help="Base HF model id")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--max-train-samples", type=int, default=None)
    args = parser.parse_args()
    train(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lora_r=args.lora_r,
        max_train_samples=args.max_train_samples,
    )


if __name__ == "__main__":
    main()
