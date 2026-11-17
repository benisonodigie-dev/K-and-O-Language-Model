#!/usr/bin/env python3
"""
Fine-tune Gemma (or another small LM) with LoRA for Urgent vs Routine SMS classification.

Gemma models on Hugging Face are *gated*. You must:
  1. Create a free account at https://huggingface.co
  2. Request access on the model page (e.g. https://huggingface.co/google/gemma-2-2b-it)
     and accept the license
  3. Create an access token: https://huggingface.co/settings/tokens  (read is enough)
  4. Authenticate before training (see below)

Authenticate in any one of these ways:
  export HF_TOKEN=hf_xxxxxxxx
  huggingface-cli login
  python src/gemma_lora_train.py --token hf_xxxxxxxx

Usage:
  python src/gemma_lora_train.py
  python src/gemma_lora_train.py --model google/gemma-2-2b-it --epochs 2
  python src/gemma_lora_train.py --model google/gemma-2-9b-it --epochs 1 --batch-size 1
  # Non-gated fallback for local testing without HF approval:
  python src/gemma_lora_train.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
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

PROMPT_TEMPLATE = (
    "Classify this maternal/infant health SMS as Urgent or Routine.\n"
    "SMS: {text}\n"
    "Label:"
)

# Models that require HF license acceptance
GATED_PREFIXES = ("google/gemma", "google/gemma-2", "google/gemma-3")


def hf_login(token: Optional[str] = None) -> None:
    """Log in to Hugging Face Hub so gated models can be downloaded."""
    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    try:
        from huggingface_hub import login, HfFolder
    except ImportError:
        logger.warning("huggingface_hub not installed; pip install huggingface_hub")
        return

    if token:
        login(token=token, add_to_git_credential=False)
        logger.info("Logged in to Hugging Face with provided token")
        return

    # Already logged in?
    existing = None
    try:
        existing = HfFolder.get_token()
    except Exception:
        pass
    if existing:
        logger.info("Using existing Hugging Face credentials")
        return

    logger.warning(
        "No HF token found. Gated models (Gemma) will fail.\n"
        "  1. Accept the license on the model page\n"
        "  2. Create a token at https://huggingface.co/settings/tokens\n"
        "  3. Run:  export HF_TOKEN=hf_...\n"
        "     or:  python src/gemma_lora_train.py --token hf_...\n"
        "     or:  huggingface-cli login"
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
    token: Optional[str] = None,
) -> Path:
    hf_login(token)

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
            "  pip install torch transformers peft datasets accelerate huggingface_hub\n"
            f"Original error: {e}"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s | Model: %s", device, model_name)

    if any(model_name.startswith(p) for p in GATED_PREFIXES):
        logger.info(
            "Note: %s is gated. Ensure you accepted the license and are logged in.",
            model_name,
        )

    ds = build_hf_dataset(DATA_DIR / "train.jsonl", DATA_DIR / "test.jsonl")
    if max_train_samples and max_train_samples < len(ds["train"]):
        ds["train"] = ds["train"].shuffle(seed=42).select(range(max_train_samples))
    logger.info("Train size=%d  Test size=%d", len(ds["train"]), len(ds["test"]))

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, token=token or True
        )
    except Exception as e:
        msg = str(e)
        if "gated" in msg.lower() or "restricted" in msg.lower() or "401" in msg or "403" in msg:
            raise SystemExit(
                "\nCannot download gated model '{model}'.\n\n"
                "Fix:\n"
                "  1. Open https://huggingface.co/{model} and accept the license\n"
                "  2. Create a token: https://huggingface.co/settings/tokens\n"
                "  3. Authenticate:\n"
                "       export HF_TOKEN=hf_your_token_here\n"
                "       # or\n"
                "       python src/gemma_lora_train.py --token hf_your_token_here\n"
                "       # or\n"
                "       huggingface-cli login\n\n"
                "Or use a non-gated model for testing:\n"
                "  python src/gemma_lora_train.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0\n"
                .format(model=model_name)
            ) from e
        raise

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            token=token or True,
        )
    except Exception as e:
        msg = str(e)
        if "gated" in msg.lower() or "restricted" in msg.lower() or "401" in msg or "403" in msg:
            raise SystemExit(
                f"\nCannot download gated model '{model_name}'. "
                "Accept the license and pass --token / HF_TOKEN (see script docstring).\n"
                f"Original error: {e}"
            ) from e
        raise

    model.config.pad_token_id = tokenizer.pad_token_id

    # Target modules: Gemma / LLaMA-style; fall back if names differ
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules=target_modules,
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
    # transformers version compatibility
    eval_strategy_key = "eval_strategy"
    try:
        from transformers import TrainingArguments as _TA
        import inspect
        if "evaluation_strategy" in inspect.signature(_TA.__init__).parameters:
            eval_strategy_key = "evaluation_strategy"
    except Exception:
        pass

    ta_kwargs = dict(
        output_dir=str(MODEL_DIR / "gemma_lora_runs"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="urgent_recall",
        greater_is_better=True,
        logging_steps=20,
        report_to=[],
        fp16=(device == "cuda"),
        remove_unused_columns=False,
    )
    ta_kwargs[eval_strategy_key] = "epoch"
    args = TrainingArguments(**ta_kwargs)

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
    parser = argparse.ArgumentParser(
        description="Fine-tune Gemma (or other LM) with LoRA for SMS triage"
    )
    parser.add_argument(
        "--model",
        default="google/gemma-2-2b-it",
        help="HF model id (Gemma is gated; use TinyLlama/... if you lack access)",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face access token (or set HF_TOKEN env var)",
    )
    args = parser.parse_args()
    train(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lora_r=args.lora_r,
        max_train_samples=args.max_train_samples,
        token=args.token,
    )


if __name__ == "__main__":
    main()
