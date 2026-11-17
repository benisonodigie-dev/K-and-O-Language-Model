#!/usr/bin/env python3
"""
Inference helper for the Gemma-LoRA urgency classifier.
Loads the fine-tuned LoRA adapter if present; otherwise reports unavailable.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Any

logger = logging.getLogger("gemma_lora_infer")

ROOT = Path(__file__).resolve().parents[1]
ADAPTER_DIR = ROOT / "models" / "gemma_lora_adapter"

_model = None
_tokenizer = None
_meta: Dict[str, Any] = {}
_load_error: Optional[str] = None


def adapter_available() -> bool:
    return (ADAPTER_DIR / "adapter_config.json").exists() or (
        ADAPTER_DIR / "triage_meta.json"
    ).exists()


def load_model(force: bool = False) -> bool:
    """Load LoRA adapter + base model. Returns True on success."""
    global _model, _tokenizer, _meta, _load_error
    if _model is not None and not force:
        return True
    if not adapter_available():
        _load_error = f"No adapter found at {ADAPTER_DIR}. Run gemma_lora_train.py first."
        logger.warning(_load_error)
        return False
    try:
        import os
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from peft import PeftModel

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        if token:
            try:
                from huggingface_hub import login
                login(token=token, add_to_git_credential=False)
            except Exception:
                pass

        meta_path = ADAPTER_DIR / "triage_meta.json"
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                _meta = json.load(f)
        base_model = _meta.get("base_model", "google/gemma-2-2b-it")
        id2label = {int(k): v for k, v in _meta.get("id2label", {"0": "Routine", "1": "Urgent"}).items()}
        label2id = _meta.get("label2id", {"Routine": 0, "Urgent": 1})

        _tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER_DIR), trust_remote_code=True, token=token or True)
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token

        base = AutoModelForSequenceClassification.from_pretrained(
            base_model,
            num_labels=2,
            id2label=id2label,
            label2id=label2id,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            token=token or True,
        )
        _model = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
        _model.eval()
        if torch.cuda.is_available():
            _model = _model.cuda()
        _load_error = None
        logger.info("Gemma-LoRA loaded from %s", ADAPTER_DIR)
        return True
    except Exception as e:
        _load_error = str(e)
        _model = None
        _tokenizer = None
        logger.error("Failed to load Gemma-LoRA: %s", e)
        return False


def predict_urgency(text: str) -> Dict[str, Any]:
    """
    Returns:
      {
        "prediction": "Urgent"|"Routine",
        "confidence": float,
        "probs": {"Routine": float, "Urgent": float},
        "available": bool,
        "error": str|None
      }
    """
    result = {
        "prediction": None,
        "confidence": None,
        "probs": {},
        "available": False,
        "error": None,
    }
    if not text or not str(text).strip():
        result["error"] = "Empty text"
        return result

    if _model is None:
        ok = load_model()
        if not ok:
            result["error"] = _load_error or "Model not available"
            return result

    try:
        import torch
        max_length = int(_meta.get("max_length", 192))
        inputs = _tokenizer(
            text.strip(),
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        )
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            logits = _model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()
        # label order: 0=Routine, 1=Urgent
        pred_id = int(probs[1] >= probs[0])
        label = "Urgent" if pred_id == 1 else "Routine"
        result.update({
            "prediction": label,
            "confidence": float(max(probs)),
            "probs": {"Routine": float(probs[0]), "Urgent": float(probs[1])},
            "available": True,
            "error": None,
        })
        return result
    except Exception as e:
        result["error"] = str(e)
        return result


if __name__ == "__main__":
    print("Adapter available:", adapter_available())
    if adapter_available():
        print(predict_urgency("My wife is bleeding heavily in Kano. Urgent help!"))
    else:
        print("Train first: python src/gemma_lora_train.py")
