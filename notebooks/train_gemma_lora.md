# Train Gemma-LoRA for SMS Triage

## 1. Install

```bash
pip install torch transformers peft datasets accelerate
```

## 2. Train

```bash
cd sms_triage
python src/gemma_lora_train.py --model google/gemma-2-2b-it --epochs 2 --batch-size 4
```

Optional: smaller run for testing

```bash
python src/gemma_lora_train.py --max-train-samples 200 --epochs 1
```

## 3. Output

- Adapter: `models/gemma_lora_adapter/`
- Metrics prioritise **Urgent recall**

## 4. Inference

Automatic via `predict_message()` (high-recall ensemble with keywords).

```python
from triage_model import predict_message
print(predict_message("Pikin dey convulsing for Wudil"))
```
