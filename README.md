# Urgent SMS Triage & Nearest-Care Referral

**Capstone Prototype - K and O Language Model**  
AI Research Foundations for Higher Education · Nigeria Cohort 2

## What this is

A lightweight system that:

1. Classifies incoming SMS / WhatsApp-style messages from caregivers as **Urgent** or **Routine** (maternal & infant health focus).
2. Extracts the mentioned community / location.
3. Suggests the **nearest Primary Health Centre** (currently focused on Kano State / Ovia North-East and nearby areas).

The model never diagnoses or gives medical advice. It only flags urgency and points to the nearest facility so Community Health Workers (CHWs) can prioritise and respond faster.

## Quick start

```bash
cd sms_triage

# (Already generated) synthetic data + trained model
python src/triage_model.py          # retrain + evaluate if needed

# Interactive demo
python demo/triage_cli.py
```

Example:

```
SMS> Please help, my wife is 8 months pregnant and bleeding heavily since morning. We are in Okada.

  -> Prediction : Urgent
  -> ML conf    : 0.87 (Urgent)
  -> Keyword    : Urgent
  -> Location   : Okada (Ovia North-East)
  -> Nearest PHC: Okada Primary Health Centre
  -> Action     : Urgent. Location: Okada (Ovia North-East). Nearest: Okada Primary Health Centre.
```

## Coverage

- **All 18 LGAs of Kano State** covered in the location gazetteer
- Coordinates enable Haversine distance and future map-based routing

## Project structure

```
sms_triage/
├── data/
│   ├── synthetic_sms_triage.jsonl   # 600 labelled messages
│   ├── train.jsonl
│   └── test.jsonl
├── models/
│   └── urgency_tfidf_logreg.joblib  # trained classifier
├── src/
│   ├── generate_synthetic_data.py
│   └── triage_model.py              # keywords + TF-IDF+LR + location matcher
├── demo/
│   └── triage_cli.py
└── README.md
```

## Model approach (current prototype)

| Component              | Implementation                                      |
|------------------------|-----------------------------------------------------|
| Urgency classifier     | TF-IDF (1-2 grams) + Logistic Regression (balanced) |
| High-recall ensemble   | ML prediction OR keyword rules -> Urgent             |
| Keyword baseline       | Regex patterns for bleeding, convulsions, etc.      |
| Location matching      | Simple gazetteer of communities -> nearest PHC       |
| Languages supported    | Nigerian Pidgin + English (with some local mixing)  |

**Design choice**: Prioritise **recall on the Urgent class**. Missing an emergency is far worse than a false alarm.

## Evaluation (held-out synthetic test set, n=120)

| Model                        | Urgent Recall | Urgent Precision | Accuracy |
|-----------------------------|---------------|------------------|----------|
| TF-IDF + LogReg             | 1.00          | 1.00             | 1.00     |
| Keyword baseline            | 0.96          | 1.00             | 0.98     |
| Location match (known locs) | 98.8%         | -                | -        |

*Note: Performance is high because the current data is synthetic and generated from the same templates/rules. Real-world performance will be lower; the next step is to collect or carefully annotate real (anonymised) messages and re-train / fine-tune a small multilingual model (e.g. Gemma 1B + LoRA or XLM-R).*

## Next steps toward the full proposal

1. Replace / augment synthetic data with real anonymised SMS (or carefully crafted multi-language examples from CHWs).
2. Fine-tune a small multilingual model (Gemma-2B / 1B + LoRA, or AfriBERTa / XLM-R) for better Pidgin + code-switched handling.
3. Expand the PHC gazetteer (full Kano State -> other states) and add distance / transport heuristics.
4. Human-in-the-loop review interface for CHWs (uncertain messages go to human).
5. Governance: no PII in training data, weekly CHW audit sample, kill-switch, NDPA 2023 compliance.

## Governance notes (from proposal)

- Tool only flags urgency + suggests facility. Never diagnoses or advises treatment.
- Every message remains visible to the CHW.
- Uncertain predictions should be routed for human review.
- Facility list must be kept current.

## Team

- Benison Blessing Odigie (Teaching Assistant, University of Benin)
- Prof. Susan Konyeha (AI Champion, University of Benin)
- Pathway: Classification

## Streamlit App

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the URL shown in the terminal (usually http://localhost:8501).

Features:
- Paste SMS in English, Pidgin, Hausa, Yoruba, Igbo or Edo
- Optional GPS coordinates for precise Haversine ranking
- Urgency prediction + nearest PHC + distance + simple map

## OpenStreetMap routing

The Streamlit app shows an interactive OpenStreetMap view from the **patient GPS**
(or matched community) to the **nearest PHC**, with:

- Patient marker (red)
- Nearest PHC markers (green/blue)
- Route line (straight-line on map)
- Link to OSM driving directions (OSRM)

```bash
pip install folium streamlit-folium
streamlit run app.py
```

## Hugging Face access (Gemma is gated)

`google/gemma-2-2b-it` and `google/gemma-2-9b-it` require license acceptance:

1. Open the model page and accept the license, e.g.  
   https://huggingface.co/google/gemma-2-2b-it  
   https://huggingface.co/google/gemma-2-9b-it
2. Create a token: https://huggingface.co/settings/tokens
3. Authenticate, then train:

```bash
export HF_TOKEN=hf_your_token_here
python src/gemma_lora_train.py --model google/gemma-2-2b-it --epochs 2

# Larger model (needs more VRAM)
python src/gemma_lora_train.py --model google/gemma-2-9b-it --epochs 1 --batch-size 1

# Or pass token on the command line
python src/gemma_lora_train.py --token hf_your_token_here --model google/gemma-2-2b-it
```

Non-gated alternative for testing without approval:

```bash
python src/gemma_lora_train.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
```
