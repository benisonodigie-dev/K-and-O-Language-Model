#!/usr/bin/env python3
"""
Urgent SMS Triage + Nearest PHC Referral
- High-recall keyword rules
- Full Kano State gazetteer (44 LGAs)
- Haversine distance for geolocation & nearest-facility ranking
- Defensive error handling throughout
"""

from __future__ import annotations

import json
import math
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("triage_model")

# Optional Gemma-LoRA classifier (loaded lazily)
try:
    from gemma_lora_infer import predict_urgency as gemma_predict, adapter_available as gemma_available
except Exception:
    try:
        from src.gemma_lora_infer import predict_urgency as gemma_predict, adapter_available as gemma_available
    except Exception:
        def gemma_predict(text):
            return {"available": False, "prediction": None, "confidence": None, "error": "module missing"}
        def gemma_available():
            return False

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Gazetteer loading with error handling
# ---------------------------------------------------------------------------
def load_gazetteer() -> Dict[str, Dict]:
    path = DATA_DIR / "locations_gazetteer.json"
    try:
        if not path.exists():
            logger.warning("Gazetteer file not found: %s", path)
            return {}
        with open(path, encoding="utf-8") as f:
            locs = json.load(f)
        if not isinstance(locs, list):
            logger.error("Gazetteer must be a JSON list of location objects")
            return {}
        gaz = {}
        for loc in locs:
            try:
                name = str(loc["name"]).strip().lower()
                if not name:
                    continue
                gaz[name] = {
                    "name": str(loc["name"]).strip(),
                    "phc": str(loc.get("phc", "Nearest Primary Health Centre")).strip(),
                    "lga": str(loc.get("lga", "Unknown")).strip(),
                    "state": str(loc.get("state", "Unknown")).strip(),
                    "lat": float(loc["lat"]),
                    "lon": float(loc["lon"]),
                }
            except (KeyError, TypeError, ValueError) as e:
                logger.warning("Skipping invalid gazetteer entry %s: %s", loc, e)
        if not gaz:
            logger.warning("Gazetteer loaded but empty after validation")
        return gaz
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in gazetteer: %s", e)
        return {}
    except OSError as e:
        logger.error("Could not read gazetteer file: %s", e)
        return {}


GAZETTEER: Dict[str, Dict] = load_gazetteer()
ALL_LOCATIONS: List[Dict] = list(GAZETTEER.values())

def available_states() -> List[str]:
    """Sorted list of states present in the gazetteer."""
    return sorted({loc.get("state", "Unknown") for loc in ALL_LOCATIONS if loc.get("state")})


def locations_for_state(state: Optional[str] = None) -> List[Dict]:
    """Return locations filtered by state (None or 'All' = no filter)."""
    if not state or str(state).strip().lower() in {"all", "any", ""}:
        return list(ALL_LOCATIONS)
    state_l = str(state).strip().lower()
    return [loc for loc in ALL_LOCATIONS if str(loc.get("state", "")).lower() == state_l]


def gazetteer_for_state(state: Optional[str] = None) -> Dict[str, Dict]:
    """Name->location dict filtered by state."""
    return {loc["name"].lower(): loc for loc in locations_for_state(state)}



# ---------------------------------------------------------------------------
# Haversine distance
# ---------------------------------------------------------------------------
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Great-circle distance between two points (decimal degrees) in kilometres.
    Raises ValueError for non-numeric or out-of-range coordinates.
    """
    try:
        lat1, lon1, lat2, lon2 = float(lat1), float(lon1), float(lat2), float(lon2)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Coordinates must be numbers: {e}") from e

    for name, val in (("lat1", lat1), ("lat2", lat2)):
        if not -90.0 <= val <= 90.0:
            raise ValueError(f"{name}={val} out of range [-90, 90]")
    for name, val in (("lon1", lon1), ("lon2", lon2)):
        if not -180.0 <= val <= 180.0:
            raise ValueError(f"{name}={val} out of range [-180, 180]")

    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    a = min(1.0, max(0.0, a))  # numerical safety
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def find_nearest_phc(
    lat: float,
    lon: float,
    top_k: int = 3,
    state: Optional[str] = None,
) -> List[Tuple[Dict, float]]:
    """Return top-k nearest facilities by Haversine distance (optionally within state)."""
    pool = locations_for_state(state)
    if not pool:
        logger.warning("No locations in gazetteer for state=%s", state)
        return []
    try:
        top_k = max(1, int(top_k))
    except (TypeError, ValueError):
        top_k = 3

    ranked: List[Tuple[Dict, float]] = []
    for loc in pool:
        try:
            dist = haversine_km(lat, lon, loc["lat"], loc["lon"])
            ranked.append((loc, dist))
        except ValueError as e:
            logger.debug("Skipping location %s: %s", loc.get("name"), e)
    ranked.sort(key=lambda x: x[1])
    return ranked[:top_k]


# ---------------------------------------------------------------------------
# Urgency keywords
# ---------------------------------------------------------------------------
URGENT_PATTERNS = [
    r"\bbleed(ing|s)?\b", r"\bconvuls(ion|ing|e)?\b", r"\bfit(ting|s)?\b",
    r"\bunconscious\b", r"\bnot breathing\b", r"\bdifficulty breathing\b",
    r"\bchest (is )?sinking\b", r"\bhigh fever\b", r"\bnot responding\b",
    r"\beyes? rolling\b", r"\bheavy bleeding\b", r"\bblood (dey |is )?flow\b",
    r"\bblood everywhere\b", r"\bcollapsed?\b", r"\bsevere (pain|bleeding|stomach)\b",
    r"\bbody (is )?cold\b", r"\bnot cried\b", r"\bno cry\b", r"\burgent\b",
    r"\bsharp sharp\b", r"\bdey bleed\b", r"\bdey shake\b", r"\bdey fit\b",
    r"\bno dey breathe\b", r"\bno dey respond\b", r"\bweak well well\b",
    r"\balmost faint\b", r"\blose (belle|consciousness)\b", r"\bfoam(ing)?\b",
    r"\bfainted?\b", r"\bhospital\b",
    # Hausa
    r"\bzubar\b", r"\bjini\b", r"\btashin hankali\b", r"\bgaggawa\b",
    r"\bzazzabi\b", r"\bnumfashi\b", r"\brauni\b",
    # Yoruba
    r"\bse eje\b", r"\begba\b", r"\bkiakia\b", r"\biba giga\b", r"\bsunkun\b",
    # Igbo
    r"\bagba obara\b", r"\bna-agba egwu\b", r"\bngwa ngwa\b", r"\bahihia\b",
    r"\biku ume\b",
    # Edo
    r"\besagi[eẹ]n\b", r"\biyob[oọ]\b", r"\bn[eẹ]gi[eẹ]gi[eẹ]\b",
]

try:
    COMPILED = [re.compile(p, re.I) for p in URGENT_PATTERNS]
except re.error as e:
    logger.error("Invalid urgency regex: %s", e)
    COMPILED = []


def keyword_score(text: str) -> int:
    if not text or not isinstance(text, str):
        return 0
    try:
        return sum(1 for p in COMPILED if p.search(text))
    except Exception as e:
        logger.warning("keyword_score failed: %s", e)
        return 0


def keyword_predict(text: str) -> str:
    try:
        return "Urgent" if keyword_score(text) >= 1 else "Routine"
    except Exception:
        return "Routine"


# ---------------------------------------------------------------------------
# Location extraction
# ---------------------------------------------------------------------------
def extract_location(text: str, state: Optional[str] = None) -> Optional[Dict]:
    if not text or not isinstance(text, str):
        return None
    gaz = gazetteer_for_state(state)
    if not gaz:
        return None
    try:
        text_l = text.lower()
        for key in sorted(gaz.keys(), key=len, reverse=True):
            if key in text_l:
                return gaz[key]
    except Exception as e:
        logger.warning("extract_location failed: %s", e)
    return None


# ---------------------------------------------------------------------------
# Referral message
# ---------------------------------------------------------------------------
def suggest_referral(
    loc: Optional[Dict],
    is_urgent: bool,
    patient_lat: Optional[float] = None,
    patient_lon: Optional[float] = None,
    state: Optional[str] = None,
) -> str:
    try:
        if not is_urgent:
            return "Routine message. No urgent referral needed."

        if loc is None and patient_lat is not None and patient_lon is not None:
            nearest = find_nearest_phc(patient_lat, patient_lon, top_k=2, state=state)
            if not nearest:
                return ("Urgent. Unable to determine nearest facility "
                        "(gazetteer empty or coordinates invalid).")
            best, dist = nearest[0]
            msg = (
                f"Urgent. Nearest facility by GPS: {best['phc']} "
                f"({best['name']}, {best['lga']}) - {dist:.1f} km away "
                f"[lat={best['lat']:.4f}, lon={best['lon']:.4f}]."
            )
            if len(nearest) > 1:
                alt, alt_dist = nearest[1]
                msg += f" Alternative: {alt['phc']} ({alt_dist:.1f} km)."
            return msg

        if loc is None:
            return ("Urgent. Location not clearly identified. "
                    "Please ask patient for community name or GPS coordinates "
                    "and direct to nearest Primary Health Centre.")

        msg = (
            f"Urgent. Location: {loc['name']} ({loc['lga']}). "
            f"Nearest: {loc['phc']} "
            f"[lat={loc['lat']:.4f}, lon={loc['lon']:.4f}]"
        )

        if patient_lat is not None and patient_lon is not None:
            try:
                dist = haversine_km(patient_lat, patient_lon, loc["lat"], loc["lon"])
                msg += f" - {dist:.1f} km from reported position"
                nearest = find_nearest_phc(patient_lat, patient_lon, top_k=2, state=state)
                if (nearest and nearest[0][0]["name"] != loc["name"]
                        and nearest[0][1] + 2 < dist):
                    alt, alt_dist = nearest[0]
                    msg += (f". Note: {alt['phc']} ({alt['name']}) is closer "
                            f"({alt_dist:.1f} km) by straight-line distance")
            except ValueError as e:
                logger.warning("Distance calculation skipped: %s", e)

        return msg + "."
    except Exception as e:
        logger.error("suggest_referral failed: %s", e)
        return "Urgent. Error generating referral - direct to nearest Primary Health Centre."


# ---------------------------------------------------------------------------
# Main prediction API
# ---------------------------------------------------------------------------
def _safe_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def predict_message(
    text: str,
    patient_lat: Optional[float] = None,
    patient_lon: Optional[float] = None,
    state: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Classify urgency and produce a referral recommendation.
    Always returns a dict; never raises to the caller.
    """
    empty = {
        "text": text if isinstance(text, str) else "",
        "prediction": "Routine",
        "keyword_score": 0,
        "location": None,
        "nearest_phc": None,
        "lga": None,
        "state": None,
        "latitude": None,
        "longitude": None,
        "distance_km": None,
        "nearest_facilities": [],
        "referral_message": "No message provided.",
        "error": None,
    }

    try:
        if not text or not isinstance(text, str) or not text.strip():
            empty["error"] = "Empty or invalid text input"
            empty["referral_message"] = "Please provide a non-empty SMS message."
            return empty

        text = text.strip()
        patient_lat = _safe_float(patient_lat)
        patient_lon = _safe_float(patient_lon)

        # Both-or-neither for coordinates
        if (patient_lat is None) ^ (patient_lon is None):
            empty["text"] = text
            empty["error"] = "Provide both latitude and longitude, or neither"
            empty["referral_message"] = empty["error"]
            # Still try text-only path below
            patient_lat = patient_lon = None

        if patient_lat is not None and patient_lon is not None:
            try:
                # Validate range early
                haversine_km(patient_lat, patient_lon, patient_lat, patient_lon)
            except ValueError as e:
                empty["text"] = text
                empty["error"] = str(e)
                empty["referral_message"] = f"Invalid coordinates: {e}"
                patient_lat = patient_lon = None

        # --- Classification: Gemma-LoRA (if available) ensembled with keywords ---
        kw_label = keyword_predict(text)
        kw_score = keyword_score(text)
        ml = gemma_predict(text)
        if ml.get("available") and ml.get("prediction"):
            # High-recall ensemble: Urgent if EITHER model says Urgent
            if ml["prediction"] == "Urgent" or kw_label == "Urgent":
                label = "Urgent"
            else:
                label = "Routine"
            ml_info = {
                "ml_prediction": ml["prediction"],
                "ml_confidence": ml.get("confidence"),
                "ml_probs": ml.get("probs"),
            }
        else:
            label = kw_label
            ml_info = {
                "ml_prediction": None,
                "ml_confidence": None,
                "ml_probs": {},
                "ml_error": ml.get("error"),
            }
        loc = extract_location(text, state=state)

        nearest_list: List[Dict] = []
        if patient_lat is not None and patient_lon is not None:
            try:
                nearest_list = [
                    {
                        "name": l["name"],
                        "phc": l["phc"],
                        "lga": l["lga"],
                        "state": l.get("state"),
                        "lat": l["lat"],
                        "lon": l["lon"],
                        "distance_km": round(d, 2),
                    }
                    for l, d in find_nearest_phc(patient_lat, patient_lon, top_k=3, state=state)
                ]
            except Exception as e:
                logger.warning("find_nearest_phc failed: %s", e)

        return {
            "text": text,
            "prediction": label,
            "keyword_score": keyword_score(text),
            "location": loc["name"] if loc else None,
            "nearest_phc": (
                loc["phc"] if loc
                else (nearest_list[0]["phc"] if nearest_list else None)
            ),
            "lga": (
                loc["lga"] if loc
                else (nearest_list[0]["lga"] if nearest_list else None)
            ),
            "state": (
                loc.get("state") if loc
                else (nearest_list[0].get("state") if nearest_list else state)
            ),
            "latitude": (
                loc["lat"] if loc
                else (nearest_list[0]["lat"] if nearest_list else None)
            ),
            "longitude": (
                loc["lon"] if loc
                else (nearest_list[0]["lon"] if nearest_list else None)
            ),
            "distance_km": nearest_list[0]["distance_km"] if nearest_list else None,
            "nearest_facilities": nearest_list,
            "referral_message": suggest_referral(
                loc, label == "Urgent", patient_lat, patient_lon, state=state
            ),
            "error": None,
        }
    except Exception as e:
        logger.exception("predict_message unexpected error")
        empty["text"] = text if isinstance(text, str) else ""
        empty["error"] = str(e)
        empty["referral_message"] = (
            "An unexpected error occurred. Please try again or contact support."
        )
        return empty


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    try:
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning("Skipping bad JSONL line %d: %s", i, e)
    except OSError as e:
        logger.error("Cannot read %s: %s", path, e)
    return rows


def evaluate() -> None:
    test_path = DATA_DIR / "test.jsonl"
    data = load_jsonl(test_path)
    if not data:
        print("No test data found - skipping evaluation.")
        return

    y_true = [d.get("label", "Routine") for d in data]
    y_pred = [keyword_predict(d.get("text", "")) for d in data]
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == "Urgent" and p == "Urgent")
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == "Routine" and p == "Urgent")
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == "Urgent" and p == "Routine")
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == "Routine" and p == "Routine")
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    print(f"Keyword model on test set (n={len(data)}):")
    print(f"  Urgent  Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}")
    print(f"  Confusion: TP={tp} FP={fp} FN={fn} TN={tn}")

    loc_ok = loc_tot = 0
    for d in data:
        if d.get("location_mentioned"):
            loc_tot += 1
            ext = extract_location(d.get("text", ""))
            if ext and ext["name"].lower() == str(d["location_mentioned"]).lower():
                loc_ok += 1
    print(f"Location match accuracy: {loc_ok}/{loc_tot} = {loc_ok/max(loc_tot,1):.1%}")


# ---------------------------------------------------------------------------
# Demo / self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Gazetteer loaded: {len(GAZETTEER)} locations across Kano State\n")
    evaluate()

    print("\n--- Haversine self-test ---")
    try:
        d = haversine_km(12.0024, 8.5132, 12.0906, 8.4967)
        print(f"Kano -> Ungogo : {d:.2f} km")
        d2 = haversine_km(12.0024, 8.5132, 11.8094, 8.8442)
        print(f"Kano -> Wudil : {d2:.2f} km")
    except ValueError as e:
        print(f"Haversine error: {e}")

    print("\n--- Nearest PHC from GPS (Kano centre) ---")
    nearest = find_nearest_phc(12.00, 8.52, top_k=3)
    for loc, dist in nearest:
        print(f"  {loc['name']:15} {loc['phc'][:40]:40} {dist:6.2f} km")

    print("\n--- Prediction demos ---")
    demos = [
        ("Please help, my wife is 8 months pregnant and bleeding heavily. We are in Kano.", None, None),
        ("Abeg my pikin dey convulsing. We dey Wudil.", None, None),
        ("Pregnant woman collapsed. Need help now.", 12.00, 8.52),
        ("She just delivered and bleeding nonstop. Location: Bichi.", 12.23, 8.23),
        ("", None, None),  # empty
        ("Hello", 999.0, 8.5),  # bad lat
    ]
    for text, lat, lon in demos:
        res = predict_message(text, patient_lat=lat, patient_lon=lon)
        preview = (text[:55] + "...") if text and len(text) > 55 else (text or "(empty)")
        print(f"\nIN : {preview}")
        print(f"OUT: {res['prediction']}")
        if res.get("error"):
            print(f"     ERROR: {res['error']}")
        print(f"     {res['referral_message']}")
