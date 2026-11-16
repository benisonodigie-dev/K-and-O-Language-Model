#!/usr/bin/env python3
"""
Interactive CLI for Urgent SMS Triage + Nearest-Care Referral
Supports state selection, optional GPS coordinates, and basic error handling.

Usage:
  python demo/triage_cli.py
  python demo/triage_cli.py --state Kano
  python demo/triage_cli.py --state Edo --lat 6.34 --lon 5.63
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from triage_model import predict_message, available_states, locations_for_state
except Exception as e:
    print(f"Failed to load triage model: {e}", file=sys.stderr)
    sys.exit(1)

STATE_CENTRES = {
    "Kano": (12.0024, 8.5132),
    "Edo": (6.3382, 5.6258),
    "Lagos": (6.5244, 3.3792),
}


def parse_args():
    states = available_states() or ["Kano", "Edo", "Lagos"]
    parser = argparse.ArgumentParser(
        description="Urgent SMS Triage CLI (state-aware + Haversine geolocation)"
    )
    parser.add_argument(
        "--state",
        choices=states,
        default=None,
        help="Patient state (filters communities and nearest PHC). If omitted, you will be prompted.",
    )
    parser.add_argument("--lat", type=float, default=None, help="Patient latitude")
    parser.add_argument("--lon", type=float, default=None, help="Patient longitude")
    return parser.parse_args(), states


def choose_state(states, preselected=None):
    if preselected and preselected in states:
        return preselected
    print("Available states:")
    for i, s in enumerate(states, 1):
        n = len(locations_for_state(s))
        print(f"  {i}. {s}  ({n} locations)")
    while True:
        try:
            raw = input(f"Select state [1-{len(states)}] or name: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            sys.exit(0)
        if not raw:
            # default first state
            return states[0]
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(states):
                return states[idx - 1]
            print(f"  Enter a number between 1 and {len(states)}.")
            continue
        # match by name (case-insensitive)
        for s in states:
            if s.lower() == raw.lower():
                return s
        print(f"  Unknown state '{raw}'. Choose from: {', '.join(states)}")


def main():
    args, states = parse_args()
    state = choose_state(states, args.state)
    n_loc = len(locations_for_state(state))

    default_lat, default_lon = STATE_CENTRES.get(state, (9.0, 8.0))
    # CLI-provided GPS overrides defaults; interactive mode can still supply per-message GPS
    session_lat = args.lat
    session_lon = args.lon

    print("=" * 64)
    print("  Urgent SMS Triage & Nearest-Care Referral")
    print(f"  State: {state}  ({n_loc} locations)")
    print("  Haversine geolocation + OpenStreetMap-ready coordinates")
    print("=" * 64)
    print("Commands:")
    print("  Type an SMS message and press Enter")
    print("  Or:  lat,lon | message     (e.g. 12.00,8.52 | bleeding heavily)")
    print("  state                      (change state)")
    print("  quit / exit                (stop)")
    if session_lat is not None and session_lon is not None:
        print(f"\nSession GPS: {session_lat}, {session_lon}")
    print()

    while True:
        try:
            raw = input("SMS> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        except Exception as e:
            print(f"  Input error: {e}")
            continue

        if not raw or raw.lower() in {"quit", "exit", "q"}:
            print("Bye.")
            break

        # Change state interactively
        if raw.lower() in {"state", "change state", "/state"}:
            state = choose_state(states)
            n_loc = len(locations_for_state(state))
            default_lat, default_lon = STATE_CENTRES.get(state, (9.0, 8.0))
            print(f"  State set to {state} ({n_loc} locations).")
            continue

        patient_lat = session_lat
        patient_lon = session_lon
        text = raw

        if "|" in raw:
            coord_part, text = raw.split("|", 1)
            text = text.strip()
            try:
                parts = [p.strip() for p in coord_part.split(",")]
                if len(parts) != 2:
                    raise ValueError("expected lat,lon")
                patient_lat = float(parts[0])
                patient_lon = float(parts[1])
                if not (-90 <= patient_lat <= 90) or not (-180 <= patient_lon <= 180):
                    print("  Coordinates out of range. Lat [-90,90], Lon [-180,180].")
                    continue
            except Exception as e:
                print(f"  Could not parse coordinates ({e}). Use: lat,lon | message")
                continue

        if not text:
            print("  Please enter a non-empty message.")
            continue

        try:
            res = predict_message(
                text,
                patient_lat=patient_lat,
                patient_lon=patient_lon,
                state=state,
            )
        except Exception as e:
            print(f"  Triage failed: {e}")
            continue

        print(f"\n  -> Prediction : {res.get('prediction')}")
        print(f"  -> State      : {res.get('state') or state}")
        if res.get("error"):
            print(f"  -> Warning    : {res['error']}")
        if res.get("location"):
            print(f"  -> Location   : {res['location']} ({res.get('lga')})")
            print(f"  -> Coordinates: {res.get('latitude')}, {res.get('longitude')}")
            print(f"  -> Nearest PHC: {res.get('nearest_phc')}")
        if res.get("distance_km") is not None:
            print(f"  -> Distance   : {res['distance_km']} km (Haversine)")
        if res.get("nearest_facilities"):
            print("  -> Top nearest facilities:")
            for f in res["nearest_facilities"]:
                print(
                    f"       {f.get('distance_km', 0):6.2f} km  "
                    f"{f.get('name')} ({f.get('lga')}, {f.get('state') or state}) - {f.get('phc')}"
                )
        # OSM directions link when we have both ends
        if (
            patient_lat is not None
            and patient_lon is not None
            and res.get("nearest_facilities")
        ):
            n = res["nearest_facilities"][0]
            try:
                nlat, nlon = float(n["lat"]), float(n["lon"])
                url = (
                    "https://www.openstreetmap.org/directions?"
                    f"engine=fossgis_osrm_car&route={patient_lat}%2C{patient_lon}%3B{nlat}%2C{nlon}"
                )
                print(f"  -> OSM route  : {url}")
            except (KeyError, TypeError, ValueError):
                pass
        print(f"  -> Action     : {res.get('referral_message')}\n")


if __name__ == "__main__":
    main()
