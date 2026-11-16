#!/usr/bin/env python3
"""
Streamlit UI for Urgent SMS Triage + Nearest-Care Referral
State selector + OpenStreetMap route: patient -> nearest PHC
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st

try:
    from triage_model import (
        predict_message,
        ALL_LOCATIONS,
        available_states,
        locations_for_state,
    )
except Exception as e:
    st.error(f"Failed to load triage model: {e}")
    st.stop()


# Default map centres per state
STATE_CENTRES = {
    "Kano": (12.0024, 8.5132),
    "Edo": (6.3382, 5.6258),
    "Lagos": (6.5244, 3.3792),
}


def osm_directions_url(from_lat, from_lon, to_lat, to_lon) -> str:
    return (
        f"https://www.openstreetmap.org/directions?"
        f"engine=fossgis_osrm_car&route={from_lat}%2C{from_lon}%3B{to_lat}%2C{to_lon}"
    )


def osm_map_url(lat, lon, zoom=14) -> str:
    return f"https://www.openstreetmap.org/?mlat={lat}&mlon={lon}#map={zoom}/{lat}/{lon}"


def build_folium_map(patient: Tuple[float, float], facilities: List[Dict[str, Any]]):
    try:
        import folium
        from folium import plugins
    except ImportError:
        return None

    plat, plon = patient
    fmap = folium.Map(location=[plat, plon], zoom_start=12, tiles="OpenStreetMap", control_scale=True)

    folium.Marker(
        [plat, plon],
        tooltip="Patient / caregiver",
        popup=folium.Popup(f"<b>Patient</b><br>lat={plat:.5f}, lon={plon:.5f}", max_width=260),
        icon=folium.Icon(color="red", icon="user", prefix="fa"),
    ).add_to(fmap)

    for i, f in enumerate(facilities):
        try:
            flat, flon = float(f["lat"]), float(f["lon"])
        except (KeyError, TypeError, ValueError):
            continue
        color = "green" if i == 0 else "blue"
        dist = f.get("distance_km")
        dist_s = f"{dist:.2f} km" if dist is not None else ""
        popup_html = (
            f"<b>{f.get('phc', 'PHC')}</b><br>"
            f"{f.get('name', '')} ({f.get('lga', '')}, {f.get('state', '')})<br>"
            f"{dist_s}<br>lat={flat:.5f}, lon={flon:.5f}"
        )
        folium.Marker(
            [flat, flon],
            tooltip=f"{'Nearest: ' if i == 0 else ''}{f.get('name', 'PHC')}",
            popup=folium.Popup(popup_html, max_width=280),
            icon=folium.Icon(color=color, icon="plus-sign", prefix="fa"),
        ).add_to(fmap)
        folium.PolyLine(
            locations=[[plat, plon], [flat, flon]],
            color="#2563eb" if i == 0 else "#93c5fd",
            weight=5 if i == 0 else 3,
            opacity=0.85 if i == 0 else 0.5,
            dash_array=None if i == 0 else "6 8",
        ).add_to(fmap)

    points = [[plat, plon]]
    for f in facilities:
        try:
            points.append([float(f["lat"]), float(f["lon"])])
        except (KeyError, TypeError, ValueError):
            pass
    if len(points) > 1:
        fmap.fit_bounds(points, padding=(30, 30))
    try:
        plugins.MeasureControl(position="topleft", primary_length_unit="kilometers").add_to(fmap)
    except Exception:
        pass
    return fmap


st.set_page_config(page_title="SMS Triage - Nigeria", page_icon="🏥", layout="wide")

st.title("🏥 Urgent SMS Triage & Nearest-Care Referral")
st.caption("Multilingual · State-aware geolocation · OpenStreetMap routing")

states = available_states() or ["Kano", "Edo", "Lagos"]

with st.sidebar:
    st.header("Patient location")

    # ----- State selector -----
    selected_state = st.selectbox(
        "State",
        options=states,
        index=states.index("Kano") if "Kano" in states else 0,
        help="Limits community matching and nearest-PHC search to this state.",
    )

    state_locs = locations_for_state(selected_state)
    st.caption(f"{len(state_locs)} communities / PHCs in {selected_state}")

    use_gps = st.checkbox("Provide GPS coordinates", value=True)
    default_lat, default_lon = STATE_CENTRES.get(selected_state, (9.0, 8.0))

    # Reset defaults when state changes
    if "prev_state" not in st.session_state:
        st.session_state.prev_state = selected_state
    if st.session_state.prev_state != selected_state:
        st.session_state.prev_state = selected_state
        st.session_state.lat_val = default_lat
        st.session_state.lon_val = default_lon

    if "lat_val" not in st.session_state:
        st.session_state.lat_val = default_lat
    if "lon_val" not in st.session_state:
        st.session_state.lon_val = default_lon

    patient_lat = patient_lon = None
    if use_gps:
        patient_lat = st.number_input(
            "Patient latitude",
            value=float(st.session_state.lat_val),
            format="%.6f",
            min_value=-90.0,
            max_value=90.0,
            key="lat_input",
        )
        patient_lon = st.number_input(
            "Patient longitude",
            value=float(st.session_state.lon_val),
            format="%.6f",
            min_value=-180.0,
            max_value=180.0,
            key="lon_input",
        )
        st.session_state.lat_val = patient_lat
        st.session_state.lon_val = patient_lon
        st.caption(f"Default centre for {selected_state}")

    st.markdown("---")
    st.markdown("**Coverage**")
    st.markdown(f"- States: {', '.join(states)}")
    st.markdown(f"- Selected: **{selected_state}** ({len(state_locs)} locations)")
    st.markdown("- Languages: EN, Pidgin, Hausa, Yoruba, Igbo, Edo")

st.subheader("1. Paste or type the SMS")

# State-aware examples
examples_by_state = {
    "Kano": [
        "Please help, my wife is 8 months pregnant and bleeding heavily. We are in Kano.",
        "Abeg my pikin dey shake body and e no dey breathe well. We dey Wudil.",
        "Don Allah a taimaka, matata tana zubar da jini sosai. Muna Bichi.",
    ],
    "Edo": [
        "Please help, my wife is bleeding heavily. We are in Benin City.",
        "Abeg my pikin dey convulsing. We dey Auchi.",
        "Pregnant woman collapsed in Okada. Need hospital now.",
    ],
    "Lagos": [
        "My wife is 8 months pregnant and bleeding. We are in Ikeja.",
        "Abeg help, pikin no dey breathe well. We dey Surulere.",
        "She just delivered, heavy bleeding. Location: Ikorodu.",
    ],
}
default_examples = examples_by_state.get(selected_state, examples_by_state["Kano"])
example = st.selectbox("Or pick an example", ["(type your own)"] + default_examples)
text = st.text_area(
    "SMS text",
    value="" if example == "(type your own)" else example,
    height=120,
    placeholder="Type or paste the caregiver SMS here...",
)

run = st.button("Triage now", type="primary")

if run:
    try:
        if not text or not str(text).strip():
            st.warning("Please enter a non-empty SMS message.")
        else:
            lat_arg = lon_arg = None
            if use_gps:
                try:
                    lat_arg = float(patient_lat)
                    lon_arg = float(patient_lon)
                    if not (-90 <= lat_arg <= 90) or not (-180 <= lon_arg <= 180):
                        st.error("Latitude must be in [-90, 90] and longitude in [-180, 180].")
                        st.stop()
                except (TypeError, ValueError):
                    st.error("GPS coordinates must be valid numbers.")
                    st.stop()

            with st.spinner(f"Running triage for {selected_state}..."):
                res = predict_message(
                    text.strip(),
                    patient_lat=lat_arg,
                    patient_lon=lon_arg,
                    state=selected_state,
                )

            if res.get("error"):
                st.warning(f"Note: {res['error']}")

            st.markdown("---")
            st.subheader("2. Result")

            if res.get("prediction") == "Urgent":
                st.error(f"**URGENT**  ·  keyword score = {res.get('keyword_score', 0)}")
            else:
                st.success(f"**ROUTINE**  ·  keyword score = {res.get('keyword_score', 0)}")

            if res.get("ml_prediction"):
                conf = res.get("ml_confidence")
                conf_s = f"{conf:.2f}" if conf is not None else "n/a"
                st.caption(
                    f"Gemma-LoRA: **{res['ml_prediction']}** (confidence {conf_s}) · "
                    f"Keyword score={res.get('keyword_score', 0)}"
                )
            else:
                st.caption("Gemma-LoRA not loaded - keyword rules only.")

            st.info(res.get("referral_message") or "No referral message generated.")

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("State", res.get("state") or selected_state)
                st.metric("Location matched", res.get("location") or "None")
            with c2:
                st.metric("LGA", res.get("lga") or "-")
                st.metric("Nearest PHC", res.get("nearest_phc") or "-")
            with c3:
                if res.get("distance_km") is not None:
                    st.metric("Distance (Haversine)", f"{res['distance_km']:.2f} km")
                if res.get("latitude") is not None:
                    st.metric("Latitude", f"{res['latitude']:.4f}")
            with c4:
                if res.get("longitude") is not None:
                    st.metric("Longitude", f"{res['longitude']:.4f}")

            facilities = res.get("nearest_facilities") or []
            if facilities:
                st.subheader("Top nearest facilities")
                rows = [
                    {
                        "Rank": i + 1,
                        "Community": f.get("name", ""),
                        "LGA": f.get("lga", ""),
                        "State": f.get("state", selected_state),
                        "PHC": f.get("phc", ""),
                        "Distance (km)": f.get("distance_km", ""),
                        "Lat": f.get("lat", ""),
                        "Lon": f.get("lon", ""),
                    }
                    for i, f in enumerate(facilities)
                ]
                st.dataframe(rows, use_container_width=True, hide_index=True)

            # OpenStreetMap
            st.subheader("3. OpenStreetMap route (patient -> nearest care)")
            map_patient = None
            if lat_arg is not None and lon_arg is not None:
                map_patient = (lat_arg, lon_arg)
            elif res.get("latitude") is not None and res.get("longitude") is not None:
                map_patient = (float(res["latitude"]), float(res["longitude"]))

            if map_patient and facilities:
                plat, plon = map_patient
                nearest = facilities[0]
                try:
                    nlat, nlon = float(nearest["lat"]), float(nearest["lon"])
                except (KeyError, TypeError, ValueError):
                    nlat = nlon = None

                if nlat is not None:
                    dir_url = osm_directions_url(plat, plon, nlat, nlon)
                    st.markdown(
                        f"**Nearest care in {selected_state}:** {nearest.get('phc', 'PHC')} "
                        f"({nearest.get('name', '')}, {nearest.get('lga', '')})  \n"
                        f"Straight-line distance: **{nearest.get('distance_km', '?')} km**  \n"
                        f"[Open driving directions in OpenStreetMap]({dir_url}) · "
                        f"[View PHC on OSM]({osm_map_url(nlat, nlon)})"
                    )
                    fmap = build_folium_map(map_patient, facilities)
                    if fmap is not None:
                        try:
                            from streamlit_folium import st_folium
                            st_folium(fmap, width=None, height=480, returned_objects=[])
                        except ImportError:
                            try:
                                st.components.v1.html(fmap.get_root().render(), height=480)
                            except Exception as e:
                                st.warning(f"Interactive map unavailable ({e}). Use the OSM link above.")
                    else:
                        st.info("Install folium for the interactive map: `pip install folium streamlit-folium`")
            elif not map_patient:
                st.info("Provide patient GPS or a known community name to show the OSM route.")
            else:
                st.info(f"No facilities ranked in {selected_state} for this query.")

    except Exception as e:
        st.error(f"Unexpected error while running triage: {e}")

st.markdown("---")
st.caption(
    "Prototype · Map tiles (c) OpenStreetMap contributors · Not a medical device"
)
