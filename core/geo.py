"""
Geographical helpers used on the server side.

The live, moving map lives in static/map.html (browser GPS). These functions
let the *assistant text* answer location questions: geocode a place, measure how
far it is from the tourist, and fetch a driving route from OSRM.
"""
from __future__ import annotations

import math
from typing import Optional, Dict, Any, List

import requests

from .config import cfg

# Rough bounding box around Zanzibar + the Tanzanian coast, used to bias
# geocoding so "beach" resolves locally rather than somewhere across the world.
# (left lon, top lat, right lon, bottom lat)
ZANZIBAR_VIEWBOX = "38.9,-5.6,40.0,-6.6"


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def geocode(query: str) -> Optional[Dict[str, Any]]:
    """Resolve a place name to coordinates via Nominatim (Zanzibar-biased)."""
    if not query.strip():
        return None
    try:
        resp = requests.get(
            cfg.NOMINATIM_URL,
            params={
                "q": query,
                "format": "json",
                "limit": 1,
                "viewbox": ZANZIBAR_VIEWBOX,
                "bounded": 0,
                "countrycodes": "tz",
            },
            headers={"User-Agent": cfg.GEO_USER_AGENT},
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return None
        top = data[0]
        return {
            "lat": float(top["lat"]),
            "lon": float(top["lon"]),
            "display_name": top.get("display_name", query),
        }
    except Exception:
        return None


def route(o_lat: float, o_lon: float, d_lat: float, d_lon: float) -> Optional[Dict[str, Any]]:
    """Driving route from origin to destination via OSRM."""
    try:
        url = (
            f"{cfg.OSRM_URL.rstrip('/')}/route/v1/driving/"
            f"{o_lon},{o_lat};{d_lon},{d_lat}"
        )
        resp = requests.get(
            url,
            params={"overview": "full", "geometries": "geojson", "steps": "true"},
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != "Ok" or not data.get("routes"):
            return None
        r = data["routes"][0]
        steps: List[str] = []
        for leg in r.get("legs", []):
            for st in leg.get("steps", []):
                man = st.get("maneuver", {})
                road = st.get("name", "")
                instr = man.get("type", "continue").replace("_", " ")
                mod = man.get("modifier", "")
                steps.append(" ".join(p for p in [instr, mod, road] if p).strip())
        return {
            "distance_km": round(r["distance"] / 1000, 2),
            "duration_min": round(r["duration"] / 60, 1),
            "steps": steps,
        }
    except Exception:
        return None


def describe_location(hits, user_lat: Optional[float], user_lon: Optional[float], lang: str) -> str:
    """
    Build a one-line distance/direction note for the top hit that has coords.
    Returns '' if we have no user location or no coordinates to work with.
    """
    if user_lat is None or user_lon is None:
        return ""
    for h in hits or []:
        if h.get("lat") is not None and h.get("lon") is not None:
            d = haversine_km(user_lat, user_lon, h["lat"], h["lon"])
            rt = route(user_lat, user_lon, h["lat"], h["lon"])
            if rt:
                if lang == "sw":
                    return (f"📍 {h['name']} iko takriban km {rt['distance_km']} "
                            f"(dakika {rt['duration_min']} kwa gari) kutoka ulipo.")
                return (f"📍 {h['name']} is about {rt['distance_km']} km "
                        f"(~{rt['duration_min']} min by road) from you.")
            if lang == "sw":
                return f"📍 {h['name']} iko takriban km {d:.1f} kutoka ulipo (mstari wa moja kwa moja)."
            return f"📍 {h['name']} is about {d:.1f} km from you (straight line)."
    return ""
