"""
Fetch UNDP GeoHub Zanzibar Tourism Attractions and convert it to the app's CSV schema.

The GeoHub page shows a "GeoHub Dataset API URL" button. If the default endpoint
changes, copy that API URL from the dataset page and pass it here:

    python scripts/fetch_undp_zanzibar.py --url "PASTE_GEOHUB_DATASET_API_URL" --out data/undp_zanzibar_attractions.csv

Dataset page:
    https://geohub.data.undp.org/data/4ca2ead25b5903e8e1c7897f8f3bae38

Notes:
- GeoHub currently lists this dataset as FlatGeobuf. If the API returns a direct
  .fgb file instead of JSON/CSV, install geopandas + pyogrio and pass --url to
  the .fgb file; this script can read it when geopandas is available.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

DATASET_ID = "4ca2ead25b5903e8e1c7897f8f3bae38"
DEFAULT_CANDIDATE_URLS = [
    f"https://geohub.data.undp.org/api/datasets/{DATASET_ID}",
    f"https://geohub.data.undp.org/api/datasets/{DATASET_ID}/attributes?format=json&limit=10000",
    f"https://geohub.data.undp.org/api/datasets/{DATASET_ID}/attributes/default?format=json&limit=10000",
    f"https://geohub.data.undp.org/api/datasets/{DATASET_ID}/attributes/0?format=json&limit=10000",
]


def flatten_json_records(payload: Any) -> List[Dict[str, Any]]:
    """Try common GeoJSON/API shapes and return a list of flat records."""
    if isinstance(payload, list):
        return [x if isinstance(x, dict) else {"value": x} for x in payload]

    if not isinstance(payload, dict):
        return []

    # GeoJSON FeatureCollection
    if isinstance(payload.get("features"), list):
        rows = []
        for f in payload["features"]:
            props = f.get("properties", {}) if isinstance(f, dict) else {}
            geom = f.get("geometry", {}) if isinstance(f, dict) else {}
            coords = geom.get("coordinates") if isinstance(geom, dict) else None
            rec = dict(props)
            if isinstance(coords, list) and len(coords) >= 2:
                rec.setdefault("longitude", coords[0])
                rec.setdefault("latitude", coords[1])
            rows.append(rec)
        return rows

    # Common API wrappers
    for key in ["data", "items", "rows", "results", "attributes"]:
        if isinstance(payload.get(key), list):
            return flatten_json_records(payload[key])
        if isinstance(payload.get(key), dict):
            nested = flatten_json_records(payload[key])
            if nested:
                return nested

    # Metadata object might contain asset URLs. Return no records; caller will try next URL.
    return []


def read_any_url(url: str) -> Optional[pd.DataFrame]:
    print(f"Trying: {url}")
    r = requests.get(url, timeout=180)
    if r.status_code >= 400:
        print(f"  HTTP {r.status_code}")
        return None

    ctype = r.headers.get("content-type", "").lower()
    text_head = r.text[:200].lstrip() if "text" in ctype or "json" in ctype or len(r.content) < 5_000_000 else ""

    if "csv" in ctype or url.lower().endswith(".csv"):
        return pd.read_csv(url)

    if "json" in ctype or text_head.startswith("{") or text_head.startswith("["):
        payload = r.json()
        records = flatten_json_records(payload)
        if records:
            return pd.DataFrame(records)

        # Look for URLs inside metadata JSON and try them.
        possible = re.findall(r"https?://[^\"'\s]+", json.dumps(payload))
        for asset_url in possible:
            if any(ext in asset_url.lower() for ext in [".geojson", ".json", ".csv", ".fgb"]):
                df = read_any_url(asset_url)
                if df is not None and len(df):
                    return df
        return None

    if url.lower().endswith(".fgb") or "flatgeobuf" in ctype:
        try:
            import geopandas as gpd  # optional heavy dependency
            gdf = gpd.read_file(url)
            df = pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))
            if hasattr(gdf, "geometry"):
                df["longitude"] = gdf.geometry.centroid.x
                df["latitude"] = gdf.geometry.centroid.y
            return df
        except Exception as exc:
            print("  Could not read FlatGeobuf automatically.")
            print(f"  Install optional GIS dependencies: pip install geopandas pyogrio ({exc})")
            return None

    return None


def pick_col(df: pd.DataFrame, candidates: List[str], default: str = "") -> pd.Series:
    lookup = {c.lower().replace(" ", "_"): c for c in df.columns}
    for name in candidates:
        key = name.lower().replace(" ", "_")
        if key in lookup:
            return df[lookup[key]].fillna("").astype(str)
    return pd.Series([default] * len(df))


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    name = pick_col(df, ["name", "title", "attraction", "attraction_name", "site_name", "location_name"])
    category = pick_col(df, ["category", "type", "tourism", "activity", "activities", "class"], "tourism attraction")
    region = pick_col(df, ["region", "district", "area", "island", "location"], "Zanzibar")
    lat = pd.to_numeric(pick_col(df, ["latitude", "lat", "y"]), errors="coerce")
    lon = pd.to_numeric(pick_col(df, ["longitude", "lon", "lng", "x"]), errors="coerce")
    desc = pick_col(df, ["description", "description_en", "details", "about", "summary", "narrative"])
    tips = pick_col(df, ["tips", "services", "nearby_services", "notes", "accessibility"], "Verify access, guide availability, opening hours, and fees before visiting.")
    keywords = pick_col(df, ["keywords", "activities", "tags", "amenities"], "Zanzibar tourism attraction")

    out = pd.DataFrame({
        "id": [f"undp-{i+1}" for i in range(len(df))],
        "name": name,
        "category": category,
        "region": region,
        "latitude": lat,
        "longitude": lon,
        "description_sw": desc,
        "description_en": desc,
        "tips": tips,
        "source": "UNDP GeoHub Zanzibar Tourism Attractions",
        "keywords": keywords,
    })
    out = out[out["name"].astype(str).str.strip().str.len() > 1]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="", help="Optional copied GeoHub Dataset API/File URL")
    parser.add_argument("--out", default="data/undp_zanzibar_attractions.csv")
    args = parser.parse_args()

    urls = [args.url] if args.url else DEFAULT_CANDIDATE_URLS
    df = None
    for url in urls:
        if not url:
            continue
        try:
            df = read_any_url(url)
            if df is not None and len(df):
                break
        except Exception as exc:
            print(f"  Failed: {exc}")

    if df is None or not len(df):
        raise SystemExit(
            "Could not fetch rows automatically. Open the UNDP GeoHub dataset page, copy the 'GeoHub Dataset API URL' or File URL, "
            "then run: python scripts/fetch_undp_zanzibar.py --url 'PASTE_URL'"
        )

    out = standardize(df)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Saved {len(out)} standardized rows to {out_path}")


if __name__ == "__main__":
    main()
