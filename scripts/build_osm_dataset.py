"""
Download tourism Points of Interest from OpenStreetMap Overpass API.

Examples:
    python scripts/build_osm_dataset.py --area zanzibar --out data/osm_zanzibar_tourism.csv
    python scripts/build_osm_dataset.py --area tanzania --out data/osm_tanzania_tourism.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import requests

BBOX = {
    # south, west, north, east
    "zanzibar": (-6.60, 39.10, -5.65, 39.65),
    "tanzania": (-11.90, 29.30, -0.90, 40.60),
}

OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def build_query(bbox):
    s, w, n, e = bbox
    return f"""
    [out:json][timeout:180];
    (
      node["tourism"]({s},{w},{n},{e});
      way["tourism"]({s},{w},{n},{e});
      relation["tourism"]({s},{w},{n},{e});
      node["historic"]({s},{w},{n},{e});
      way["historic"]({s},{w},{n},{e});
      relation["historic"]({s},{w},{n},{e});
      node["natural"~"beach|peak|waterfall|bay"]({s},{w},{n},{e});
      way["natural"~"beach|peak|waterfall|bay"]({s},{w},{n},{e});
      node["leisure"~"park|nature_reserve"]({s},{w},{n},{e});
      way["leisure"~"park|nature_reserve"]({s},{w},{n},{e});
      node["amenity"~"restaurant|cafe|bar|fast_food|ferry_terminal|bus_station"]({s},{w},{n},{e});
    );
    out center tags;
    """


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--area", choices=BBOX.keys(), default="zanzibar")
    parser.add_argument("--out", default="data/osm_zanzibar_tourism.csv")
    args = parser.parse_args()

    query = build_query(BBOX[args.area])
    print("Downloading OSM data from Overpass API...")
    r = requests.post(OVERPASS_URL, data={"data": query}, timeout=240)
    r.raise_for_status()
    data = r.json()["elements"]

    rows = []
    for item in data:
        tags = item.get("tags", {})
        name = tags.get("name") or tags.get("name:en") or tags.get("name:sw")
        if not name:
            continue
        lat = item.get("lat") or item.get("center", {}).get("lat")
        lon = item.get("lon") or item.get("center", {}).get("lon")
        category = tags.get("tourism") or tags.get("historic") or tags.get("natural") or tags.get("leisure") or tags.get("amenity") or "place"
        desc = tags.get("description") or tags.get("description:en") or f"{name} is listed in OpenStreetMap as {category}."
        rows.append({
            "id": f"osm-{item.get('type')}-{item.get('id')}",
            "name": name,
            "category": category,
            "region": args.area.title(),
            "latitude": lat,
            "longitude": lon,
            "description_sw": desc,
            "description_en": desc,
            "tips": "Check recent reviews, opening hours, route, and local guidance before visiting.",
            "source": "OpenStreetMap Overpass",
            "keywords": json.dumps(tags, ensure_ascii=False),
            "website": tags.get("website", ""),
        })

    df = pd.DataFrame(rows).drop_duplicates(subset=["id"])
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")

if __name__ == "__main__":
    main()
