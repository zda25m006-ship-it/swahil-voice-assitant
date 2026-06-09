"""
Download Tanzania places from GeoNames country dump and convert to app schema.

Run:
    python scripts/fetch_geonames_tanzania.py --out data/geonames_tanzania.csv
"""
from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

URL = "https://download.geonames.org/export/dump/TZ.zip"
COLS = [
    "geonameid", "name", "asciiname", "alternatenames", "latitude", "longitude",
    "feature_class", "feature_code", "country_code", "cc2", "admin1_code", "admin2_code",
    "admin3_code", "admin4_code", "population", "elevation", "dem", "timezone", "modification_date"
]

FEATURE_MEANING = {
    "P": "city/town/village",
    "A": "administrative region",
    "T": "mountain/hill/terrain feature",
    "H": "water feature",
    "L": "park/area",
    "S": "spot/building/facility",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/geonames_tanzania.csv")
    parser.add_argument("--min_population", type=int, default=1000)
    args = parser.parse_args()

    print("Downloading GeoNames Tanzania dump...")
    z = zipfile.ZipFile(io.BytesIO(requests.get(URL, timeout=180).content))
    with z.open("TZ.txt") as f:
        raw = pd.read_csv(f, sep="\t", names=COLS, low_memory=False)

    # Keep bigger populated places and important named features to avoid huge noisy RAG.
    raw["population"] = pd.to_numeric(raw["population"], errors="coerce").fillna(0)
    df = raw[(raw["population"] >= args.min_population) | (raw["feature_class"].isin(["T", "H", "L"]))].copy()
    df["category"] = df["feature_class"].map(FEATURE_MEANING).fillna(df["feature_code"])
    df["region"] = "Tanzania"
    df["description_en"] = df.apply(lambda r: f"{r['name']} is a {r['category']} in Tanzania. Population: {int(r['population'])}.", axis=1)
    df["description_sw"] = df["description_en"]
    df["tips"] = "Use this as a geographic place reference; verify tourism services separately."
    df["source"] = "GeoNames Tanzania"
    df["keywords"] = df["alternatenames"].fillna("").astype(str) + " " + df["feature_code"].fillna("").astype(str)
    df["id"] = "geonames-" + df["geonameid"].astype(str)

    out_cols = ["id", "name", "category", "region", "latitude", "longitude", "description_sw", "description_en", "tips", "source", "keywords"]
    out = df[out_cols].drop_duplicates(subset=["id"])
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Saved {len(out)} rows to {out_path}")

if __name__ == "__main__":
    main()
