"""
Run the dataset collection scripts that are safe to automate.

Run from project root:
    python scripts/run_data_builders.py

UNDP GeoHub sometimes requires copying the dataset API/File URL from the website,
so that script is not forced here. Run it separately with --url if needed.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

commands = [
    [sys.executable, str(ROOT / "scripts" / "build_osm_dataset.py"), "--area", "zanzibar", "--out", str(ROOT / "data" / "osm_zanzibar_tourism.csv")],
    [sys.executable, str(ROOT / "scripts" / "fetch_geonames_tanzania.py"), "--out", str(ROOT / "data" / "geonames_tanzania.csv")],
]

for cmd in commands:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

print("Done. Restart app.py so it reloads the new CSV files.")
