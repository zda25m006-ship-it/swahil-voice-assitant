#!/usr/bin/env bash
# SautiSafari Advanced - Linux/macOS/Git Bash launcher
set -e
[ -d .venv ] || python -m venv .venv
# shellcheck disable=SC1091
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; else source .venv/Scripts/activate; fi
python -m pip install --upgrade pip
pip install -r requirements.txt
[ -f .env ] || cp .env.example .env
python app.py
