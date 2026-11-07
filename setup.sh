#!/usr/bin/env bash
# One-time setup (Linux/macOS): ./setup.sh
set -e
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python inference/check_setup.py
