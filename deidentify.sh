#!/usr/bin/env bash
# ./deidentify.sh path/to/file.csv [label|mask|x]   (model read from ./HeMed_Anonimizer_Model)
cd "$(dirname "$0")"
[ -n "$1" ] || { echo "Usage: ./deidentify.sh path/to/file.csv [label|mask|x]"; exit 1; }
exec .venv/bin/python -X utf8 inference/de_identify.py --input "$(realpath "$1")" --mode "${2:-label}"
