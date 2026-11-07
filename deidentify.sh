#!/usr/bin/env bash
# ./deidentify.sh path/to/file.csv [label|mask|x] [model_name]
# Models are the folders inside ./models ; no model name = models/HeMed_Anonimizer_Model (or the only model there).
cd "$(dirname "$0")"
[ -n "$1" ] || { echo "Usage: ./deidentify.sh path/to/file.csv [label|mask|x] [model_name]"; exit 1; }
MODEL=(); [ -n "$3" ] && MODEL=(--model "$3")
exec .venv/bin/python -X utf8 inference/de_identify.py --input "$(realpath "$1")" --mode "${2:-label}" "${MODEL[@]}"
