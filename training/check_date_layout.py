"""Counts-only look at how DATE-like spans are laid out in the real train split.

Answers: are dates annotated as one span (12/03/2020) or as separate DAY/MONTH/YEAR pieces?
Prints numbers only: no text, no ids.  Usage: python -X utf8 training\check_date_layout.py [--data-dir DIR]
"""
import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import find_split_file, load_jsonl  # noqa: E402

DEFAULT_DIR = r"D:\De ID All\NER_anon_dataset\regular"
DATE_RAW = {"DATE", "DAY", "MONTH", "YEAR"}


def find_train(d):
    p = find_split_file(d, "train")
    if p is None:
        sys.exit("no train split in %s" % d)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DEFAULT_DIR)
    a = ap.parse_args()
    recs = load_jsonl(find_train(a.data_dir))
    raw = collections.Counter()
    lens = []
    adjacent = 0
    total = 0
    shapes = collections.Counter()
    for r in recs:
        spans = sorted(
            (s for s in r.get("label", []) if str(s["label"]).replace("\\", "") in DATE_RAW),
            key=lambda s: s["start"],
        )
        prev = None
        for s in spans:
            lab = str(s["label"]).replace("\\", "")
            raw[lab] += 1
            total += 1
            lens.append(s["end"] - s["start"])
            if prev is not None and s["start"] - prev["end"] <= 1:
                adjacent += 1
            prev = s
            txt = r["text"][s["start"]:s["end"]]
            shape = "".join("9" if c.isdigit() else ("a" if c.isalpha() else c) for c in txt)
            shapes[shape[:12]] += 1
    print("documents:", len(recs))
    print("raw date-like labels:", dict(raw))
    print("date-like spans:", total, "| directly adjacent to the previous one (gap <= 1 char):", adjacent)
    if lens:
        lens.sort()
        print("span length chars: median", lens[len(lens) // 2], "| max", lens[-1],
              "| <=2 chars:", sum(1 for x in lens if x <= 2), "| >=8 chars:", sum(1 for x in lens if x >= 8))
    print("most common span shapes (digits=9, letters=a):")
    for sh, n in shapes.most_common(12):
        print("  %-14s %d" % (sh, n))


if __name__ == "__main__":
    main()
