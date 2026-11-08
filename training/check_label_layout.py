"""Counts-only look at how spans of some labels are laid out: real train split vs the synthetic sets.

Prints numbers only (no text, no ids): words per span, span shapes (digits=9, Hebrew letters=a, Latin=L),
whether the span starts with a Hebrew prefix letter (ה/ב/ל/מ/ו/ש/כ) glued to the word, whether the character
before/after the span is a letter (cut inside a word).  Same output for real and synthetic, to compare conventions.
Usage: python -X utf8 training\\check_label_layout.py [--labels MED_ORG ORG CONTACT] [--data-dir DIR]
"""
import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import find_split_file, load_jsonl  # noqa: E402

DEFAULT_DIR = r"D:\De ID All\NER_anon_dataset\regular"
HERE = Path(__file__).resolve().parent
SYNTH = [HERE / "synth_eval" / "rare_hard.jsonl", HERE / "synth_eval" / "long.jsonl",
         HERE / "synth_eval_ambiguous" / "ambiguous.jsonl"]


def shape(t):
    out = "".join("9" if c.isdigit() else ("a" if "\u05d0" <= c <= "\u05ea" else ("L" if c.isalpha() else c)) for c in t)
    out2 = []
    for c in out:  # collapse runs
        if not out2 or out2[-1] != c:
            out2.append(c)
    return "".join(out2)[:14]


def report(name, recs, labels):
    for lab in labels:
        words = collections.Counter()
        shapes = collections.Counter()
        n = prefix = cut_l = cut_r = 0
        for r in recs:
            t = r["text"]
            for s in r.get("label", []):
                if str(s["label"]).replace("\\", "") != lab:
                    continue
                n += 1
                sp = t[s["start"]:s["end"]]
                words[min(len(sp.split()), 5)] += 1
                shapes[shape(sp)] += 1
                prefix += sp[:1] in "הבלמושכ" and len(sp) > 3
                cut_l += s["start"] > 0 and t[s["start"] - 1].isalpha()
                cut_r += s["end"] < len(t) and t[s["end"]].isalpha()
        print("[%s] %s: %d spans | words/span %s | starts with prefix-like letter %d | cut inside a word: left %d right %d"
              % (name, lab, n, dict(sorted(words.items())), prefix, cut_l, cut_r))
        for sh, c in shapes.most_common(6):
            print("      %-16s %d" % (sh, c))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", default=["MED_ORG", "ORG", "CONTACT"])
    ap.add_argument("--data-dir", default=DEFAULT_DIR)
    a = ap.parse_args()
    p = find_split_file(a.data_dir, "train")
    if p is None:
        sys.exit("no train split in %s" % a.data_dir)
    report("real train", load_jsonl(p), a.labels)
    for f in SYNTH:
        if f.exists():
            report("synth " + f.stem, load_jsonl(f), a.labels)


if __name__ == "__main__":
    main()
