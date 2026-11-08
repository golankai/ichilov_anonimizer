"""Score pipeline outputs against the ground truth of the fake test set (inference/dummy_data/*.truth.jsonl).

Usage:  python inference/score_dummy.py [--results-dir inference/dummy_results] [--mode label] [--only t2_formats]

Reads <tier>_deid_<mode>_pipeline*.json (written by de_identify.py) and prints, per tier / label / situation tag:
  masked   share of truth spans whose every character is covered by some predicted entity (anything less is a leak)
  partial  spans only partly covered (a leak of the uncovered part)      missed  spans not touched at all
  label    share of covered spans whose best-overlapping prediction has the right label
  exact    share of spans predicted with exactly the right boundaries and label
  survive  hospital-style check: the truth string still appears in the de-identified text (a leak)
and the over-masking side: predictions that hit a declared trap, and extra predictions on text that is not PHI.

Offsets: the pipeline normalises dates/IDs first, so for some documents the JSON offsets no longer fit the original
text ("drift"). Those documents are counted separately and scored by the `survive` check only.
Prints no patient text; the test set is fake, so example strings are shown for the failures.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRUTH_DIR = ROOT / "dummy_data"
DEFAULT_RESULTS = ROOT / "dummy_results"
DEID_KEYS = ("deid text", "de_identified_text", "deid_text")


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def find_output(results_dir, stem, mode):
    hits = sorted(Path(results_dir).glob(f"{stem}_deid_{mode}_pipeline*.json"))
    return hits[0] if hits else None


def cover(span, preds):
    """(covered chars, best overlapping prediction)"""
    covered, best, best_ov = set(), None, 0
    for p in preds:
        a, b = max(span["start"], p["start"]), min(span["end"], p["end"])
        if b > a:
            covered.update(range(a, b))
            if b - a > best_ov:
                best, best_ov = p, b - a
    return len(covered), best


def overlaps(a, b):
    return min(a["end"], b["end"]) > max(a["start"], b["start"])


def score_doc(truth, out):
    text, preds = truth["text"], out.get("entities", [])
    deid = next((out[k] for k in DEID_KEYS if k in out), "")
    drift = any(p.get("text") != text[p["start"]:p["end"]] for p in preds)
    keep_strings = {text[k["start"]:k["end"]] for k in truth["keep"]}
    rows = []
    for s in truth["label"]:
        value = text[s["start"]:s["end"]]
        row = {"label": s["label"], "tag": s.get("tag") or "-", "policy": s.get("policy", False), "value": value,
               "doc": truth["id"]}
        # a string that is also a declared trap (e.g. the name "לב" next to the organ "לב") cannot be string-checked
        checkable = len(value) >= 3 and value not in keep_strings
        row["survive"] = (value in deid) if checkable else None
        if not drift:
            n, best = cover(s, preds)
            row["status"] = "masked" if n == s["end"] - s["start"] else ("partial" if n else "missed")
            row["label_ok"] = bool(best and best["label"] == s["label"]) if n else False
            row["exact"] = any(p["start"] == s["start"] and p["end"] == s["end"] and p["label"] == s["label"]
                               for p in preds)
        else:
            row["status"] = row["label_ok"] = row["exact"] = None
        rows.append(row)
    traps, extras = [], []
    if not drift:
        for k in truth["keep"]:
            hit = [p for p in preds if overlaps(k, p)]
            traps.append({"note": k["note"], "value": text[k["start"]:k["end"]], "hit": bool(hit),
                          "label": hit[0]["label"] if hit else None, "doc": truth["id"]})
        for p in preds:
            if not any(overlaps(p, s) for s in truth["label"]) and not any(overlaps(p, k) for k in truth["keep"]):
                extras.append({"value": text[p["start"]:p["end"]], "label": p["label"], "doc": truth["id"]})
    return rows, traps, extras, drift


def pct(n, d):
    return "  -  " if not d else f"{100 * n / d:5.1f}"


def table(title, groups):
    print(f"\n{title}")
    print(f"  {'':34s}{'n':>5s}{'masked%':>9s}{'partial':>8s}{'missed':>7s}{'label%':>8s}{'exact%':>8s}{'survive':>9s}")
    for key, rows in sorted(groups.items()):
        sc = [r for r in rows if r["status"] is not None]
        m = sum(r["status"] == "masked" for r in sc)
        cov = [r for r in sc if r["status"] != "missed"]
        sv = [r for r in rows if r["survive"] is not None]
        print(f"  {key:34s}{len(rows):5d}{pct(m, len(sc)):>9s}{sum(r['status'] == 'partial' for r in sc):8d}"
              f"{sum(r['status'] == 'missed' for r in sc):7d}{pct(sum(r['label_ok'] for r in cov), len(cov)):>8s}"
              f"{pct(sum(r['exact'] for r in sc), len(sc)):>8s}{sum(bool(r['survive']) for r in sv):>5d}/{len(sv):<3d}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    ap.add_argument("--mode", default="label")
    ap.add_argument("--only", help="score just this tier (file stem), e.g. t2_formats")
    ap.add_argument("--show", type=int, default=12, help="how many failing examples to list per section")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    by_label, by_tag, by_tier = defaultdict(list), defaultdict(list), defaultdict(list)
    all_traps, all_extras, drift_docs, scored_tiers = [], [], defaultdict(int), []
    for tf in sorted(TRUTH_DIR.glob("*.truth.jsonl")):
        stem = tf.name[: -len(".truth.jsonl")]
        if args.only and stem != args.only:
            continue
        out_path = find_output(args.results_dir, stem, args.mode)
        if not out_path:
            print(f"[skip] {stem}: no {stem}_deid_{args.mode}_pipeline*.json in {args.results_dir}")
            continue
        outs = {o["id"]: o for o in json.load(open(out_path, encoding="utf-8"))}
        scored_tiers.append(stem)
        for t in load_jsonl(tf):
            if t["id"] not in outs:
                print(f"[warn] {stem} id {t['id']} missing from the output")
                continue
            rows, traps, extras, drift = score_doc(t, outs[t["id"]])
            drift_docs[stem] += drift
            for r in rows:
                r["tier"] = stem
                by_label[r["label"]].append(r)
                by_tag[f"{r['label']}: {r['tag']}" + (" (policy)" if r["policy"] else "")].append(r)
                by_tier[stem].append(r)
            for x in traps + extras:
                x["tier"] = stem
            all_traps += traps
            all_extras += extras
    if not scored_tiers:
        print("nothing to score: run inference\\run_dummy.bat first")
        return 1

    table("PER TIER", by_tier)
    table("PER LABEL (all tiers)", by_label)
    table("PER SITUATION (label: tag)", by_tag)

    print("\nOVER-MASKING (things that should stay readable)")
    hit = [t for t in all_traps if t["hit"]]
    print(f"  declared traps hit: {len(hit)} of {len(all_traps)}")
    for t in hit[: args.show]:
        print(f"    [{t['tier']} #{t['doc']}] '{t['value']}' masked as {t['label']}  ({t['note']})")
    print(f"  extra predictions on non-PHI text: {len(all_extras)}")
    for x in all_extras[: args.show]:
        print(f"    [{x['tier']} #{x['doc']}] '{x['value']}' -> {x['label']}")

    bad = [r for rows in by_label.values() for r in rows if r["status"] in ("partial", "missed")]
    print(f"\nLEAKS: {len(bad)} truth spans are not fully masked (listing {min(len(bad), args.show * 2)})")
    for r in bad[: args.show * 2]:
        print(f"    [{r['tier']} #{r['doc']}] {r['status']:7s} {r['label']:13s} {r['tag']:28s} '{r['value']}'")
    drift_total = sum(drift_docs.values())
    if drift_total:
        print(f"\nOFFSET DRIFT: {drift_total} documents have entity offsets that do not fit the original text "
              f"({dict(drift_docs)}). They were scored by the 'survive' check only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
