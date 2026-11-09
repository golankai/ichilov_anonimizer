"""Why do organisations fail?  Diagnostic for MED_ORG / ORG.

Part 1 (FAKE set, text is safe): for every MED_ORG / ORG span of inference/dummy_data that is not predicted exactly,
print the span, what the pipeline predicted over it (text + label) and a bit of context.  Strict mode (rules + F8).
Part 2 (REAL data, COUNTS ONLY, optional): for every gold MED_ORG / ORG span of a real split, how the model covered it:
exact / wider / narrower / missed, and which label it gave.  No text is printed or written.
   NOTE: the production model was trained on ALL real data, so on the real split its numbers are too good.
   Use a model trained without that split (a campaign model) for an honest picture.

Usage:  python -X utf8 training/diagnose_medorg.py <model_folder> [--real-split test] [--data-dir DIR] [--name medorg-diagnostic]
Writes training/runs/<name>/medorg_fake.txt and medorg_real_counts.json  (push with training\\push_results.bat).
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "inference"))
sys.path.insert(0, str(ROOT / "training"))

from utils.doc_pipeline import build_doc_pipeline  # noqa: E402
from utils.nlp_utils import get_pipeline, setup_nltk  # noqa: E402
from utils.pipeline_utils import DeidDoc  # noqa: E402

ORG_LABELS = ("MED_ORG", "ORG")
sys.path.insert(0, str(ROOT / "inference"))
from score_dummy import strip_head  # noqa: E402  (fake truth: score the bare name, like the scorer does)


def predict(ner, text):
    pipe = build_doc_pipeline(ner, "label", normalize_dates=False, normalize_ids=False, use_rules=True, consistency=True)
    return pipe(DeidDoc(text=text)).entities


def relation(span, preds):
    """How the predictions cover a gold span: exact / wider / narrower / partial / missed, plus the labels involved."""
    over = [p for p in preds if min(p["end"], span["end"]) > max(p["start"], span["start"])]
    if not over:
        return "missed", []
    labels = sorted({p["label"] for p in over})
    if len(over) == 1:
        p = over[0]
        if (p["start"], p["end"]) == (span["start"], span["end"]):
            return "exact", labels
        if p["start"] <= span["start"] and p["end"] >= span["end"]:
            return "wider", labels
        return "narrower", labels
    covered = set()
    for p in over:
        covered.update(range(max(p["start"], span["start"]), min(p["end"], span["end"])))
    return ("split-full" if len(covered) == span["end"] - span["start"] else "split-partial"), labels


def fake_part(ner, out_file):
    lines, rel_count = [], Counter()
    for f in sorted((ROOT / "inference" / "dummy_data").glob("*.truth.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            text = rec["text"]
            gold = [strip_head(text, s) for s in rec["label"] if s["label"] in ORG_LABELS]
            if not gold:
                continue
            preds = predict(ner, text)
            for s in gold:
                rel, labels = relation(s, preds)
                rel_count[(s["label"], rel, "+".join(labels) or "-")] += 1
                if rel == "exact" and labels == [s["label"]]:
                    continue
                shown = [f"[{text[p['start']:p['end']]}]={p['label']}" for p in preds
                         if min(p["end"], s["end"]) > max(p["start"], s["start"])]
                ctx = text[max(0, s["start"] - 20): s["end"] + 20].replace("\n", " ")
                lines.append(f"{rec['tier']} #{rec['id']}  {s['label']:7} gold=[{text[s['start']:s['end']]}] -> {rel:13} "
                             f"pred={' '.join(shown) or '-'}   ctx: ...{ctx}...")
    lines.append("")
    lines.append("summary (gold label, relation, predicted label(s)): count")
    for k, n in sorted(rel_count.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {k[0]:7} {k[1]:13} {k[2]:20} {n}")
    out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[-12:]))


def real_part(ner, split, data_dir, out_file):
    from dataset import load_splits
    recs = load_splits(data_dir).get(split)
    if not recs:
        print(f"no real split '{split}' in {data_dir}; skipping the real part")
        return
    counts, docs = Counter(), 0
    for rec in recs:
        gold = [s for s in rec["label"] if s["label"] in ORG_LABELS]
        if not gold:
            continue
        docs += 1
        preds = predict(ner, rec["text"])
        for s in gold:
            rel, labels = relation(s, preds)
            counts[f"{s['label']} | {rel} | {'+'.join(labels) or '-'}"] += 1
    out = {"split": split, "documents_with_org": docs, "counts": dict(sorted(counts.items()))}
    out_file.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for k, n in sorted(counts.items()):
        print(f"  {k}: {n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--name", default="medorg-diagnostic")
    ap.add_argument("--real-split", default=None, help="e.g. test; omit to skip the real part")
    ap.add_argument("--data-dir", default=r"D:\De ID All\NER_anon_dataset\regular")
    a = ap.parse_args()
    if not (Path(a.model) / "config.json").is_file():
        sys.exit(f"not a model folder: {a.model}")
    out = ROOT / "training" / "runs" / a.name
    out.mkdir(parents=True, exist_ok=True)
    setup_nltk()
    ner = get_pipeline(a.model)
    fake_part(ner, out / "medorg_fake.txt")
    if a.real_split:
        real_part(ner, a.real_split, a.data_dir, out / "medorg_real_counts.json")
    print(f"done. Results in {out}. Now run training\\push_results.bat")


if __name__ == "__main__":
    main()
