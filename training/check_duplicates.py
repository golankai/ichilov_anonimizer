"""Duplicate and leakage check for the annotated training data. Prints COUNTS ONLY (no text, no ids) by default.

Why: the data may contain the same document twice under different ids (e.g. `second_<id>.txt`), sometimes with
different annotations. If copies land in both train and test, the test score is inflated; if the copies disagree, the
model learns contradictory labels. The split code checks ids only, never texts.

Usage (run where the data is; nothing is modified):
    python training/check_duplicates.py                          # the default split folder of the GPU machine
    python training/check_duplicates.py --data-dir <folder with train/validation/test.jsonl>
    python training/check_duplicates.py --data annotated.jsonl   # one file, no splits

Options:
    --groups FILE       also write the ids of every duplicate group (ids only, no text) to FILE. Ids look like record
                        numbers, so keep that file out of git (it is for the annotation owners).
    --write-clean DIR   write a cleaned copy of the splits to DIR (the originals are never touched):
                          * identical copies (same text, same annotations) -> one kept, in the split that is judged
                            (test > validation > train), so a test document never has a twin in train;
                          * same text, different annotations -> dropped entirely (--conflicts first keeps one);
                          * --drop-near-from-train: train documents that are near duplicates of a validation/test
                            document are dropped as well.
    --near-threshold T  Jaccard similarity of 5-word shingles for "near duplicate" (default 0.9)
"""
import argparse
import json
import re
import sys
import zlib
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import SPLIT_NAMES, dump_jsonl, find_split_file, load_jsonl  # noqa: E402

DEFAULT_DATA_DIR = r"D:\De ID All\NER_anon_dataset\regular"  # keep in sync with train.py
PRIORITY = {"test": 0, "validation": 1, "train": 2, "all": 2}  # which copy survives
SHINGLE = 5
SKETCH = 128
COMMON = 200  # a sketch value shared by more documents than this is boilerplate: ignore it when looking for candidates


def norm_text(t):
    return re.sub(r"\s+", " ", str(t).replace("\u00a0", " ")).strip()


def label_signature(rec):
    """Annotations compared by value (label + the text it covers), so whitespace-only differences do not matter."""
    t = rec["text"]
    return tuple(sorted((s["label"], norm_text(t[s["start"]:s["end"]])) for s in rec["label"]))


def stem(doc_id):
    s = re.sub(r"\.[A-Za-z0-9]{1,4}$", "", str(doc_id))
    return re.sub(r"^(second_|copy_|dup_)", "", s)


def sketch(text):
    words = norm_text(text).split()
    grams = [" ".join(words[i:i + SHINGLE]) for i in range(max(1, len(words) - SHINGLE + 1))]
    return sorted({zlib.crc32(g.encode("utf-8")) for g in grams})[:SKETCH]


def jaccard(a, b):
    if not a or not b:
        return 0.0
    union = sorted(set(a) | set(b))[:SKETCH]
    sa, sb = set(a), set(b)
    return sum(1 for h in union if h in sa and h in sb) / len(union)


def near_pairs(items, threshold):
    """items: [(key, sketch)] for DISTINCT texts. Returns [(i, j, jaccard)] with i < j."""
    sk = [s for _, s in items]
    index = defaultdict(list)
    for i, s in enumerate(sk):
        for h in s:
            index[h].append(i)
    out = []
    for i, s in enumerate(sk):
        shared = Counter(j for h in s if len(index[h]) <= COMMON for j in index[h] if j > i)
        for j, c in shared.items():
            if c >= min(8, len(s)):
                sim = jaccard(s, sk[j])
                if sim >= threshold:
                    out.append((i, j, sim))
    return out


def load_all(args):
    """[(split, record)] with ORIGINAL records (no label cleaning, no ignore list) plus the cleaned label view."""
    docs = []
    if args.data:
        for r in load_jsonl(args.data):
            docs.append(("all", r))
    else:
        for name in SPLIT_NAMES:
            p = find_split_file(args.data_dir, name)
            if p is not None:
                docs += [(name, r) for r in load_jsonl(p)]
    if not docs:
        sys.exit(f"no data found ({args.data or args.data_dir})")
    # cleaned labels (AGE\_ABOVE\_89 -> AGE_ABOVE_89, DAY/MONTH/YEAR -> DATE) only to compare annotations
    clean = {id(r): load_records_one(r) for _, r in docs}
    return docs, clean


def load_records_one(record):
    from dataset import normalize_record
    return normalize_record(record)


def pct(n, d):
    return f"{100 * n / d:.1f}%" if d else "-"


def conflict_kind(sigs):
    """How do the annotations of one text differ? (counts only)"""
    first = sigs[0]
    kinds = set()
    for other in sigs[1:]:
        a, b = Counter(first), Counter(other)
        if a == b:
            continue
        only_a, only_b = a - b, b - a
        values_a, values_b = {v for _, v in only_a}, {v for _, v in only_b}
        if values_a & values_b:
            kinds.add("same words, different label type")
        else:
            kinds.add("spans present in only one version")
    return kinds or {"identical"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--data")
    ap.add_argument("--groups")
    ap.add_argument("--write-clean")
    ap.add_argument("--conflicts", choices=["drop", "first"], default="drop")
    ap.add_argument("--drop-near-from-train", action="store_true")
    ap.add_argument("--near-threshold", type=float, default=0.9)
    args = ap.parse_args(argv)

    docs, clean = load_all(args)
    split_names = [n for n in ("train", "validation", "test", "all") if any(s == n for s, _ in docs)]
    per_split = Counter(s for s, _ in docs)
    print(f"DOCUMENTS: {len(docs)} (" + ", ".join(f"{n} {per_split[n]}" for n in split_names) + ")")

    # ---- ids
    ids = [r["id"] for _, r in docs]
    id_counts = Counter(ids)
    prefixed = [i for i in ids if re.match(r"^(second_|copy_|dup_)", str(i))]
    by_stem = defaultdict(list)
    for i in set(ids):
        by_stem[stem(i)].append(i)
    twin_ids = [s for s, v in by_stem.items() if len(v) > 1]
    print(f"IDS: {len(id_counts)} distinct; {sum(c > 1 for c in id_counts.values())} ids used more than once; "
          f"{len(prefixed)} ids with a second_/copy_/dup_ prefix; {len(twin_ids)} base ids that exist in more than one form")

    # ---- exact duplicates (text equal after whitespace normalisation)
    groups = defaultdict(list)
    for k, (split, r) in enumerate(docs):
        groups[norm_text(r["text"])].append(k)
    dup_groups = {t: ks for t, ks in groups.items() if len(ks) > 1}
    redundant = sum(len(ks) - 1 for ks in dup_groups.values())
    identical, conflicting = [], []
    kind_count = Counter()
    for t, ks in dup_groups.items():
        sigs = [label_signature(clean[id(docs[k][1])]) for k in ks]
        if len(set(sigs)) == 1:
            identical.append(ks)
        else:
            conflicting.append(ks)
            for kd in conflict_kind(sigs):
                kind_count[kd] += 1
    print("\nEXACT DUPLICATES (same text after whitespace normalisation)")
    print(f"  distinct texts: {len(groups)}; texts that occur more than once: {len(dup_groups)} "
          f"(covering {sum(len(k) for k in dup_groups.values())} documents, {redundant} of them redundant = "
          f"{pct(redundant, len(docs))} of the data)")
    print(f"  group size histogram: {dict(sorted(Counter(len(k) for k in dup_groups.values()).items()))}")
    print(f"  identical annotations: {len(identical)} groups ({sum(len(k) - 1 for k in identical)} redundant documents)")
    print(f"  CONFLICTING annotations: {len(conflicting)} groups ({sum(len(k) for k in conflicting)} documents)")
    for kd, c in kind_count.most_common():
        print(f"      {kd}: {c} groups")
    same_stem = sum(1 for ks in dup_groups.values()
                    if len({stem(docs[k][1]["id"]) for k in ks}) == 1)
    print(f"  groups whose ids share one base id (second_<id> style): {same_stem} of {len(dup_groups)}")

    # ---- leakage across splits
    if len(split_names) > 1:
        print("\nLEAKAGE ACROSS SPLITS (exact text)")
        in_split = defaultdict(set)
        for t, ks in groups.items():
            for k in ks:
                in_split[t].add(docs[k][0])
        spanning = [t for t, s in in_split.items() if len(s) > 1]
        print(f"  texts present in more than one split: {len(spanning)}")
        for a, b in (("test", "train"), ("validation", "train"), ("test", "validation")):
            if per_split[a] and per_split[b]:
                n = sum(1 for k, (s, r) in enumerate(docs) if s == a and
                        any(docs[j][0] == b for j in groups[norm_text(r["text"])]))
                print(f"  {a} documents whose text also occurs in {b}: {n} of {per_split[a]} ({pct(n, per_split[a])})")

    # ---- near duplicates among distinct texts
    reps = [(t, ks) for t, ks in groups.items()]
    pairs = near_pairs([(i, sketch(t)) for i, (t, _) in enumerate(reps)], args.near_threshold)
    near_docs = set()
    cross = 0
    near_eval_train = set()
    for i, j, sim in pairs:
        ki, kj = reps[i][1], reps[j][1]
        near_docs.update(ki + kj)
        si, sj = {docs[k][0] for k in ki}, {docs[k][0] for k in kj}
        if len(si | sj) > 1:
            cross += 1
        for ev in ("validation", "test"):
            if ev in si and "train" in sj:
                near_eval_train.update(k for k in ki if docs[k][0] == ev)
            if ev in sj and "train" in si:
                near_eval_train.update(k for k in kj if docs[k][0] == ev)
    print(f"\nNEAR DUPLICATES (different text, {SHINGLE}-word-shingle Jaccard >= {args.near_threshold})")
    print(f"  pairs of distinct texts: {len(pairs)}; documents involved: {len(near_docs)}; pairs across splits: {cross}")
    if len(split_names) > 1:
        for ev in ("validation", "test"):
            n = sum(1 for k in near_eval_train if docs[k][0] == ev)
            print(f"  {ev} documents with a near-duplicate in train: {n} of {per_split[ev]} ({pct(n, per_split[ev])})")

    # ---- the ignore list
    from dataset import load_ignorable_ids
    ign = load_ignorable_ids()
    present = [i for i in ign if i in id_counts]
    dup_ids = {docs[k][1]["id"] for ks in dup_groups.values() for k in ks}
    print(f"\nIGNORE LIST (training/ignorable_ids.txt): {len(ign)} ids listed, {len(present)} present in this data, "
          f"{sum(1 for i in present if i in dup_ids)} of those are in a duplicate group. "
          f"(The default `train` mode reads pre-made split folders and does not apply this list.)")

    # ---- what a cleaning would leave
    keep = set(range(len(docs)))
    removed = Counter()
    for t, ks in dup_groups.items():
        order = sorted(ks, key=lambda k: (PRIORITY[docs[k][0]], str(docs[k][1]["id"])))
        sigs = {label_signature(clean[id(docs[k][1])]) for k in ks}
        if len(sigs) == 1 or args.conflicts == "first":
            for k in order[1:]:
                keep.discard(k)
                removed["redundant copy"] += 1
        else:
            for k in ks:
                keep.discard(k)
                removed["conflicting annotations"] += 1
    if args.drop_near_from_train:
        for i, j, sim in pairs:
            ki, kj = reps[i][1], reps[j][1]
            for a, b in ((ki, kj), (kj, ki)):
                if any(docs[k][0] in ("validation", "test") for k in a):
                    for k in b:
                        if docs[k][0] == "train" and k in keep:
                            keep.discard(k)
                            removed["train near-duplicate of an evaluation document"] += 1
    left = Counter(docs[k][0] for k in keep)
    print("\nIF CLEANED" + (f" ({args.conflicts} for conflicts" + (", near twins removed from train" if args.drop_near_from_train else "") + ")") + ":")
    print("  removed: " + (", ".join(f"{v} {k}" for k, v in removed.items()) or "nothing"))
    print("  documents left: " + ", ".join(f"{n} {left[n]}" for n in split_names) + f" (total {len(keep)})")

    if args.groups:
        with open(args.groups, "w", encoding="utf-8") as f:
            for t, ks in dup_groups.items():
                sigs = {label_signature(clean[id(docs[k][1])]) for k in ks}
                f.write(json.dumps({"kind": "identical" if len(sigs) == 1 else "conflicting",
                                    "docs": [{"id": docs[k][1]["id"], "split": docs[k][0]} for k in ks]},
                                   ensure_ascii=False) + "\n")
        print(f"\nwrote duplicate groups (ids only) to {args.groups}: keep it out of git")
    if args.write_clean:
        out = Path(args.write_clean)
        for name in split_names:
            recs = [docs[k][1] for k in sorted(keep) if docs[k][0] == name]
            dump_jsonl(recs, out / ("train.jsonl" if name == "all" else f"{name}.jsonl"))
        print(f"wrote the cleaned data to {out} (originals untouched)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
