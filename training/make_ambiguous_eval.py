"""Generate the AMBIGUOUS-WORDS evaluation set: training/synth_eval_ambiguous/ambiguous.jsonl (+ ambiguous.meta.json)

Source of the words: ``inference/lexicons/ambiguous_words.tsv`` (words that can be PHI and also mean something else).
For every word the set has, as far as the list says it applies:
  name   the word used as a value that MUST be masked (first/last name, month as DATE, LOC, MED_ORG, ORG);
  plain  the word in its ordinary sense (``plain_example`` column): NOTHING may be masked (a trap);
  mixed  both in one document (name sentence + plain sentence, in alternating order): only the name is masked.
          This is the hard case for any "mask every other occurrence" rule.

Same shape as ``training/synth_eval/*.jsonl`` (``id``, ``text``, ``label`` with start/end exclusive/label), so
``train.py --synth-eval-dir training/synth_eval_ambiguous`` scores it with the existing code. It lives in its own
folder on purpose: ``training/synth_eval`` has a fixed set of files that tests check. The sidecar
``ambiguous.meta.json`` maps each id to word / kind / tag / policy, for per-word and per-kind reporting.

Conventions (as in ``inference/dummy_data/README.md``): a Hebrew prefix is outside the entity, a title is outside, a
medical organisation / company includes its head noun. ``policy: true`` marks conventions we only assumed (month alone
or month + year as a DATE span). Everything is invented. Deterministic (fixed seed);
``training/tests/test_ambiguous_eval.py`` checks that re-generating reproduces the committed files.

Run:  python training/make_ambiguous_eval.py
"""
import csv
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "inference"))
from make_dummy_data import FILLER, FIRST, LAST, E, build  # noqa: E402

LEXICON = ROOT / "inference" / "lexicons" / "ambiguous_words.tsv"
OUT = HERE / "synth_eval_ambiguous"
SEED = 20261109


def load_lexicon(path=LEXICON):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


HEBREW_MONTHS = {"אדר", "ניסן", "אייר", "סיון", "סיוון", "תמוז", "אב", "אלול", "תשרי", "טבת", "שבט", "כסלו", "חשוון"}
HEBREW_YEARS = ["תשפ״ג", "תשפ״ד", "תשפ״ה", "תשע״ט"]


def pick(rng, pool, avoid):
    while True:
        x = rng.choice(pool)
        if x != avoid:
            return x


# --------------------------------------------------------------------------------------------- name-use templates
# Each returns (parts for build(), tag, policy). `rng` only picks the invented co-names/years.
def name_docs(rng, row):
    w, head = row["word"], row["phi_head"]
    out = []
    for lab in row["can_be_phi_as"].split("|"):
        if lab == "FIRST_NAME":
            last = pick(rng, LAST, w)
            out.append((["שם המטופל: ", E("FIRST_NAME", w), " ", E("LAST_NAME", last), "."], "first-name-header", False))
            last = pick(rng, LAST, w)
            out.append(([E("FIRST_NAME", w), " ", E("LAST_NAME", last),
                         " הגיע לבדיקה במרפאה בעקבות כאבים בחזה."], "first-name-running", False))
        elif lab == "LAST_NAME":
            first = pick(rng, FIRST, w)
            out.append((["שם המטופל: ", E("FIRST_NAME", first), " ", E("LAST_NAME", w), "."], "last-name-header", False))
            first = pick(rng, FIRST, w)
            out.append((["מר ", E("FIRST_NAME", first), " ", E("LAST_NAME", w),
                         " הגיע לבדיקה במרפאה בעקבות כאבים בחזה."], "last-name-running", False))
        elif lab == "DATE":
            year = rng.choice(HEBREW_YEARS) if w in HEBREW_MONTHS else rng.randint(1995, 2024)
            out.append((["התסמינים החלו ב", E("DATE", w, policy=True), "."], "month-alone", True))
            out.append((["נבדק לאחרונה ב", E("DATE", f"{w} {year}", policy=True), " ונמצא תקין."], "month-year", True))
        elif lab == "LOC":
            out.append((["המטופל מתגורר ב", E("LOC", w), "."], "loc-prefixed-ב", False))
            out.append((["הופנה מ", E("LOC", w), " לבירור נוסף."], "loc-prefixed-מ", False))
        elif lab == "MED_ORG":
            org = f"{head} {w}"
            out.append((["הופנה ל", E("MED_ORG", org), " לצורך המשך בירור."], "med-org-prefixed-ל", False))
            out.append((["אושפז ב", E("MED_ORG", org), " בשבוע שעבר."], "med-org-prefixed-ב", False))
        elif lab == "ORG":
            org = f"{head} {w}"
            out.append((["המטופל עובד ב", E("ORG", org), "."], "org-prefixed-ב", False))
            out.append((["הבקשה הוגשה אל ", E("ORG", org), " לפני חודשיים."], "org-plain", False))
    return out


def render():
    rng = random.Random(SEED)
    rows = load_lexicon()
    docs = []  # (text, spans, meta)

    def add(parts, word, kind, tag, policy=False):
        text, spans, _keeps = build(*parts)
        if rng.random() < 0.5:  # a neutral sentence before, so the word is not always at the very start
            lead = rng.choice(FILLER) + " "
            text = lead + text
            spans = [{**s, "start": s["start"] + len(lead), "end": s["end"] + len(lead)} for s in spans]
        for s in spans:
            assert s["end"] <= len(text)
        docs.append((text, spans, {"word": word, "kind": kind, "tag": tag, "policy": policy}))

    plain_done = set()
    for row in rows:
        w = row["word"]
        for parts, tag, policy in name_docs(rng, row):
            add(parts, w, "name", tag, policy)
        plain = row["plain_example"]
        if plain and w not in plain_done:
            plain_done.add(w)
            add([plain], w, "plain", "plain-use")
        # mixed: the same word is a person's name in one sentence and an ordinary word in another
        person = [l for l in row["can_be_phi_as"].split("|") if l in ("FIRST_NAME", "LAST_NAME")]
        if plain and person:
            if person[0] == "FIRST_NAME":
                name_parts = ["המטופל ", E("FIRST_NAME", w), " ", E("LAST_NAME", pick(rng, LAST, w)), " הגיע למרפאה."]
            else:
                name_parts = ["המטופל ", E("FIRST_NAME", pick(rng, FIRST, w)), " ", E("LAST_NAME", w), " הגיע למרפאה."]
            if len(docs) % 2:
                add([name_parts, " ", plain], w, "mixed", f"{person[0].lower()}-then-plain")
            else:
                add([plain, " ", name_parts], w, "mixed", f"plain-then-{person[0].lower()}")
    return docs


def serialize(docs):
    lines, meta = [], {}
    for i, (text, spans, m) in enumerate(docs, 1):
        lines.append(json.dumps(
            {"id": i, "text": text, "label": [{"start": s["start"], "end": s["end"], "label": s["label"]} for s in spans]},
            ensure_ascii=False))
        meta[str(i)] = m
    data = ("\n".join(lines) + "\n").encode("utf-8")
    meta_bytes = (json.dumps(meta, ensure_ascii=False, indent=1) + "\n").encode("utf-8")
    return data, meta_bytes


def main(out=OUT):
    out = Path(out)
    out.mkdir(exist_ok=True)
    docs = render()
    data, meta = serialize(docs)
    (out / "ambiguous.jsonl").write_bytes(data)  # utf-8, LF: identical on every machine
    (out / "ambiguous.meta.json").write_bytes(meta)
    kinds, labels = {}, {}
    for _t, spans, m in docs:
        kinds[m["kind"]] = kinds.get(m["kind"], 0) + 1
        for s in spans:
            labels[s["label"]] = labels.get(s["label"], 0) + 1
    print(f"{len(docs)} documents: " + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items())))
    print("entities: " + ", ".join(f"{k} {v}" for k, v in sorted(labels.items())))


if __name__ == "__main__":
    main()
