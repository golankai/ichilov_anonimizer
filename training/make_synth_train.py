"""Generate SYNTHETIC TRAINING documents: training/synth_train/synth_train.jsonl (+ meta + word split)

Purpose: help the base model with what rules cannot solve - first and last names, above all the AMBIGUOUS ones
(words that are also ordinary words, months, places: ``inference/lexicons/ambiguous_words.tsv``). Short documents
(one to three sentences), mixed with the real training split by ``train.py --extra-train``.

Design (read this before trusting any number):
  * Word split: 70% of the ambiguous words are "seen" (get training documents), 30% are "held out" (none). The split is a
    fixed hash of the word (``word_split.json``), so it never changes. ``train.py`` reports the evaluation set
    ``ambiguous`` additionally as ``ambig_seen`` and ``ambig_heldout``. Held-out words show whether the model learned
    "a name needs context" or only memorised words.
  * Disjoint from the evaluation sets: other sentence frames than ``make_synth_eval.py`` / ``make_ambiguous_eval.py``,
    other first/last names (checked by ``tests/test_synth_train.py``), and the plain-use sentences are generic frames per
    category, NOT the lexicon's ``plain_example`` (that would put the evaluation sentences into training).
    Weakness: generic frames are less natural than the lexicon sentences.
  * Annotation conventions are the REAL ones: dates are DAY/MONTH/YEAR pieces, a Hebrew prefix letter is outside the
    entity, MED_ORG / ORG are the bare name only (the head word such as בית החולים is plain text).
  * Not included: CONTACT, INNER_ID, AGE_ABOVE_89, LOC beyond ambiguous words (rules and real data cover them).
Everything is invented and deterministic (fixed seed). ``tests/test_synth_train.py`` checks the committed files.

Run:  python training/make_synth_train.py [--n 3000]      (experiments: 70% of the words)
      python training/make_synth_train.py --all-words      (PRODUCTION: every word seen; train-all only, after the recipe is chosen)
"""
import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "inference"))
sys.path.insert(0, str(HERE))
from make_dummy_data import FILLER, FIRST, HYPHEN_LAST, LAST, E, build, israeli_id, rdate  # noqa: E402
from make_synth_eval import DATE_FORMATS, FILL, split_date_spans  # noqa: E402

LEXICON = ROOT / "inference" / "lexicons" / "ambiguous_words.tsv"
OUT = HERE / "synth_train"
SEED = 20261110
DEFAULT_N = 3000
SEEN_SHARE = 0.7  # of the lexicon words

# Names that appear nowhere in the evaluation sets / fake pipeline set (filtered against them below).
_FIRST_POOL = """אילן גדי זאב חגי טוביה יהודה לירן מרדכי נתן עמית פנחס צבי קובי רפאל שלמה תומר אריאל בועז גלעד דניאל ורד
זיוה חנה טובה יהודית לאה מרים נורית סיגל עדי פנינה צופיה קרן רבקה שושנה תהילה אביגיל בתיה גילה דליה הילה זהבה חיה ימית
כוכבה ליאת מורן ניצן סמדר ענת פזית אמנון בנימין גבריאל דרור הרצל ויקטור זבולון יגאל כפיר מנחם נחמן סער עידו פרץ קלמן
רוני שמעון תמיר אושרית בלהה גאולה דבורה הדר ורדית זיסל חוה טליה יפעת כרמית לימור מיטל נירית סיוון עופרה פולינה צילה קסניה
רינת שרון תקוה אביעד בר דור הראל וולדימיר ינון לביא מאור נדב אלעד אייל""".split()
_LAST_POOL = """אזולאי בן דוד גבאי דדון הרוש ויזמן זכאי חזן טובול יפרח כץ לביא מלכה נחום סבג עמר פינטו צדוק קורן רוזן שטרית
תורגמן אשכנזי בוזגלו גרינברג דוידוב הורוביץ וקנין זילברמן חכמון טייב ישראלי כרמי לנדאו מימון נעים סויסה עטיה פוגל צרפתי
קדוש רביבו תמם אדרי בלומנפלד גולדשטיין דיין הלוי ווקנין זיו חביב טל יוסף כהנא לוין מנסור נחמיAS סלומון עזריה פלדמן צור קליין
רוטשטיין שחם תבורי אלבז בן שושן גרוסמן דאהר הכט וינברג זוהר חיים טנא ילין כספי למברגר מילר ניסים סעדה עוז פולק צמח קסלר
רייך שוורץ""".replace("נחמיAS", "נחמיאס").split()

# (head word, bare name): the head is plain text, only the name is the entity (real convention).
MED_ORGS = [("בית החולים", "שערי צדק"), ("בית החולים", "מעייני הישועה"), ("בית החולים", "קפלן"), ("בית החולים", "אלישע"),
            ("בית החולים", "הרצוג"), ("בית החולים", "בילינסון"), ("מרכז רפואי", "תל השומר"), ("בית החולים", "יוספטל"),
            ("בית החולים", "צאנז"), ("ביה\"ח", "קפלן"), ("ביה\"ח", "הדסה"), ("קופ\"ח", "כללית"), ("קופ\"ח", "מאוחדת"),
            ("קופת חולים", "לאומית"), ("מרפאת", "מאוחדת"), ("בית החולים", "נהריה"), ("מכון", "דוידוף")]
ORGS = [("חברת", "אינטל"), ("בנק", "מזרחי"), ("בנק", "לאומי"), ("עיריית", "חיפה"), ("חברת", "שטראוס"), ("חברת", "אלקטרה"),
        ("משרד", "הביטחון"), ("משרד", "הפנים"), ("רשת", "רמי לוי"), ("אוניברסיטת", "בר אילן"), ("חברת", "נטפים"),
        ("חברת", "סופר פארם"), ("מועצה אזורית", "גליל עליון"), ("חברת", "אגד"), ("בית ספר", "אורט")]
TOWNS_T = ["אילת", "בית שאן", "קריית שמונה", "מודיעין", "חדרה", "כפר סבא", "אשקלון", "נהריה", "טירת כרמל", "יקנעם", "ערד",
           "בית שמש", "קריית מלאכי"]


def load_lexicon(path=LEXICON):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def word_is_seen(word):
    """Fixed 70/30 split by hash of the word: stable when the lexicon grows, and the same on every machine."""
    h = int(hashlib.sha1(word.encode("utf-8")).hexdigest(), 16) % 100
    return h < int(SEEN_SHARE * 100)


def word_split(rows=None):
    words = sorted({r["word"] for r in (rows or load_lexicon())})
    return {"seen": [w for w in words if word_is_seen(w)], "heldout": [w for w in words if not word_is_seen(w)]}


def _blocked_names():
    """Everything the evaluation sets / fake pipeline set may contain: the pools below must not use it."""
    from make_ambiguous_eval import HEBREW_MONTHS  # noqa: F401  (imported for its side effect: path setup)
    blocked = set(FIRST) | set(LAST) | set(HYPHEN_LAST) | {r["word"] for r in load_lexicon()}
    return blocked


def name_pools():
    blocked = _blocked_names()
    first = sorted({n for n in _FIRST_POOL if n not in blocked})
    last = sorted({n for n in _LAST_POOL if n not in blocked})
    return first, last


# ------------------------------------------------------------------------------------------------ frames
TITLES = ["המטופל", "המטופלת", "הנבדק", "הנבדקת", "החולה"]


def first_frames(F, L, rng):
    """Frames where F is a first name and L a last name (L may be a plain pool name). Prefix letters stay outside."""
    t = rng.choice(TITLES)
    return [
        lambda: [t, " ", E("FIRST_NAME", F), " ", E("LAST_NAME", L), " הופנה לבדיקת המשך בעוד שבועיים."],
        lambda: ["שם פרטי: ", E("FIRST_NAME", F), ", שם משפחה: ", E("LAST_NAME", L), "."],
        lambda: ["הרופא שוחח עם ", E("FIRST_NAME", F), " ", E("LAST_NAME", L), " ועם בני משפחתו."],
        lambda: ["לבקשת ", E("FIRST_NAME", F), " ", E("LAST_NAME", L), " נשלח מכתב סיכום."],
        lambda: [E("FIRST_NAME", F), " ", E("LAST_NAME", L), " מדווח על שיפור ניכר מאז הביקור הקודם."],
        lambda: ["המכתב נמסר ל", E("FIRST_NAME", F), " ", E("LAST_NAME", L), " בתום הביקור."],
        lambda: ["ביקור בית אצל ", E("FIRST_NAME", F), " ", E("LAST_NAME", L), " עבר ללא אירועים."],
        lambda: ["אחות אחראית: ", E("FIRST_NAME", F), " ", E("LAST_NAME", L)],
        lambda: ["בתו, ", E("FIRST_NAME", F), ", ליוותה אותו לבדיקה."],
        lambda: ["לאחר שיחה עם ", E("FIRST_NAME", F), " הוחלט להמשיך בטיפול הנוכחי."],
        lambda: ["תודה מראש, ", E("FIRST_NAME", F), " ", E("LAST_NAME", L)],
        lambda: ["הוזמן ל", E("FIRST_NAME", F), " תור למעקב בעוד חודש."],
    ]


def last_frames(F, L, rng):
    t = rng.choice(TITLES)
    return [
        lambda: [E("FIRST_NAME", F), " ", E("LAST_NAME", L), " נבדק בבוקר ושוחרר הביתה."],
        lambda: ["משפחת ", E("LAST_NAME", L), " עודכנה בתוצאות הבדיקה."],
        lambda: ["שם משפחה: ", E("LAST_NAME", L), "."],
        lambda: ["הוזמן ל", E("FIRST_NAME", F), " ", E("LAST_NAME", L), " תור למעקב."],
        lambda: ["בן הזוג, מר ", E("LAST_NAME", L), ", ליווה אותה לבדיקה."],
        lambda: ["ד\"ר ", E("LAST_NAME", L), " ייעץ בנושא התאמת הטיפול."],
        lambda: [t, " ", E("FIRST_NAME", F), " ", E("LAST_NAME", L), " סובל מכאבי גב כרוניים."],
        lambda: ["נחתם: ", E("FIRST_NAME", F), " ", E("LAST_NAME", L), ", רופא מטפל."],
        lambda: ["אל: גב' ", E("LAST_NAME", L), ", ", "בנושא תוצאות הבדיקה."],
        lambda: ["לכבוד ", E("FIRST_NAME", F), " ", E("LAST_NAME", L), ", מצורף סיכום הביקור."],
    ]


# plain-use frames by category: the word is NOT a name here. {w} sits after a prefix or a head word.
PLAIN_BY_CATEGORY = {
    "disease_eponym": ["אובחנה אצל המטופל מחלת {w} לפני שנתיים.", "בבירור נשללה תסמונת {w}.",
                       "ברקע הרפואי מחלת {w} ללא החמרה."],
    "season": ["ההחמרה מתרחשת בעיקר ב{w}.", "ב{w} חלה עלייה בתלונות."],
}
PLAIN_GENERIC = ["הרופא התייחס ל{w} בהסבר שנתן למטופל.", "בסיכום נכתב על ה{w} ללא קשר לזהות המטופל.",
                 "נושא ה{w} עלה בשיחה עם המשפחה."]
NO_PLAIN_CATEGORIES = {"month", "place", "hospital", "hmo", "company"}  # no ordinary sense that a generic frame can state


def plain_sentences(row):
    cat = row["category"]
    if cat in NO_PLAIN_CATEGORIES:
        return []
    return PLAIN_BY_CATEGORY.get(cat, PLAIN_GENERIC)


def name_sentences(rng, row, pools):
    """[(parts, kind-tag)] for the ways the lexicon says this word can be PHI (several per label, different frames)."""
    first, last = pools
    w, head, out = row["word"], row["phi_head"], []
    for lab in row["can_be_phi_as"].split("|"):
        if lab == "FIRST_NAME":
            fr = first_frames(w, rng.choice(last), rng)
            for k in rng.sample(range(len(fr)), 2):
                out.append((fr[k](), "first-name"))
        elif lab == "LAST_NAME":
            fr = last_frames(rng.choice(first), w, rng)
            for k in rng.sample(range(len(fr)), 2):
                out.append((fr[k](), "last-name"))
        elif lab == "DATE":
            year = rng.randint(1995, 2024)
            out.append((["התלונות החלו ב", E("DATE", w), "."], "month-alone"))
            out.append((["הביקור האחרון היה ב", E("DATE", f"{w} {year}"), " ללא ממצא חדש."], "month-year"))
        elif lab == "LOC":
            out.append((["לדברי המטופל הוא גר ב", E("LOC", w), " כבר שנים."], "loc"))
            out.append((["הגיעה מ", E("LOC", w), " לבדיקה."], "loc"))
        elif lab in ("MED_ORG", "ORG"):
            hd = head + " " if head else ""
            frames = [["הופנה ל", hd, E(lab, w), " לבירור."], ["המטופל שוחרר מ", hd, E(lab, w), " אתמול."],
                      ["לדבריו עבר טיפול ב", hd, E(lab, w), "."]]
            for fr in rng.sample(frames, 2):
                out.append((fr, lab.lower()))
    return out


def ambiguous_docs(rng, pools, all_words=False):
    """Training documents for the SEEN words (or for ALL words, production): name use, plain use, and both."""
    docs = []
    for row in load_lexicon():
        w = row["word"]
        if not all_words and not word_is_seen(w):
            continue
        names = name_sentences(rng, row, pools)
        plains = plain_sentences(row)
        for parts, tag in names:
            docs.append((parts, {"word": w, "kind": "name", "tag": tag}))
        if plains:
            for s in rng.sample(plains, min(2, len(plains))):
                docs.append(([s.format(w=w)], {"word": w, "kind": "plain", "tag": "plain-use"}))
            person = [(p, t) for p, t in names if t in ("first-name", "last-name")]
            for parts, tag in person[:2]:
                plain = rng.choice(plains).format(w=w)
                both = [parts, " ", plain] if rng.random() < 0.5 else [plain, " ", parts]
                docs.append((both, {"word": w, "kind": "mixed", "tag": tag}))
    return docs


def generic_name_docs(rng, pools, n):
    first, last = pools
    docs = []
    for _ in range(n):
        f, l = rng.choice(first), rng.choice(last)
        kind = rng.random()
        if kind < 0.7:
            fr = first_frames(f, l, rng) if rng.random() < 0.5 else last_frames(f, l, rng)
            parts = rng.choice(fr)()
        elif kind < 0.85:  # two people in one sentence (patient and a relative or a doctor)
            f2, l2 = rng.choice(first), rng.choice(last)
            parts = ["המטופל ", E("FIRST_NAME", f), " ", E("LAST_NAME", l), " הגיע עם ", rng.choice(["אחיו", "אמו", "בנו"]), " ",
                     E("FIRST_NAME", f2), " ", E("LAST_NAME", l2), "."]
        else:  # a name inside a short list
            f2, l2 = rng.choice(first), rng.choice(last)
            parts = ["צוות מטפל: ד\"ר ", E("FIRST_NAME", f), " ", E("LAST_NAME", l), ", האחות ", E("FIRST_NAME", f2), "."]
        docs.append((parts, {"word": "", "kind": "name-generic", "tag": "name"}))
    return docs


def org_docs(rng, n):
    docs = []
    for _ in range(n):
        if rng.random() < 0.6:
            head, name = rng.choice(MED_ORGS)
            lab = "MED_ORG"
            frame = rng.choice([["עבר ניתוח ב", head + " ", E(lab, name), " בחודש שעבר."], ["הועבר מ", head + " ", E(lab, name),
                                " לצורך בירור."], ["מעקב בקופ\"ח: ", E(lab, name), "."], ["סיכום שחרור מ", head + " ", E(lab, name)]])
        else:
            head, name = rng.choice(ORGS)
            lab = "ORG"
            frame = rng.choice([["המטופל מועסק ב", head + " ", E(lab, name), "."], ["אישור מעסיק מ", head + " ", E(lab, name)],
                                ["עבד שנים רבות ב", head + " ", E(lab, name), "."]])
        # about 1 in 3 organisation names appears glued to the prefix, cut inside the word like in the real data
        docs.append((frame, {"word": "", "kind": "org", "tag": lab.lower()}))
    return docs


def date_docs(rng, n):
    tags = [t for t in DATE_FORMATS if t != "date-no-punctuation"]
    docs = []
    for _ in range(n):
        tag = rng.choice(tags)
        d = DATE_FORMATS[tag](*rdate(rng))
        frame = rng.choice([["הניתוח בוצע ב", E("DATE", d), " ללא סיבוכים."], ["תאריך הקבלה: ", E("DATE", d)],
                            ["נקבע מעקב ליום ", E("DATE", d), "."], ["בדיקת הדם האחרונה מ-", E("DATE", d), " תקינה."]])
        docs.append((frame, {"word": "", "kind": "date", "tag": tag}))
    return docs


def id_docs(rng, n):
    docs = []
    for _ in range(n):
        v = israeli_id(rng)
        frame = rng.choice([["מספר זהות של המטופל: ", E("ID", v)], ["נבדק לפי ת\"ז ", E("ID", v), " בקבלה."],
                            ["זיהוי: ", E("ID", v), ", אושר מול המרשם."]])
        docs.append((frame, {"word": "", "kind": "id", "tag": "id"}))
    return docs


def empty_docs(rng, n):
    return [([rng.choice(FILL)], {"word": "", "kind": "empty", "tag": "none"}) for _ in range(n)]


def render(n=DEFAULT_N, all_words=False):
    rng = random.Random(SEED)
    pools = name_pools()
    docs = ambiguous_docs(rng, pools, all_words)
    rest = max(0, n - len(docs))
    shares = {"name": 0.62, "org": 0.14, "date": 0.12, "id": 0.07, "empty": 0.05}
    docs += generic_name_docs(rng, pools, int(rest * shares["name"]))
    docs += org_docs(rng, int(rest * shares["org"]))
    docs += date_docs(rng, int(rest * shares["date"]))
    docs += id_docs(rng, int(rest * shares["id"]))
    docs += empty_docs(rng, n - len(docs) if n > len(docs) else 0)
    rng.shuffle(docs)
    out = []
    for parts, meta in docs:
        if rng.random() < 0.35:  # a neutral sentence before or after, so the entity is not always at the document edge
            parts = [rng.choice(FILL), " ", parts] if rng.random() < 0.5 else [parts, " ", rng.choice(FILL)]
        text, spans, _keeps = build(parts)
        spans = split_date_spans(text, spans)
        for s in spans:
            assert 0 <= s["start"] < s["end"] <= len(text)
        out.append((text, spans, meta))
    return out


def serialize(docs):
    lines, meta = [], {}
    for i, (text, spans, m) in enumerate(docs, 1):
        doc_id = f"synthtrain_{i:05d}"
        lines.append(json.dumps({"id": doc_id, "text": text,
                                 "label": [{"start": s["start"], "end": s["end"], "label": s["label"]} for s in spans]},
                                ensure_ascii=False))
        meta[doc_id] = m
    return ("\n".join(lines) + "\n").encode("utf-8"), (json.dumps(meta, ensure_ascii=False, indent=1) + "\n").encode("utf-8")


def main(out=OUT, n=DEFAULT_N, all_words=False):
    """all_words=True: PRODUCTION variant (synth_train_all.*) with every ambiguous word; no word is held out."""
    out = Path(out)
    out.mkdir(exist_ok=True)
    docs = render(n, all_words)
    data, meta = serialize(docs)
    stem = "synth_train_all" if all_words else "synth_train"
    (out / f"{stem}.jsonl").write_bytes(data)
    (out / f"{stem}.meta.json").write_bytes(meta)
    split = word_split()
    (out / "word_split.json").write_bytes((json.dumps(split, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))
    kinds, labels = {}, {}
    for _t, spans, m in docs:
        kinds[m["kind"]] = kinds.get(m["kind"], 0) + 1
        for s in spans:
            labels[s["label"]] = labels.get(s["label"], 0) + 1
    print(f"{len(docs)} documents: " + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items())))
    print("entities: " + ", ".join(f"{k} {v}" for k, v in sorted(labels.items())))
    print(f"ambiguous words: {len(split['seen'])} seen, {len(split['heldout'])} held out")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=DEFAULT_N)
    ap.add_argument("--all-words", action="store_true",
                    help="production variant: documents for every ambiguous word (use only after the recipe is settled)")
    a = ap.parse_args()
    main(n=a.n, all_words=a.all_words)
