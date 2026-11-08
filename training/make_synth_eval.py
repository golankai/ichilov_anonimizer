"""Generate the SYNTHETIC EVALUATION SETS for the base model: training/synth_eval/{rare_hard,long}.jsonl

Purpose: the real validation/test splits hold almost no examples of the rare labels (ID, INNER_ID have none at all,
ORG and AGE_ABOVE_89 a dozen) and no way to tell how the model copes with hard cases or long documents. These sets
are scored by ``train.py`` after every run, next to the real test split, and are NEVER used to pick an epoch.

  rare_hard.jsonl  short documents (one window), about 40+ examples of every rare label, name/word collisions,
                   prefixes, and look-alike numbers that must NOT be masked (lab values, doses, blood pressure).
  long.jsonl       documents of 450 to 2500 words (several 512-token windows), entities spread over the whole text
                   so many fall near window edges.

This is the evaluation of the BASE MODEL (token classification as trained). The evaluation of the full pipeline
(chunking, rules, post-processing on raw text) is separate: ``inference/dummy_data`` + ``inference/score_dummy.py``.
The two sets are generated with different seeds and different templates; they share only the invented name lists.

Everything is invented. Offsets are computed while the text is assembled, so the labels are exact. Conventions follow
``inference/dummy_data/README.md`` (Hebrew prefix outside the entity, titles outside, first/last name separate spans,
medical organisation includes its head noun). Situations whose convention we only guessed (month alone as a date,
street addresses, passport numbers, letter-prefixed case numbers, countries) are left out on purpose. INNER_ID is
the exception we cannot avoid: the real data has only 3 examples, so what an INNER_ID looks like here is an
assumption (6-8 digit case numbers after a label such as "מספר תיק"). Read that row accordingly.

Output is deterministic (fixed seeds); ``training/tests/test_synth_eval.py`` checks that re-generating reproduces
the committed files.

Run:  python training/make_synth_eval.py
"""
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "inference"))
from make_dummy_data import (  # noqa: E402  (invented-data building blocks shared with the pipeline fake set)
    CITIES, DATE_FORMATS, FILLER, FIRST, HEB_CAL_DATES, HYPHEN_LAST, LAST, E, K, build, digits, israeli_id,
    landline, mobile, name, rdate,
)

OUT = HERE / "synth_eval"

# ------------------------------------------------------------------------------------------------ vocabulary
TOWNS = CITIES + ["קריית גת", "טבריה", "צפת", "דימונה", "עכו", "לוד", "רמלה", "נס ציונה", "יבנה", "אור יהודה",
                  "קריית אתא", "נתיבות", "ערד", "מעלות", "גבעתיים", "רעננה", "הוד השרון"]
ORG_LIST = [("בנק", "הפועלים"), ("בנק", "דיסקונט"), ("עיריית", "נתניה"), ("עיריית", "ראשון לציון"), ("משרד", "החינוך"),
            ("משרד", "הבריאות"), ("חברת", "תנובה"), ("חברת", "החשמל"), ("רשת", "שופרסל"), ("חברת", "אל על"),
            ("אוניברסיטת", "תל אביב"), ("אוניברסיטת", "בן גוריון"), ("אוניברסיטת", "חיפה"), ("חברת", "טבע"), ("חברת", "פלאפון"),
            ("מפעלי", "ים המלח"), ("חברת", "סלקום"), ("בית הספר", "הריאלי"), ("חברת", "אלביט"), ("חברת", "בזק")]
# REAL CONVENTION (training/check_label_layout.py): only the bare name is labeled; the generic head word (בנק, חברת,
# בית החולים, ...) is plain text, and a Hebrew prefix letter stays outside the span. Each entry is (head, name).
MED_ORG_LIST = [("בית החולים", "סורוקה"), ("בית החולים", "הלל יפה"), ("מרכז רפואי", "רבין"), ("בית החולים", "פוריה"),
                ("בית החולים", "אסף הרופא"), ("בית החולים", "כרמל"), ("מרכז רפואי", "וולפסון"), ("בית החולים", "מאיר"),
                ("בית החולים", "העמק"), ("בית החולים", "זיו"), ("מרפאת", "כללית"), ("מרפאת", "מכבי"),
                ("בית החולים", "לוינשטיין"), ("מרכז רפואי", "ברזילי"), ("בית החולים", "הדסה הר הצופים"),
                ("בית החולים", "איכילוב"), ("בית החולים", "שיבא"), ("ביה\"ח", "רמב\"ם"), ("קופ\"ח", "מכבי"),
                ("קופת חולים", "לאומית")]
GENERIC_UNITS = ["מחלקת ילדים", "היחידה לטיפול נמרץ", "המכון לקרדיולוגיה", "חדר ניתוח", "מרפאת נשים", "מחלקה פנימית א'",
                 "הרופא התורן", "יחידת הדיאליזה"]
DRUGS = ["אקמול", "אספירין", "קומדין", "מטפורמין", "אומפרדקס", "לוסק", "ויטמין D"]
EPONYMS = ["מחלת פרקינסון", "תסמונת דאון", "מחלת קרוהן", "מחלת אלצהיימר", "מחלת הודג'קין", "תסמונת קושינג"]
NAME_WORDS = [  # (first name, ordinary phrase where the same word is NOT a name)
    ("לב", "אי ספיקת לב"), ("אושר", "הטיפול אושר"), ("גל", "גל חום"), ("אור", "רגישות לאור")]
PLAIN_NAMES = ["שחר", "ניסן", "אדר", "טל", "ים", "נוי"]
EXTRA_FILLER = [
    "ללא שינוי משמעותי בהשוואה לביקור הקודם.",
    "הומלץ על שתיית מים בכמות מספקת ועל מנוחה.",
    "הבדיקה הגופנית הייתה תקינה למעט ממצאים קלים שפורטו לעיל.",
    "המשפחה עודכנה בתוכנית הטיפול והביעה הסכמה.",
    "תוצאות הבדיקות יישלחו לרופא המטפל עם סיום הבירור.",
    "נקבע מעקב לאחר חודש עם תוצאות בדיקות דם חוזרות.",
]
FILL = FILLER + EXTRA_FILLER


def P(label, value):
    return E(label, value)


def org_parts(label, pair):
    """(head, name) -> parts: the head word is plain text, only the name is the entity."""
    head, name = pair
    return [head + " " if head else "", P(label, name)]


# ------------------------------------------------------------------------------------------------ snippets
# Every snippet is fn(rng) -> list of parts for build(). One function per situation, so coverage is by design.
def date_str(rng, tag):
    return DATE_FORMATS[tag](*rdate(rng))


def snip_id(rng):
    n = israeli_id(rng)
    dashed = f"{n[:8]}-{n[8]}"
    eight = israeli_id(rng, True)
    f, l = name(rng)
    return [
        lambda: ["ת.ז. ", P("ID", n), " אומתה בקבלה."],
        lambda: ["תעודת זהות מספר ", P("ID", n), " הוצגה בקבלה."],
        lambda: ["מס' זהות: ", P("ID", n)],
        lambda: ["ת\"ז ", P("ID", eight), " (ללא אפס מוביל)."],
        lambda: ["ת.ז. ", P("ID", dashed), " ושם המטופל נבדקו מול המרשם."],
        lambda: ["המטופל נרשם במערכת תחת ", P("ID", n), " לצורך קביעת תור."],
        lambda: ["שם: ", P("FIRST_NAME", f), " ", P("LAST_NAME", l), " | ת.ז.: ", P("ID", n)],
        lambda: ["ID: ", P("ID", n), ", יש לאמת מול המבוטח."],
        lambda: ["נרשם בעבר תחת ת.ז. ", P("ID", n), " בקופת החולים."],
        lambda: ["בן הזוג, ת.ז. ", P("ID", israeli_id(rng)), ", ליווה את המטופל."],
    ]


def snip_inner(rng):
    mk = lambda: digits(rng, rng.choice([6, 7, 8]))
    return [
        lambda: ["מספר תיק ", P("INNER_ID", mk()), " במערכת בית החולים."],
        lambda: ["מס' מקרה ", P("INNER_ID", mk()), " נפתח בקבלה."],
        lambda: ["תיק רפואי ", P("INNER_ID", mk()), "."],
        lambda: ["אירוע מספר ", P("INNER_ID", mk()), " תועד במחלקה."],
        lambda: ["מספר מטופל פנימי ", P("INNER_ID", mk()), " מופיע בכל המסמכים."],
        lambda: ["מס' אשפוז ", P("INNER_ID", mk()), ", מחלקה פנימית."],
        lambda: ["תיק ", P("INNER_ID", mk()), " הועבר לארכיון."],
        lambda: ["פתיחת תיק: ", P("INNER_ID", mk()), "."],
    ]


def snip_age(rng):
    a = lambda: rng.choice([90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 104])
    return [
        lambda: ["מטופל בן ", P("AGE_ABOVE_89", str(a())), " הגיע עם בתו."],
        lambda: ["מטופלת בת ", P("AGE_ABOVE_89", str(a())), " עצמאית בפעולות היומיום."],
        lambda: ["גיל ", P("AGE_ABOVE_89", str(a())), ", מתגורר בדיור מוגן."],
        lambda: ["חולה בגיל ", P("AGE_ABOVE_89", str(a())), " שנים עם יתר לחץ דם."],
        lambda: ["הגברת, בת ", P("AGE_ABOVE_89", str(a())), ", נבדקה היום."],
        lambda: ["(גיל ", P("AGE_ABOVE_89", str(a())), ") אושפז בעקבות נפילה."],
        # must stay readable: ages up to 89, children, and numbers of 90 or more that are NOT ages
        lambda: ["מטופל בן ", K(str(rng.choice([85, 87, 88, 89])), "age up to 89"), " הגיע לביקורת."],
        lambda: ["הילדה בת ", K(str(rng.randint(2, 14)), "child age"), " הגיעה עם אמה."],
        lambda: ["רוויה ", K(str(rng.randint(90, 99)), "oxygen saturation"), "% באוויר חדר."],
        lambda: ["לחץ דם ", K(f"{rng.randint(100, 150)}/{rng.randint(60, 95)}", "blood pressure"), ", דופק ",
                 K(str(rng.randint(90, 110)), "pulse"), "."],
        lambda: ["משקל ", K(str(rng.randint(90, 120)), "weight"), " ק\"ג, גובה ", K("172", "height"), " ס\"מ."],
        lambda: ["רמת סוכר בצום ", K(str(rng.randint(92, 140)), "lab value"), " מ\"ג/ד\"ל."],
    ]


def snip_org(rng):
    o = lambda: rng.choice(ORG_LIST)
    return [
        lambda: ["הוא עובד ב", org_parts("ORG", o()), " כבר שנים."],
        lambda: ["מועסקת ב", org_parts("ORG", o()), " כמזכירה."],
        lambda: ["התקבל מכתב מ", org_parts("ORG", o()), " בנושא ביטוח."],
        lambda: ["המטופל פרש מ", org_parts("ORG", o()), " לפני שנתיים."],
        lambda: ["לדברי המטופל, עבד ב", org_parts("ORG", o()), " ובמקום נוסף."],
        lambda: ["אישור מחלה הונפק ל", org_parts("ORG", o()), "."],
        lambda: ["לומד ב", org_parts("ORG", o()), " ומתגורר במעונות."],
        lambda: ["סטודנט ב", org_parts("ORG", o()), ", שנה ג'."],
    ]


def snip_medorg(rng):
    m = lambda: rng.choice(MED_ORG_LIST)
    return [
        lambda: ["הועבר ל", org_parts("MED_ORG", m()), " להמשך בירור."],
        lambda: ["שוחרר מ", org_parts("MED_ORG", m()), " לביתו."],
        lambda: ["מעקב ב", org_parts("MED_ORG", m()), " בעוד חודש."],
        lambda: ["נבדק בעבר ב", org_parts("MED_ORG", m()), " ללא ממצא."],
        lambda: ["הופנה מ", org_parts("MED_ORG", m()), " עם מכתב רופא."],
        lambda: ["אושפז ב", org_parts("MED_ORG", m()), " ואחר כך ב", org_parts("MED_ORG", m()), "."],
        lambda: ["הגיע לחדר מיון ב", org_parts("MED_ORG", m()), " באמבולנס."],
        lambda: ["המטופל הועבר ל", K(rng.choice(GENERIC_UNITS), "generic unit, not a named organisation"),
                 " לצורך המשך טיפול."],
    ]


def snip_contact(rng):
    return [
        lambda: ["ניתן ליצור קשר בנייד ", P("CONTACT", mobile(rng, 0)), "."],
        lambda: ["טלפון: ", P("CONTACT", mobile(rng, 1))],
        lambda: ["מספר הנייד ", P("CONTACT", mobile(rng, 2)), " פעיל גם בווטסאפ."],
        lambda: ["להשיג בטלפון ", P("CONTACT", mobile(rng, 3)), " בשעות הבוקר."],
        lambda: ["טל' בבית: ", P("CONTACT", landline(rng)), "."],
        lambda: ["פקס המרפאה ", P("CONTACT", landline(rng)), "."],
        lambda: ["דוא\"ל: ", P("CONTACT", f"{rng.choice(['yael', 'dr.levi', 'moshe_k', 'info'])}"
                                       f"{rng.choice(['', '.', '_'])}{rng.randint(10, 99)}@"
                                       f"{rng.choice(['gmail.com', 'walla.co.il', 'demo-med.org.il', 'outlook.com'])}")],
        lambda: ["לשאלות ניתן לכתוב ל-", P("CONTACT", f"clinic.{rng.choice('abcdef')}@demo-med.org.il"), " בלבד."],
        lambda: ["אימייל: ", P("CONTACT", f"a.b.{rng.randint(1, 9)}@demo-med.org.il"), ", נייד: ",
                 P("CONTACT", mobile(rng, 0))],
        # numbers that look like contacts but are not
        lambda: ["הוזמן לבדיקה בשעה ", K("08:30", "time"), " ובקוד ", K(digits(rng, 4), "short code"), "."],
        lambda: ["מינון ", K("10000", "dose"), " יחידות, ספירת טסיות ", K("250000", "lab value"), " למיקרוליטר."],
    ]


def snip_loc(rng):
    c = lambda: rng.choice(TOWNS)
    return [
        lambda: ["המטופל מתגורר ב", P("LOC", c()), " עם משפחתו."],
        lambda: ["הגיע מ", P("LOC", c()), " באמבולנס."],
        lambda: ["הופנה ל", P("LOC", c()), " לבדיקה."],
        lambda: ["תושב ", P("LOC", c()), " הגיע לביקורת."],
        lambda: ["אשתו גרה בעיר ", P("LOC", c()), "."],
        lambda: ["עבר לאחרונה מ", P("LOC", c()), " ל", P("LOC", c()), "."],
        lambda: ["עיר מגורים: ", P("LOC", c())],
        lambda: ["חזר ל", P("LOC", c()), " לאחר השחרור."],
    ]


def snip_name(rng):
    f, l = name(rng)
    f2, l2 = name(rng)
    hl = rng.choice(HYPHEN_LAST)
    nw, nw_phrase = rng.choice(NAME_WORDS)
    out = [
        lambda: ["המטופל ", P("FIRST_NAME", f), " ", P("LAST_NAME", l), " הגיע לביקורת."],
        lambda: ["הגב' ", P("FIRST_NAME", f), " ", P("LAST_NAME", l), " מתלוננת על כאבים."],
        lambda: ["הומלץ ל", P("FIRST_NAME", f), " ", P("LAST_NAME", l), " להמשיך בטיפול."],
        lambda: ["הוסבר ל", P("FIRST_NAME", f), " מה צפוי בהמשך."],
        lambda: ["מר ", P("LAST_NAME", l), " הגיע עם אשתו."],
        lambda: ["ד\"ר ", P("FIRST_NAME", f), " ", P("LAST_NAME", l), " ביצע את ההערכה."],
        lambda: ["האחות ", P("FIRST_NAME", f), " נתנה את התרופה."],
        lambda: ["המטופלת ", P("FIRST_NAME", f), " ", P("LAST_NAME", hl), " הגיעה לביקורת."],
        lambda: ["שם משפחה: ", P("LAST_NAME", l), ", שם פרטי: ", P("FIRST_NAME", f)],
        lambda: ["בנה ", P("FIRST_NAME", f2), " ", P("LAST_NAME", l), " מלווה אותה."],
        lambda: ["מכתב ממנהל המחלקה פרופ' ", P("FIRST_NAME", f2), " ", P("LAST_NAME", l2), "."],
        lambda: ["המטופל ", P("FIRST_NAME", nw), " ", P("LAST_NAME", l), " מוכר במרפאה."],
        lambda: ["ל", P("FIRST_NAME", rng.choice(PLAIN_NAMES)), " ", P("LAST_NAME", l), " יש רגישות לפניצילין."],
        lambda: ["המטופל ", P("FIRST_NAME", nw), " ", P("LAST_NAME", l2), " מדווח על ", K(nw_phrase, "ordinary phrase"),
                 "."],
        lambda: ["מטופל עם ", K(rng.choice(EPONYMS), "disease named after a person"), ", ", P("FIRST_NAME", f), " ",
                 P("LAST_NAME", l), ", נמצא יציב."],
        lambda: ["קיבל ", K(rng.choice(DRUGS), "drug"), " לפי הוראת ד\"ר ", P("LAST_NAME", l2), "."],
    ]
    return out


def snip_date(rng):
    # 8 digits without separators ("12032020") never occurs as one span in the real data (longest date piece is 7 characters)
    fmts = [t for t in DATE_FORMATS if t != "date-no-punctuation"]
    d = lambda tag: P("DATE", date_str(rng, tag))
    out = [(lambda tag=tag: ["המטופל אושפז בתאריך ", d(tag), " ושוחרר לאחר מכן."]) for tag in fmts]
    out += [
        lambda: ["הבדיקה בוצעה ב-", d("date-numeric-dot"), " במרפאה."],
        lambda: ["הטיפול החל מ-", d("date-numeric-slash"), "."],
        lambda: ["הופנה ל-", d("date-numeric-dot-nopad"), " לצילום."],
        lambda: ["המטופל נולד ב", P("DATE", rng.choice(HEB_CAL_DATES)), "."],
        lambda: ["בין ", d("date-numeric-dot"), " ל-", d("date-numeric-dot"), " אושפז במחלקה."],
        lambda: ["תאריך לידה: ", d("date-numeric-slash")],
        lambda: ["ביקור ביום ", d("date-numeric-dot"), " בשעה ", K("09:15", "time"), "."],
        lambda: ["הציון ", K("12/15", "score, not a date"), " בסולם ההערכה."],
        lambda: ["לחץ דם ", K("130/80", "blood pressure, not a date"), " בביקור."],
        lambda: ["קוד אבחנה ", K(rng.choice(["I10", "E11.9", "J18.9"]), "ICD code"), " נרשם."],
    ]
    return out


def snip_header(rng):
    """A form-like block with many entities close together (the dense, hard case)."""
    f, l = name(rng)
    fields = [
        ["שם: ", P("FIRST_NAME", f), " ", P("LAST_NAME", l)],
        ["ת.ז.: ", P("ID", israeli_id(rng))],
        ["תאריך לידה: ", P("DATE", date_str(rng, "date-numeric-slash"))],
        ["טלפון: ", P("CONTACT", mobile(rng, rng.randint(0, 3)))],
        ["עיר: ", P("LOC", rng.choice(TOWNS))],
        ["קופת חולים: ", P("MED_ORG", rng.choice(MED_ORG_LIST)[1])],
        ["מספר תיק: ", P("INNER_ID", digits(rng, 7))],
        ["מקום עבודה: ", P("ORG", rng.choice(ORG_LIST)[1])],
    ]
    keep = rng.sample(fields, rng.randint(4, len(fields)))
    parts = []
    for i, fld in enumerate(keep):
        parts += fld + [rng.choice(["\n", " | ", ", "])]
    return parts[:-1] + ["\n"]


GROUPS = {  # group -> (snippet-list factory, documents in rare_hard)
    "id": (snip_id, 50), "inner_id": (snip_inner, 40), "age": (snip_age, 84), "org": (snip_org, 40),
    "med_org": (snip_medorg, 40), "contact": (snip_contact, 66), "loc": (snip_loc, 40), "name": (snip_name, 96),
    "date": (snip_date, 90),
}
HEADER_DOCS = 40
WEIGHTS = {"id": 2, "inner_id": 2, "age": 2, "org": 2, "med_org": 2, "contact": 3, "loc": 2, "name": 3, "date": 3}


def wrap(rng, snippet_parts):
    lead = [rng.choice(FILL), " "] if rng.random() < 0.4 else []
    tail = [" ", rng.choice(FILL)] if rng.random() < 0.4 else []
    return build(lead, snippet_parts, tail)


# ------------------------------------------------------------------------------------------------ the two sets
def rare_hard(rng):
    docs = []
    for group, (factory, n) in GROUPS.items():
        for i in range(n):
            snippets = factory(rng)  # fresh random values each time, same situation list
            docs.append((group, wrap(rng, snippets[i % len(snippets)]())))
    for _ in range(HEADER_DOCS):
        docs.append(("header", build(snip_header(rng))))
    rng.shuffle(docs)
    return [d for _, d in docs]


def long_doc(rng, target_words):
    """Filler paragraphs with entities of every kind spread through the whole text, header at the start."""
    f, l = name(rng)
    parts = ["סיכום רפואי מפורט\n"] + snip_header(rng) + ["\n"]
    order = [g for g, w in WEIGHTS.items() for _ in range(w)]
    rng.shuffle(order)
    k = 0
    while len(build(parts)[0].split()) < target_words:
        parts.append(" ".join(rng.sample(FILL, rng.randint(2, 5))) + " ")
        for _ in range(rng.randint(1, 2)):
            group = order[k % len(order)]
            k += 1
            factory = GROUPS[group][0]
            snippets = factory(rng)
            parts += [snippets[rng.randrange(len(snippets))](), " "]
        if rng.random() < 0.25:  # the patient's name comes back in the running text
            parts += ["לדברי ", P("FIRST_NAME", f), " ", P("LAST_NAME", l), " אין שינוי. "]
        parts.append("\n")
    parts += ["\nסוף המסמך: ", P("FIRST_NAME", f), " ", P("LAST_NAME", l), ", ", P("CONTACT", mobile(rng)), ", ",
              P("DATE", date_str(rng, "date-numeric-dot")), "."]
    return build(parts)


LONG_WORDS = [450] * 6 + [800] * 6 + [1500] * 6 + [2500] * 6


def long_set(rng):
    return [long_doc(rng, w) for w in LONG_WORDS]


SETS = {"rare_hard": (rare_hard, 2001), "long": (long_set, 2002)}


def check(text, spans):
    taken = sorted((s["start"], s["end"]) for s in spans)
    for (a, b), (c, d) in zip(taken, taken[1:]):
        assert b <= c, "overlapping spans"
    assert all(0 <= a < b <= len(text) for a, b in taken)


_DATE_SEP = re.compile(r"[\s./\-]+")


def split_date_spans(text, spans):
    """Real annotation: a date is DAY, MONTH and YEAR spans (renamed DATE on load), the separators are unlabeled.
    Measured on the real train split: 2097 date spans, median length 2, none longer than 7, half directly adjacent."""
    out = []
    for sp in spans:
        if sp["label"] != "DATE":
            out.append(sp)
            continue
        pos = sp["start"]
        for piece in _DATE_SEP.split(text[sp["start"]:sp["end"]]):
            if not piece:
                continue
            a = text.index(piece, pos)
            out.append({**sp, "start": a, "end": a + len(piece)})
            pos = a + len(piece)
    return sorted(out, key=lambda x: x["start"])


def render(name_, seed):
    """JSONL text of one set (id, text, label with start/end/label only: the training format)."""
    docs = SETS[name_][0](random.Random(seed))
    lines = []
    for i, (text, spans, _keeps) in enumerate(docs, 1):
        spans = split_date_spans(text, spans)
        check(text, spans)
        label = [{"start": s["start"], "end": s["end"], "label": s["label"]} for s in spans]
        lines.append(json.dumps({"id": f"synth_{name_}_{i:04d}", "text": text, "label": label}, ensure_ascii=False))
    return ("\n".join(lines) + "\n").encode("utf-8"), docs


def main(out=OUT):
    out = Path(out)
    out.mkdir(exist_ok=True)
    for name_, (_fn, seed) in SETS.items():
        data, docs = render(name_, seed)
        (out / f"{name_}.jsonl").write_bytes(data)  # utf-8, LF: identical on every machine
        per_label = {}
        for _t, spans, _k in docs:
            for s in split_date_spans(_t, spans):
                per_label[s["label"]] = per_label.get(s["label"], 0) + 1
        print(f"{name_}: {len(docs)} documents, {sum(per_label.values())} entities: "
              + ", ".join(f"{k} {v}" for k, v in sorted(per_label.items())))


if __name__ == "__main__":
    main()
