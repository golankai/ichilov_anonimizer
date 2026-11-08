"""Generate the fake test set WITH ground truth: inference/dummy_data/<tier>.csv + <tier>.truth.jsonl.

Everything here is invented (names, numbers, phones, emails); nothing comes from real patients. Offsets are computed
while the text is assembled, so the ground truth is exact by construction. The output is deterministic (fixed seeds):
re-running must reproduce the committed files byte for byte (checked by inference/tests/test_dummy_truth.py).

Truth file, one JSON object per line (same shape as the training data, plus extra keys that readers may ignore):
  {"id", "tier", "text",
   "label": [{"start", "end", "label", "tag", "policy"?}],   # must be masked (end exclusive)
   "keep":  [{"start", "end", "note"}]}                        # must stay readable (traps against over-masking)
`tag` names the situation (e.g. "date-hebrew-calendar"); `policy: true` marks conventions we assumed and that should
be checked against the real annotation guidelines. Anything not listed in `label` is non-PHI by construction.

Conventions (taken from training/dummy_data): a Hebrew prefix (ב ל מ ה ו ש) is NOT part of the entity; titles
(ד"ר, הגב', מר) are not part of it; first and last name are separate spans; a medical organisation includes its head
noun ("בית החולים איכילוב"); a date is one span including day, month and year.

Run:  python inference/make_dummy_data.py
"""
import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent / "dummy_data"
TIERS = ["t1_simple", "t2_formats", "t3_traps", "t4_notes", "t5_long"]


# ----------------------------------------------------------------------------------------------- building blocks
class E:
    """An entity that must be masked."""

    def __init__(self, label, value, tag="", policy=False):
        self.label, self.value, self.tag, self.policy = label, value, tag, policy


class K:
    """A span that must NOT be masked."""

    def __init__(self, value, note):
        self.value, self.note = value, note


def build(*parts):
    text, spans, keeps = "", [], []

    def add(p):
        nonlocal text
        if isinstance(p, (list, tuple)):
            for q in p:
                add(q)
        elif isinstance(p, str):
            text += p
        elif isinstance(p, E):
            s = {"start": len(text), "end": len(text) + len(p.value), "label": p.label, "tag": p.tag}
            if p.policy:
                s["policy"] = True
            spans.append(s)
            text += p.value
        elif isinstance(p, K):
            keeps.append({"start": len(text), "end": len(text) + len(p.value), "note": p.note})
            text += p.value
        else:
            raise TypeError(p)

    add(parts)
    return text, spans, keeps


FIRST = ["דוד", "שרה", "יוסף", "רחל", "משה", "מירב", "נועם", "תמר", "רועי", "אורי", "יעל", "אבי", "דנה", "עומר",
         "ליאור", "מיכל", "איתן", "הדס", "גיא", "נעמה", "אלון", "שירה", "בני", "טלי", "זהר", "אסף", "רונית", "חיים"]
LAST = ["כהן", "לוי", "מזרחי", "פרץ", "ביטון", "דהן", "אברהם", "שפירא", "גולן", "אוחיון", "חדד", "פרידמן", "שמעוני",
        "אזולאי", "מלכה", "רוזן", "קדוש", "סעדון", "נחום", "ברוך"]
HYPHEN_LAST = ["אלמוג-כץ", "בן-דוד", "בר-לב", "אבו-חצירה", "שטרן-לוי", "בן-עמי", "כהן-צדק"]
CITIES = ["תל אביב", "חיפה", "ירושלים", "באר שבע", "נתניה", "פתח תקווה", "רמת גן", "אשדוד", "בני ברק", "חולון",
          "כפר סבא", "נהריה", "אילת", "קריית שמונה", "רחובות", "מודיעין", "הרצליה", "עפולה"]
STREETS = ["הרצל", "בן יהודה", "ויצמן", "ז'בוטינסקי", "רוטשילד", "הנשיא", "הגפן", "הרימון", "דיזנגוף", "סוקולוב"]
ORGS = ["עיריית חולון", "בנק לאומי", "משרד הביטחון", "אוניברסיטת בן גוריון", "חברת הראל ביטוח", "חברת אלביט",
        "מכבי שירותי בריאות", "משטרת ישראל"]
MED_ORGS = ["בית החולים איכילוב", 'בית החולים רמב"ם', "מרכז רפואי סורוקה", "בית החולים הדסה עין כרם",
            "קופת חולים מכבי", "מרכז רפואי שיבא", "בית החולים שערי צדק", "מרפאת כללית בחולון"]
MONTHS_HE = ["ינואר", "פברואר", "מרץ", "אפריל", "מאי", "יוני", "יולי", "אוגוסט", "ספטמבר", "אוקטובר", "נובמבר",
             "דצמבר"]
HEB_CAL_DATES = ["י״א בתשרי תשפ״ד", "ג׳ באדר תשפ״ג", "כ״ה בניסן תשפ״ד", "ט״ו בשבט תשפ״ה", "כ״ט בסיון תשפ״ב"]
FILLER = [
    "בבדיקה גופנית המטופל בהכרה מלאה וללא סימני מצוקה נשימתית.",
    "סימנים חיוניים יציבים, הדופק סדיר ולחץ הדם תקין.",
    "בבדיקת הבטן לא נמצאה רגישות ולא סימני גירוי פריטוניאלי.",
    "בבדיקות הדם לא נמצאו חריגות משמעותיות בספירת הדם.",
    "הומלץ על המשך מעקב ועל הימנעות ממאמץ גופני מוגבר.",
    "המטופל קיבל הסבר מפורט על הטיפול התרופתי ועל תופעות הלוואי האפשריות.",
    "בצילום החזה לא נצפו ממצאים חדשים בהשוואה לצילום הקודם.",
    "במהלך האשפוז לא נרשמו אירועים חריגים והמטופל שיתף פעולה עם הצוות.",
    "הומלץ על תזונה דלת מלח ועל פעילות גופנית מתונה בהתאם ליכולת.",
    "תוצאות בדיקת האולטרסאונד תואמות את הממצאים הקליניים.",
    "בשיחה עם המטופל הובהרה חשיבות נטילת התרופות באופן קבוע.",
    "לא דווח על כאבים, בחילות או הקאות במהלך הלילה.",
    "המטופל הונחה לפנות לחדר מיון במקרה של החמרה בתסמינים.",
    "בבדיקה נוירולוגית לא נמצא ליקוי מוקד והתגובות סימטריות.",
    "ממצאי בדיקת השתן תקינים ואין עדות לזיהום.",
    "הטיפול התרופתי הותאם לתפקודי הכליה והכבד של המטופל.",
]


def israeli_id(rng, eight_digits=False):
    """A valid Israeli ID (check digit included). eight_digits drops the leading zero of a zero-padded ID."""
    d = [0 if eight_digits else rng.randint(1, 9)] + [rng.randint(0, 9) for _ in range(7)]
    total = 0
    for i, x in enumerate(d):
        v = x * (1 if i % 2 == 0 else 2)
        total += v if v < 10 else v - 9
    d.append((10 - total % 10) % 10)
    s = "".join(map(str, d))
    return s[1:] if eight_digits else s


def digits(rng, n):
    return "".join(rng.choice("0123456789") for _ in range(n))


def mobile(rng, style=0):
    p, n = rng.choice(["50", "52", "53", "54", "55", "58"]), digits(rng, 7)
    return [f"0{p}-{n}", f"0{p}{n}", f"+972-{p}-{n}", f"0{p} {n[:3]} {n[3:]}"][style % 4]


def landline(rng):
    return f"0{rng.choice('2348')}-{digits(rng, 7)}"


def name(rng):
    return rng.choice(FIRST), rng.choice(LAST)


def rdate(rng, first_year=1938):
    return rng.randint(1, 28), rng.randint(1, 12), rng.randint(first_year, 2025)


DATE_FORMATS = {
    "date-numeric-dot": lambda d, m, y: f"{d:02d}.{m:02d}.{y}",
    "date-numeric-dot-nopad": lambda d, m, y: f"{d}.{m}.{y}",
    "date-numeric-slash": lambda d, m, y: f"{d:02d}/{m:02d}/{y}",
    "date-numeric-dash": lambda d, m, y: f"{d:02d}-{m:02d}-{y}",
    "date-two-digit-year": lambda d, m, y: f"{d:02d}.{m:02d}.{y % 100:02d}",
    "date-iso": lambda d, m, y: f"{y}-{m:02d}-{d:02d}",
    "date-no-punctuation": lambda d, m, y: f"{d:02d}{m:02d}{y}",
    "date-written-hebrew": lambda d, m, y: f"{d} {MONTHS_HE[m - 1]} {y}",
    "date-written-prefixed-month": lambda d, m, y: f"{d} ב{MONTHS_HE[m - 1]} {y}",
}


def ev(label, value, tag="", policy=False):
    return E(label, value, tag, policy)


# ----------------------------------------------------------------------------------------------------- tier 1
def tier1(rng):
    docs = []
    for i in range(40):
        f, l = name(rng)
        d, m, y = rdate(rng)
        c, c2 = rng.sample(CITIES, 2)
        mo = rng.choice(MED_ORGS)
        o = rng.choice(ORGS)
        age = rng.choice([92, 94, 97, 101])
        young = rng.choice([8, 34, 45, 67, 81])
        templates = [
            ["המטופל ", ev("FIRST_NAME", f), " ", ev("LAST_NAME", l), f" בן {young} אושפז ביום ",
             ev("DATE", DATE_FORMATS["date-numeric-dot"](d, m, y), "date-numeric-dot"), " במחלקה פנימית."],
            ["הגב' ", ev("FIRST_NAME", f), " ", ev("LAST_NAME", l), " מתגוררת ב", ev("LOC", c, "city-prefixed"),
             " והתקשרה למספר ", ev("CONTACT", mobile(rng), "phone-mobile"), " לקביעת תור."],
            ["מטופל בן ", ev("AGE_ABOVE_89", str(age), "age-above-89"), " הגיע לחדר מיון ב", ev("MED_ORG", mo,
             "medorg-prefixed"), " ביום ", ev("DATE", DATE_FORMATS["date-numeric-slash"](d, m, y), "date-numeric-slash"),
             "."],
            ["ת.ז. ", ev("ID", israeli_id(rng), "id-with-context"), " של ", ev("FIRST_NAME", f), " ",
             ev("LAST_NAME", l), ", מספר תיק ", ev("INNER_ID", digits(rng, 8), "inner-id"), "."],
            ["ד\"ר ", ev("FIRST_NAME", f), " ", ev("LAST_NAME", l), " מ", ev("MED_ORG", mo, "medorg-prefixed"),
             " המליץ על טיפול תרופתי חדש."],
            ["פנייה טלפונית מאת ", ev("FIRST_NAME", f), " ", ev("LAST_NAME", l), " למספר ",
             ev("CONTACT", landline(rng), "phone-landline"), f" בנוגע לתוצאות בדיקה."],
            ["המטופלת ", ev("FIRST_NAME", f), f" בת {young} עובדת ב", ev("ORG", o, "org-prefixed"), " ומתגוררת ב",
             ev("LOC", c2, "city-prefixed"), "."],
            ["הילד ", ev("FIRST_NAME", f), " ", ev("LAST_NAME", l), f" בן {rng.randint(3, 15)} הגיע עם הוריו בתאריך ",
             ev("DATE", DATE_FORMATS["date-numeric-dot"](d, m, y), "date-numeric-dot"), "."],
            ["סיכום ביקור: ללא תלונות חדשות, סימנים חיוניים תקינים, המשך טיפול כמקודם."],  # no PHI at all
            ["בדיקות המעבדה של ", ev("FIRST_NAME", f), " ", ev("LAST_NAME", l), " תקינות. מומלץ מעקב בעוד שלושה חודשים."],
        ]
        docs.append(build(templates[i % len(templates)]))
    return docs


# ----------------------------------------------------------------------------------------------------- tier 2
def tier2(rng):
    docs = []
    # dates, every format, in a plain sentence
    for tag, fn in DATE_FORMATS.items():
        for _ in range(3):
            d, m, y = rdate(rng)
            docs.append(build("המטופל אושפז בתאריך ", ev("DATE", fn(d, m, y), tag),
                              " ושוחרר לביתו לאחר שלושה ימים."))
    for _ in range(3):  # prefix before a numeric date: the prefix stays outside the entity
        for pre in ("ב", "ב-", "מ-", "ל-"):
            d, m, y = rdate(rng)
            docs.append(build("הבדיקה בוצעה ", pre, ev("DATE", DATE_FORMATS["date-numeric-dot"](d, m, y),
                                                       "date-with-prefix"), " במרפאה."))
    for hd in HEB_CAL_DATES:
        docs.append(build("המטופל נולד ב", ev("DATE", hd, "date-hebrew-calendar"), " והתגורר בעיר כל חייו."))
    for _ in range(4):
        m = rng.randint(1, 12)
        docs.append(build("הופיעו תסמינים חדשים ב", ev("DATE", MONTHS_HE[m - 1], "date-month-only", True),
                          " ולכן נשלח לבירור."))
        docs.append(build("הוא נבדק לאחרונה ב", ev("DATE", f"{MONTHS_HE[m - 1]} {rng.randint(2015, 2025)}",
                                                     "date-month-year", True), " ונמצא תקין."))
    # IDs and inner ids
    for style in range(6):
        if style == 0:
            docs.append(build("ת.ז. ", ev("ID", israeli_id(rng), "id-with-context"), " של המטופל אומתה בקבלה."))
        elif style == 1:
            docs.append(build("מספר זהות: ", ev("ID", israeli_id(rng), "id-with-context"), "."))
        elif style == 2:
            docs.append(build("ת\"ז ", ev("ID", israeli_id(rng, True), "id-eight-digits"), " (ללא אפס מוביל)."))
        elif style == 3:
            docs.append(build("המטופל זוהה לפי מספר ", ev("ID", israeli_id(rng), "id-no-context"), " במערכת."))
        elif style == 4:
            n = israeli_id(rng)
            docs.append(build("ת.ז. ", ev("ID", f"{n[:8]}-{n[8]}", "id-dash-check-digit"), "."))
        else:
            docs.append(build("דרכון ", ev("ID", digits(rng, 8), "passport-number", True), " הוצג בקבלה."))
    for kind, tpl in [("inner-id", "מספר תיק "), ("inner-id", "מס' מקרה "), ("inner-id", "מספר פנימי "),
                      ("inner-id", "תיק רפואי "), ("inner-id", "אירוע מספר "), ("inner-id-letter", "מספר מטופל H-"),
                      ("inner-id-no-context", "ראו "), ("inner-id", "מקרה ")]:
        n = digits(rng, rng.choice([6, 7, 8]))
        docs.append(build(tpl, ev("INNER_ID", n, kind), " במערכת בית החולים."))
    # phones
    for style in range(4):
        docs.append(build("ניתן ליצור קשר בטלפון ", ev("CONTACT", mobile(rng, style), f"phone-mobile-{style}"), "."))
    for _ in range(3):
        docs.append(build("מספר הטלפון במרפאה הוא ", ev("CONTACT", landline(rng), "phone-landline"), "."))
    docs.append(build("פקס: ", ev("CONTACT", landline(rng), "phone-fax"), " (שעות פעילות בלבד)."))
    # emails
    for local, dom, tag in [("genetics.clinic", "demo-med.org.il", "email-dotted-local"),
                            ("dr.cohen83", "gmail.com", "email-digits"), ("rachel_levi", "walla.co.il", "email-underscore"),
                            ("secretary", "demo-med.org.il", "email-simple"),
                            ("info-dept", "demo-med.org.il", "email-hyphen"),
                            ("a.b.c.d", "demo-med.org.il", "email-many-dots")]:
        docs.append(build("לשאלות ניתן לכתוב לכתובת ", ev("CONTACT", f"{local}@{dom}", tag), " בלבד."))
    docs.append(build("אימייל: ", ev("CONTACT", f"yossi.k@{rng.choice(['gmail.com', 'walla.co.il'])}", "email-after-label"),
                      ", טלפון: ", ev("CONTACT", mobile(rng), "phone-mobile"), "."))
    # names with prefixes, hyphenated and compound surnames, titles
    f, l = name(rng)
    docs.append(build("הומלץ ל", ev("FIRST_NAME", f, "name-prefixed"), " ", ev("LAST_NAME", l, "name-prefixed"),
                      " להמשיך בטיפול."))
    f, l = name(rng)
    docs.append(build("התקבל מכתב מ", ev("FIRST_NAME", f, "name-prefixed"), " ", ev("LAST_NAME", l, "name-prefixed"),
                      "."))
    docs.append(build("הוסבר ל", ev("FIRST_NAME", rng.choice(FIRST), "name-prefixed"), " מה צפוי בהמשך."))
    docs.append(build("ידוע כי ", ev("FIRST_NAME", rng.choice(FIRST), "name-bare"), " נוטל תרופות באופן קבוע."))
    for _ in range(4):
        f = rng.choice(FIRST)
        hl = rng.choice(HYPHEN_LAST)
        docs.append(build("המטופלת ", ev("FIRST_NAME", f), " ", ev("LAST_NAME", hl, "surname-hyphenated"),
                          " הגיעה לביקורת."))
    for title in ["ד\"ר", "פרופ'", "הגב'", "מר", "האחות"]:
        f, l = name(rng)
        docs.append(build(title, " ", ev("FIRST_NAME", f, "name-with-title"), " ", ev("LAST_NAME", l,
                                                                                       "name-with-title"),
                          " ביצע את ההערכה."))
    docs.append(build("מר ", ev("LAST_NAME", rng.choice(LAST), "last-name-only"), " הגיע לביקורת מתוכננת."))
    docs.append(build("ל", ev("FIRST_NAME", rng.choice(FIRST), "first-name-only"), " יש רגישות לפניצילין."))
    # cities and addresses with prefixes
    for pre in ("ב", "מ", "ל", "ה"):
        c = rng.choice(CITIES)
        docs.append(build("המטופל " + {"ב": "מתגורר ב", "מ": "הגיע מ", "ל": "הופנה ל", "ה": "אשתו גרה בעיר "}[pre],
                          ev("LOC", c, "city-prefixed" if pre != "ה" else "city-bare"), "."))
    for _ in range(3):
        st, c = rng.choice(STREETS), rng.choice(CITIES)
        docs.append(build("כתובת המטופל: ", ev("LOC", f"רחוב {st} {rng.randint(1, 120)}, {c}", "address", True), "."))
    # organisations with prefixes
    for _ in range(4):
        docs.append(build("הועבר ל", ev("MED_ORG", rng.choice(MED_ORGS), "medorg-prefixed"), " להמשך בירור."))
    docs.append(build("הגיע ב", ev("MED_ORG", 'רמב"ם', "medorg-short-prefixed"), " באמבולנס."))
    docs.append(build("שוחרר מ", ev("MED_ORG", "איכילוב", "medorg-short-prefixed"), " לביתו."))
    for _ in range(3):
        docs.append(build("הוא עובד ב", ev("ORG", rng.choice(ORGS), "org-prefixed"), " כבר עשר שנים."))
    # ages: the boundary is 90 (above 89)
    for a, mask in [(45, False), (88, False), (89, False), (90, True), (93, True), (101, True)]:
        if mask:
            docs.append(build(f"מטופלת בת ", ev("AGE_ABOVE_89", str(a), "age-above-89"), " הגיעה עם בנה."))
        else:
            docs.append(build("מטופלת בת ", K(str(a), f"age {a} is not above 89"), " הגיעה עם בנה."))
    docs.append(build("גיל ", ev("AGE_ABOVE_89", "95", "age-above-89-context"), ", עצמאי בפעולות היומיום."))
    return docs


# ----------------------------------------------------------------------------------------------------- tier 3
def tier3(rng):
    """Traps: things that look like PHI but are not, and names that look like ordinary words."""
    docs = []
    eponyms = ["מחלת פרקינסון", "תסמונת דאון", "מחלת קרוהן", "מחלת אלצהיימר", "תסמונת גילן-ברה", "מחלת הודג'קין",
               "תסמונת קושינג", "מחלת בכטרב"]
    for e in eponyms:
        docs.append(build("המטופל סובל מ", K(e, "disease named after a person"), " מזה מספר שנים."))
    for tail in ["אי ספיקת לב", "מחלת לב איסכמית", "צנתור לב", "קוצר נשימה ולב מוגדל"]:
        pre, _, organ = tail.rpartition("לב")
        docs.append(build("האבחנה: ", pre, K("לב", "organ, not a name"), organ, "."))
    docs.append(build("המטופל ", K("אושר", "verb 'was approved'"), " לשחרור לאחר בדיקת רופא."))
    docs.append(build("הטיפול ", K("אושר", "verb 'was approved'"), " על ידי הוועדה."))
    docs.append(build("נרשם ", K("גל", "word 'wave'"), " חום בשבוע האחרון ובעקבותיו ירידה בתיאבון."))
    docs.append(build("המטופל קיבל ", K("אקמול", "drug"), " ו", K("אספירין", "drug"), " לפי הצורך."))
    docs.append(build("טיפול ב", K("קומדין", "drug"), " ובדיקת ", K("INR", "lab test"), " שבועית."))
    docs.append(build("חשד ל", K("קדחת ים תיכונית משפחתית", "disease containing a place-like word"), "."))
    # numbers that are not identifiers (a bare 'any 5+ digits' rule would mask these)
    for what, num, unit in [("מינון הפרין", "10000", " יחידות"), ("ספירת תאים לבנים", "15000", " למיקרוליטר"),
                            ("נפח השתן", "12000", " מ\"ל ביממה"), ("רמת פריטין", "24500", " ננוגרם"),
                            ("מספר טסיות", "250000", " למיקרוליטר")]:
        docs.append(build(what, " ", K(num, "lab value or dose, not an ID"), unit, "."))
    docs.append(build("לחץ דם ", K("130/80", "blood pressure, not a date"), ", דופק ", K("88", "pulse"), ", חום ",
                      K("38.5", "temperature"), "."))
    docs.append(build("קוד אבחנה ", K("I10", "ICD code"), " וגם ", K("E11.9", "ICD code"), "."))
    docs.append(build("הוזמן לבדיקה בשעה ", K("08:30", "time"), " בבוקר."))
    docs.append(build("CRP ", K("125", "lab value"), " ו-", K("ESR", "lab test"), " ", K("60", "lab value"), "."))
    docs.append(build("נלקחו ", K("5", "count"), " מ\"ג ביום במשך ", K("14", "count"), " ימים."))
    docs.append(build("ציון ", K("12/15", "score, not a date"), " בסולם ההערכה."))
    # generic places/units that are not named entities
    for g in ["חדר מיון", "יחידת טיפול נמרץ", "המחלקה לכירורגיה", "מחלקה פנימית ב'", "המכון לקרדיולוגיה"]:
        docs.append(build("המטופל הועבר ל", K(g, "generic unit, not a named organisation"), " לצורך המשך טיפול."))
    # names that are ALSO ordinary words: the name must be masked, the plain use must stay
    f, l = name(rng)
    docs.append(build("המטופל ", ev("FIRST_NAME", "לב", "name-is-common-word"), " ", ev("LAST_NAME", l),
                      " מאובחן עם אי ספיקת ", K("לב", "organ; same string as the patient's first name"), "."))
    docs.append(build("המטופל ", ev("FIRST_NAME", "אושר", "name-is-common-word"), " ", ev("LAST_NAME", rng.choice(LAST)),
                      " ", K("אושר", "verb; same string as the patient's first name"), " לשחרור."))
    docs.append(build("הגב' ", ev("FIRST_NAME", "גל", "name-is-common-word"), " ", ev("LAST_NAME", rng.choice(LAST)),
                      " סובלת מ", K("גל", "word 'wave'"), " חום."))
    docs.append(build("המטופל ", ev("FIRST_NAME", "ניסן", "name-is-month-name"), " ", ev("LAST_NAME", rng.choice(LAST)),
                      " אושפז בניסן ", ev("DATE", "תשפ״ד", "date-hebrew-calendar", True), "."))
    docs.append(build("הגב' ", ev("FIRST_NAME", "אדר", "name-is-month-name"), " ", ev("LAST_NAME", rng.choice(LAST)),
                      " הגיעה לביקורת."))
    docs.append(build(ev("FIRST_NAME", "שחר", "name-is-common-word"), " ", ev("LAST_NAME", rng.choice(LAST)),
                      " סיים את הטיפול."))
    # true PHI inside medical-sounding sentences
    for _ in range(6):
        f, l = name(rng)
        d, m, y = rdate(rng)
        docs.append(build("מטופל עם ", K("מחלת פרקינסון", "disease named after a person"), ", ",
                          ev("FIRST_NAME", f), " ", ev("LAST_NAME", l), ", נבדק ביום ",
                          ev("DATE", DATE_FORMATS["date-numeric-dot"](d, m, y), "date-numeric-dot"),
                          " ונמצא יציב."))
    for _ in range(4):
        f, l = name(rng)
        docs.append(build("מינון הפרין ", K("10000", "dose, not an ID"), " יחידות. ת.ז. ",
                          ev("ID", israeli_id(rng), "id-with-context"), " של ", ev("FIRST_NAME", f), " ",
                          ev("LAST_NAME", l), "."))
    return docs


# ----------------------------------------------------------------------------------------------------- tier 4
def note(rng, k):
    f, l = name(rng)
    f2, l2 = name(rng)
    dr_f, dr_l = name(rng)
    nurse = rng.choice(FIRST)
    c, c2 = rng.sample(CITIES, 2)
    mo = rng.choice(MED_ORGS)
    d1, d2, d3 = rdate(rng, 2015), rdate(rng, 2015), rdate(rng, 2015)
    dob = rdate(rng)
    fmt = lambda t, tag: ev("DATE", DATE_FORMATS[tag](*t), tag)
    age = rng.choice([91, 94, 96])
    parts = [
        "סיכום אשפוז\n",
        "שם המטופל: ", ev("FIRST_NAME", f), " ", ev("LAST_NAME", l), "\n",
        "ת.ז.: ", ev("ID", israeli_id(rng), "id-with-context"), "\n",
        "מספר תיק: ", ev("INNER_ID", digits(rng, 7), "inner-id"), "\n",
        "תאריך לידה: ", fmt(dob, "date-numeric-slash"), "\n\n",
        "מחלה נוכחית: ", "מטופל בן ", ev("AGE_ABOVE_89", str(age), "age-above-89"), " המתגורר ב",
        ev("LOC", c, "city-prefixed"), ", הגיע ", "ב", ev("MED_ORG", mo, "medorg-prefixed"), " ביום ",
        fmt(d1, "date-numeric-dot"), " עם קוצר נשימה. ", ev("FIRST_NAME", f, "name-repeat"), " מדווח על החמרה מזה כמה ימים. ",
        rng.choice(FILLER), " ", rng.choice(FILLER), "\n\n",
        "רקע משפחתי: ", "בנו ", ev("FIRST_NAME", f2, "name-relative"), " ", ev("LAST_NAME", l, "name-repeat"),
        " מלווה אותו ונמצא בקשר בטלפון ", ev("CONTACT", mobile(rng), "phone-mobile"), ". לדברי מר ",
        ev("LAST_NAME", l, "name-repeat-title"), " אין היסטוריה משפחתית מחשידה. ", rng.choice(FILLER), "\n\n",
        "מהלך האשפוז: ", rng.choice(FILLER), " ", "ד\"ר ", ev("FIRST_NAME", dr_f, "name-staff"), " ",
        ev("LAST_NAME", dr_l, "name-staff"), " והאחות ", ev("FIRST_NAME", nurse, "name-staff"), " טיפלו ב",
        ev("FIRST_NAME", f, "name-repeat-prefixed"), " לאורך הלילה. ", rng.choice(FILLER), " ",
        "ביום ", fmt(d2, "date-written-hebrew"), " עבר צילום חזה. ", rng.choice(FILLER), "\n\n",
        "המלצות בשחרור: ", rng.choice(FILLER), " ", "מעקב במרפאה ב", ev("MED_ORG", rng.choice(MED_ORGS),
                                                                       "medorg-prefixed"), " בעוד חודש, ",
        "ובדיקת דם בתאריך ", fmt(d3, "date-numeric-dot-nopad"), ". לבירורים: ",
        ev("CONTACT", f"clinic.{rng.choice(['a', 'b', 'c'])}@demo-med.org.il", "email-dotted-local"), ", ",
        ev("CONTACT", landline(rng), "phone-landline"), ". ",
        "שוחרר לביתו של ", ev("FIRST_NAME", f), " ", ev("LAST_NAME", l, "name-repeat"), " בביתם.\n",
    ]
    return build(parts)


def tier4(rng):
    return [note(rng, k) for k in range(12)]


# ----------------------------------------------------------------------------------------------------- tier 5
def long_doc(rng, target_words):
    """Long document: PHI at the very start, every few paragraphs, and at the very end; the patient's name repeats."""
    f, l = name(rng)
    parts = ["סיכום רפואי מפורט\n", "המטופל: ", ev("FIRST_NAME", f), " ", ev("LAST_NAME", l), ", ת.ז. ",
             ev("ID", israeli_id(rng), "id-with-context"), "\n\n"]
    kinds = ["phone", "date", "city", "medorg", "name-repeat", "inner", "email", "org", "age", "relative"]
    n = 0
    while len(build(parts)[0].split()) < target_words:
        parts.append(" ".join(rng.sample(FILLER, 6)) + "\n")
        n += 1
        if n % 2 == 0:
            kind = kinds[(n // 2) % len(kinds)]
            c = rng.choice(CITIES)
            if kind == "phone":
                parts += ["ליצירת קשר: ", ev("CONTACT", mobile(rng, n), "phone-mobile"), ". "]
            elif kind == "date":
                parts += ["בוצעה בדיקה ביום ", ev("DATE", DATE_FORMATS["date-numeric-dot"](*rdate(rng)),
                                                   "date-numeric-dot"), ". "]
            elif kind == "city":
                parts += ["המטופל חזר ל", ev("LOC", c, "city-prefixed"), " לאחר השחרור. "]
            elif kind == "medorg":
                parts += ["הופנה ל", ev("MED_ORG", rng.choice(MED_ORGS), "medorg-prefixed"), " להמשך מעקב. "]
            elif kind == "name-repeat":
                parts += ["בשיחה עם ", ev("FIRST_NAME", f, "name-repeat"), " ", ev("LAST_NAME", l, "name-repeat"),
                          " הוסבר התהליך. ", "ל", ev("FIRST_NAME", f, "name-repeat-prefixed"), " ניתנו הנחיות בכתב. "]
            elif kind == "inner":
                parts += ["מספר תיק ", ev("INNER_ID", digits(rng, 7), "inner-id"), ". "]
            elif kind == "email":
                parts += ["כתובת: ", ev("CONTACT", f"unit.{rng.choice('abcd')}@demo-med.org.il", "email-dotted-local"),
                          ". "]
            elif kind == "org":
                parts += ["הוא עובד ב", ev("ORG", rng.choice(ORGS), "org-prefixed"), ". "]
            elif kind == "age":
                parts += ["מטופל בן ", ev("AGE_ABOVE_89", str(rng.choice([90, 92, 95])), "age-above-89"), ". "]
            else:
                f2, _ = name(rng)
                parts += ["בנו ", ev("FIRST_NAME", f2, "name-relative"), " ", ev("LAST_NAME", l, "name-repeat"),
                          " נמצא במקום. "]
    parts += ["\nבסוף המסמך: ", ev("FIRST_NAME", f, "name-repeat"), " ", ev("LAST_NAME", l, "name-repeat"),
              ", ", ev("CONTACT", mobile(rng), "phone-mobile"), ", ",
              ev("DATE", DATE_FORMATS["date-numeric-dot"](*rdate(rng)), "date-numeric-dot"), "."]
    return build(parts)


def tier5(rng):
    return [long_doc(rng, w) for w in (700, 1500, 3000, 5000)]


# --------------------------------------------------------------------------------------------------------- IO
BUILDERS = {"t1_simple": tier1, "t2_formats": tier2, "t3_traps": tier3, "t4_notes": tier4, "t5_long": tier5}


def check(text, spans, keeps):
    taken = sorted([(s["start"], s["end"]) for s in spans] + [(k["start"], k["end"]) for k in keeps])
    for (a, b), (c, d) in zip(taken, taken[1:]):
        assert b <= c, "overlapping spans"
    assert all(0 <= a < b <= len(text) for a, b in taken)


def csv_quote(s):
    return '"' + s.replace('"', '""') + '"'


def main(out=OUT):
    out = Path(out)
    out.mkdir(exist_ok=True)
    for seed, tier in enumerate(TIERS):
        docs = BUILDERS[tier](random.Random(1000 + seed))
        rows, lines = [], []
        for i, (text, spans, keeps) in enumerate(docs, 1):
            check(text, spans, keeps)
            rows.append(f"{i},{csv_quote(text)}")
            lines.append(json.dumps({"id": i, "tier": tier, "text": text, "label": spans, "keep": keeps},
                                    ensure_ascii=False))
        # utf-8, LF newlines, so the files are identical on every machine
        (out / f"{tier}.csv").write_bytes(("id,text\n" + "\n".join(rows) + "\n").encode("utf-8"))
        (out / f"{tier}.truth.jsonl").write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
        n_ent = sum(len(d[1]) for d in docs)
        print(f"{tier}: {len(docs)} documents, {n_ent} entities to mask, {sum(len(d[2]) for d in docs)} traps")


if __name__ == "__main__":
    main()
