"""check_duplicates.py on a tiny synthetic data set: counts, leakage, conflicts, cleaning. No real data, no model."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import check_duplicates as cd


def rec(i, text, labels=()):
    return {"id": i, "text": text, "label": [{"start": text.index(v), "end": text.index(v) + len(v), "label": l}
                                              for v, l in labels]}


A = "המטופל דוד כהן אושפז ביום 15.03.2024 בבית החולים איכילוב עם קוצר נשימה ולחץ דם גבוה"
B = "המטופלת שרה לוי התקשרה למספר 053-1234567 לקביעת תור במרפאה בחיפה לבדיקת מעקב שגרתית"
C = "ללא תלונות חדשות בבדיקה סימנים חיוניים תקינים והמשך טיפול כמקודם וקביעת תור לבדיקת מעקב"
D = "מטופל בן 92 עם יתר לחץ דם הגיע לחדר מיון ברמב\"ם בתאריך 01.02.2024 עם כאבים בחזה"
NEAR = A + " ובדיקה"


def write(tmp, name, recs):
    (tmp / f"{name}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n",
                                       encoding="utf-8")


def make(tmp):
    names = [("דוד", "FIRST_NAME")]
    write(tmp, "train", [
        rec("1.txt", A, names),
        rec("second_1.txt", A, names),            # identical copy, same split
        rec("2.txt", B, [("שרה", "FIRST_NAME")]),
        rec("3.txt", C),
        rec("4.txt", D, [("92", "AGE_ABOVE_89")]),
        rec("second_4.txt", D),                   # same text, different annotations
        rec("5.txt", NEAR, names),                # near duplicate of A (A is in test)
    ])
    write(tmp, "validation", [rec("6.txt", C)])  # twin of train 3.txt
    write(tmp, "test", [rec("7.txt", A, names), rec("8.txt", "טקסט ייחודי לגמרי שאין לו תאום באף קובץ אחר כלל")])


def run(tmp, capsys, *extra):
    cd.main(["--data-dir", str(tmp), *extra])
    return capsys.readouterr().out


def test_counts_and_leakage(tmp_path, capsys):
    make(tmp_path)
    out = run(tmp_path, capsys)
    assert "DOCUMENTS: 10 (train 7, validation 1, test 2)" in out
    assert "2 ids with a second_/copy_/dup_ prefix" in out
    assert "texts that occur more than once: 3" in out                 # A (3 copies), D (2), C (2)
    assert "CONFLICTING annotations: 1 groups (2 documents)" in out    # D
    assert "test documents whose text also occurs in train: 1 of 2" in out
    assert "validation documents whose text also occurs in train: 1 of 1" in out
    assert "test documents with a near-duplicate in train: 1 of 2" in out  # 7.txt ~ 5.txt


def test_output_has_no_text_or_ids(tmp_path, capsys):
    make(tmp_path)
    out = run(tmp_path, capsys)
    assert "דוד" not in out and "1.txt" not in out and "second_1" not in out


def test_clean_keeps_the_evaluated_copy_and_drops_conflicts(tmp_path, capsys):
    make(tmp_path)
    clean = tmp_path / "clean"
    run(tmp_path, capsys, "--write-clean", str(clean), "--drop-near-from-train")
    got = {n: [json.loads(l)["id"] for l in (clean / f"{n}.jsonl").read_text(encoding="utf-8").splitlines()]
           for n in ("train", "validation", "test")}
    assert got["test"] == ["7.txt", "8.txt"]
    assert got["validation"] == ["6.txt"]
    assert got["train"] == ["2.txt"]  # A copies (test has one), D conflict, C (validation has it), 5.txt near-twin: gone


def test_conflicts_first_keeps_one(tmp_path, capsys):
    make(tmp_path)
    clean = tmp_path / "clean"
    run(tmp_path, capsys, "--write-clean", str(clean), "--conflicts", "first")
    ids = [json.loads(l)["id"] for l in (clean / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    assert sum(i in ("4.txt", "second_4.txt") for i in ids) == 1


def test_single_file_mode(tmp_path, capsys):
    f = tmp_path / "all.jsonl"
    f.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in
                           [rec("a", A, [("דוד", "FIRST_NAME")]), rec("second_a", A, [("דוד", "FIRST_NAME")])]),
                 encoding="utf-8")
    cd.main(["--data", str(f)])
    out = capsys.readouterr().out
    assert "identical annotations: 1 groups (1 redundant documents)" in out
