# Hospital test results, issues and candidate fixes

Source: the e-mail thread "ichilov_anonimizer DOC pipeline - תוצאות בדיקה" (Apr-Nov 2026) and the three result tables in
it, plus our own dummy-run findings ([DUMMY_RUN_2026-11-08.md](DUMMY_RUN_2026-11-08.md)) and a read of
[HebSafeHarbor](https://github.com/8400TheHealthNetwork/HebSafeHarbor) (HSH, MIT licence).
Status: **analysis only, nothing here is fixed yet.** The fake test set with ground truth
(`inference/dummy_data/`, see its README) is how each fix will be measured.

## 1. How the hospital measures (read this before comparing any numbers)

- Population: June-2025 records from the free-text "descriptions" table (current illness, summaries; long texts),
  12,836 / 12,837 records that contain at least one patient identifier.
- For each record they take the patient's own value (first name, last name, ID, case number, phone, city, a hospital or
  organisation from their catalogue, age above 89, a date's month) and check whether it **still appears anywhere in the
  de-identified text** ("Exist After hashing" out of "Exist in original"). Success % = 1 - after/original.
- Caveats that matter for interpretation:
  1. It measures **recall only**. No false-positive (over-masking) check was done; the hospital says so itself.
  2. A match counts even when the string is a different person or an ordinary word: the patient "אושר", "אדר", "ניסן",
     "סיון", "אייר" also exist as words and months. So the real leak rate of names is probably lower than reported, by an
     unknown amount; the hospital checked patient names only, not family or staff.
  3. Several parameters have few values (CaseNum 22, Phone 104, Id_Num 336): one miss in CaseNum is 4.5 points.
  4. The April numbers and the later ones are not comparable: the counting method changed slightly ("שינוי קל בשיטת הבדיקה"),
     which is why "Exist in original" differs between the tables. Compare only within one table.
  5. The check is not span based, so it cannot tell "not detected" from "detected but the same string survived elsewhere".

## 2. The three tables, in time order

**(a) 26 Apr 2026, first run, old pipeline** (CPU, about 5.5 hours; other counting method)

| Parameter | Left after / in original | Success |
|---|---|---|
| First name | 541 / 2156 | 74.9% |
| Last name | 120 / 773 | 84.5% |
| Id_Num | 4 / 55 | 92.7% |
| CaseNum | 0 / 6 | 100% |
| Phone | 1 / 53 | 98.1% |
| City | 20 / 181 | 89.0% |
| Hospital/organisation | 8 / 492 | 98.4% |
| Age > 89 | 1 / 232 | 99.6% |
| Month | 1121 / 8367 | 86.6% |

**(b) 8 Sep 2026, new "doc pipeline" (this repo) vs the old pipeline, recounted with the same method**

| Parameter | Old pipeline (left / success) | New pipeline (left / success) |
|---|---|---|
| First name | 583 / 75.2% | 581 / 75.3% |
| Last name | 129 / 84.1% | 123 / 84.8% |
| Id_Num | 7 / 97.9% | 2 / 99.4% |
| CaseNum | 2 / 90.9% | 2 / 90.9% |
| Phone | 1 / 99.0% | 1 / 99.0% |
| City | 20 / 89.7% | 21 / 89.2% |
| Hospital/organisation | 18 / 92.3% | 12 / 94.8% |
| Age > 89 | 2 / 99.3% | 2 / 99.3% |
| Month | 1880 / 79.0% | 1834 / 79.5% |

Hospital's reading: no significant change. (The new pipeline is marginally better on ID and organisations, marginally
worse on city; all within noise.)

**(c) 4 Nov 2026, comparison with HebSafeHarbor** (same file; HSH is the tool the hospital already knows)

| Parameter | In original | HSH left / success | Ours left / success |
|---|---|---|---|
| First name | 2353 | 630 / 73.2% | 581 / 75.3% |
| Last name | 811 | 122 / 85.0% | 123 / 84.8% |
| Id_Num | 336 | 1 / 99.7% | 2 / 99.4% |
| **CaseNum** | 22 | **0 / 100%** | 2 / 90.9% |
| Phone | 104 | 1 / 99.0% | 1 / 99.0% |
| City | 194 | 96 / 50.5% | 21 / 89.2% |
| Hospital/organisation | 233 | 22 / 90.6% | 12 / 94.8% |
| Age > 89 | 267 | 267 / 0.0% (not handled) | 2 / 99.3% |
| Month | 8937 | 8906 / 0.3% (not handled) | 1834 / 79.5% |

Reading: HSH is better only on CaseNum (and a hair on last name and ID); we are far better on city, organisation, age
and month. HSH does not handle age or month at all. The hospital's own suggestion (by a project lead): CaseNum is a
templated type, so it should be fixable "even with a regex".

## 3. Issues

### 3.1 Reported by the hospital

| # | Issue | Evidence | Most likely cause |
|---|---|---|---|
| H1 | About 1 in 4 first names survives | 581 of 2353 | Hard for the model (names that are also words, rare names, names in odd contexts); inflated by the measurement caveat 2; partly H4 |
| H2 | About 1 in 6 last names survives | 123 of 811 | Same |
| H3 | Months: 1 in 5 survives; dates sometimes partly masked (day masked, month not) | 1834 of 8937 | Month-only mentions and Hebrew-calendar dates are not recognised (our date stage only knows Gregorian numeric and "D month YYYY"); see P2 |
| H4 | A value is masked at the start of the text but not at its other occurrences | hospital's repeated observation | Each window/sentence is decided independently; no document-level consistency |
| H5 | City 11% survives | 21 of 194 | Model recall on rarer cities and prefixed forms (P3) |
| H6 | Organisations 5% survive | 12 of 233 | Catalogue names the model has not seen; prefixes |
| H7 | CaseNum 2 of 22 survive | 90.9% | No rule for long digit runs; the model only knows `INNER_ID` from context |
| H8 | Id_Num 2 of 336 survive | 99.4% | Unusual formats; no checksum rule |
| H9 | Phone 1 of 104, Age > 89 2 of 267 | 99%+ | Residual |
| H10 | Names that are also words: "אושר", "אדר", "ניסן", "סיון", "אייר" | hospital note | Inherent ambiguity; affects both the model and the measurement |
| H11 | Speed: CPU-only run of about 5.5 h; the hospital server ran at 100% CPU and was stopped after more than 4 h; about 1 h per 5000 records on a GPU | thread | Per-row processing, no batching; the run was on CPU |
| H12 | No false-positive measurement exists | hospital note | Needs a labelled non-PHI sample (hospital will prepare 50 texts from 4 sources) |

Requests and offers in the thread: a **gold set** (100 records per parameter, only the sentence containing the missed
value), the two **modes** (strict / lenient; the system has no such mode today, repeated-value masking would be one),
a corrected version before the model is handed to the second hospital, synthetic records for the failure cases
(our offer), and the hospital's own small changes to the script (utf-8-sig encoding, TABs replaced by spaces, `.txt`
output, row counter, start/end time). Already here: TAB replacement, `.txt` output, row progress, start/end time, input read as `utf-8-sig`.
To check: the encoding of the written CSV (Excel).

### 3.2 Found in our own dummy run (60 rows) and code reading

| # | Issue | Evidence |
|---|---|---|
| P1 | **Privacy leak: email prefix** | `genetics.clinic@...` -> `genetics.[CONTACT]` |
| P2 | Dates split into pieces; written dates garbled | `15.03.2024` -> `[DATE].[DATE].[DATE]` (19 of 30 rows). `2 במרץ 2024` -> `[DATE]-[CONTACT]02`. **Cause of the written-date garbling is confirmed:** `normalize_dates_stage` rewrites it to `2024-03-02` before the model sees it. Numeric dates keep their `dd.mm.yyyy` shape (only zero-padded), so there the cause is the model/aggregation labelling each piece separately |
| P3 | Hebrew prefix absorbed into the entity | `בחיפה` -> `[LOC]`, `ברמב"ם` -> `[MED_ORG]` (training data keeps the prefix outside the entity) |
| P4 | Hyphenated surname split, hyphen lost | `אלמוג-כץ` -> `[LAST_NAME][LAST_NAME]` |
| M1 | Model leak: a surname missed (`פנינה אוחנה` -> `פנינה [LAST_NAME]`), a compound foreign surname half-masked | dummy-run doc |
| M2 | Model labels: inner ids `987654` / `H-456789` as CONTACT instead of INNER_ID; a kibbutz as MED_ORG instead of LOC; `אליהו משה` as FIRST+FIRST | dummy-run doc |
| D1 | **Offset drift**: entity offsets in the JSON/CSV are in the *normalised* text, not in the original text, so for rewritten dates and `[ID]` placeholders the spans do not fit `text` (already a known quirk in `test_deidentification_regression.py`) | the scorer counts these documents separately |
| D2 | `normalize_dates_stage` docstring promises ISO, numeric dates actually stay `dd.mm.yyyy`, written ones become ISO: two formats, one of which the model was not trained on | code |
| D3 | Rows are processed one at a time | code |
| D4 | The ID stage only fires after `ת.ז.`/`תז`/`מספר דרכון` followed by 7-9 digits | code |
| D5 (housekeeping) | `inference/tests/test_deidentification_regression.py` had been left without its gold file when old outputs were removed from `dummy_data/`; restored to `inference/tests/gold/` (all 61 tests pass) | git history |

## 4. Candidate fixes

### 4.1 Deterministic rules, in the order I would do them

Ideas marked HSH come from reading HebSafeHarbor; the licence allows reuse, but I would write our own small versions, not
import its Presidio / HebSpacy stack or its large lexicons.

| # | Fix | Targets | Source / note |
|---|---|---|---|
| F1 | **Digit-run rule:** any run of 5 or more digits (dashes allowed) that is still visible after the model is masked as `ID` / `INNER_ID`. Exclude digits inside already-masked spans, decimals, times, `130/80`, and dose/lab contexts via a small unit list | H7 (CaseNum), H8 | HSH `GeneralIdRecognizer` (`(\d[-]?){4,}\d`, confidence 0.6). Risk: lab values such as `15000` (tier 3 has these traps on purpose) |
| F2 | **Israeli ID checksum** for 8-9 digit runs, with or without a dash before the check digit, to raise confidence of an ID decision | H8 | HSH `IsraeliIdNumberRecognizer`; its check uses `digit > 10` where `>= 10` is correct, so do not copy it |
| F3 | **Email as one unit:** extend any CONTACT span that touches an `@` to the whole address (previous whitespace to next whitespace), also mask bare emails the model misses | P1 | Regex, trivial |
| F4 | **Merge adjacent DATE pieces** separated only by `. / - ,` or spaces, and stop rewriting written dates (or keep an offset map) | P2, D1, D2, H3 | Our code |
| F5 | **Hebrew date grammar:** Hebrew calendar dates (`י״א בתשרי תשפ״ד`), month-year, day-month, written months with prefixes (ב ל מ מה ה), no-punctuation dates (`10122018`), dates with spaces around the separator | H3 | HSH `date_regex.py` has these patterns |
| F6 | **Prefix-aware span repair:** if a prediction starts with ב/ל/מ/ה/ו/ש and the rest is a known entity or the prefix is followed by the entity, move the boundary | P3 | Training convention: prefix outside |
| F7 | **Hyphen joining:** merge two LAST_NAME predictions separated by a hyphen into one | P4 | Trivial |
| F8 | **Document-level consistency:** after detection, mask every other occurrence of a masked name / city / organisation value in the same document (prefix tolerant), with a stop-list of ordinary words (לב, גל, אושר...) and a strict/lenient switch | H4, H1, H2, H10 | The hospital's "mode" idea. Needs the tier-3 and tier-4 tests, because it can over-mask the organ "לב" |
| F9 | **Medical stop-list:** drop name/organisation predictions that coincide with a disease, drug or test term (eponyms: פרקינסון, דאון, קרוהן...) | over-masking | HSH `MedicalPostConsolidator`; build a small list from our own false positives, not their lexicons |
| F10 | **City and institution gazetteer** as a data file (names only), prefix tolerant, used to catch what the model misses | H5, H6 | HSH `city_utils.py`, hospital lexicon; take the names, not the code |
| F11 | Birth date vs medical date by context (`נולד`, `ת.לידה` within 10 characters before the date) | later policy | HSH `DateEntitySplitter`; not needed for masking |
| F12 | Context words that raise confidence of an ID/CONTACT guess (`ת.ז`, `טלפון`, `מייל`, `מס' מקרה`) | H7, M1/M2 | HSH `CATEGORY_TO_CONTEXT_PHRASES` |

Not worth taking: HSH's anonymiser operators (we mask, we do not replace), its full lexicons, its dependency stack.

### 4.2 Data and model

| # | Fix | Targets |
|---|---|---|
| T1 | Add hard cases to the training data from the hospital's "gold records": the sentences where a value survived, 100 per parameter | H1-H9 |
| T2 | Synthetic records for the failure shapes (name that is also a word, Hebrew calendar dates, case numbers in many layouts, prefixed cities/organisations, hyphenated surnames, emails) - our own idea in the thread; the generator in `inference/make_dummy_data.py` is a starting point | H1, H2, H3, H5-H7, P1-P4 |
| T3 | Add negative examples (organ "לב", verb "אושר", eponymous diseases, dose and lab numbers) so the model does not learn the opposite | H10, H12 |
| T4 | Check the annotation convention on prefixes and on addresses before training more (P3) | P3 |
| T5 | After a baseline training run, re-check M1/M2 and the name rows | H1, H2, M1, M2 |

### 4.3 Operations

| # | Fix | Targets |
|---|---|---|
| O1 | Batch the NER calls (and sort by length) so a GPU is used properly; report rows per second | H11, D3 |
| O2 | Write the CSV in `utf-8-sig` if Excel is the consumer (progress, timing and TAB replacement already exist) | thread |
| O3 | Ask the hospital for the false-positive set (50 texts, 4 sources) so precision can be measured, not only recall | H12 |

## 5. How we will measure

`inference/dummy_data/` now has five tiers with exact ground truth, and `inference/score_dummy.py` scores a run.
It reports per label and per situation tag: fully masked, partial, missed, right label, exact boundaries, and the
hospital's own "string still survives" check, plus over-masking on declared traps. The mapping:

| Issue | Where it shows up |
|---|---|
| P1 | `t2_formats` `CONTACT: email-*` rows; `t4_notes` |
| P2, H3 | `t2_formats` `DATE: date-*` rows (9 numeric/written formats, Hebrew calendar, month only, with prefixes) |
| P3 | `*-prefixed` tags |
| P4 | `LAST_NAME: surname-hyphenated` |
| H4 | `t4_notes`, `t5_long` (`name-repeat*` tags, and the "survive" column) |
| H7, H8, F1, F2 | `t2_formats` `INNER_ID`/`ID` rows; `t3_traps` number traps (must stay readable) |
| H10, F8, F9 | `t3_traps` |
| Chunking, long documents | `t5_long` (700 to 5000 words) |

Rules of the test set: everything is invented; truth is generated together with the text, so it is exact; tags marked
`policy` are conventions we assumed (month alone as a date, a street address as one LOC span, a passport number as ID)
and must be confirmed against the real annotation guidelines before anyone treats a failure there as a bug.

## 6. Suggested order

1. Run `inference\run_dummy.bat` on the GPU box, paste the scoring summary back: this gives the real baseline per
   situation (this is what we compare every fix against).
2. F3, F1+F2, F4+F5 (rules with clear, testable behaviour), then F6/F7.
3. F8 behind a switch, after tier 3/4 show what it costs.
4. Baseline training in parallel; T1-T3 when the hospital's gold records arrive.
