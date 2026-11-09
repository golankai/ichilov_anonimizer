# Running the new version

The repo has a lot of files. You do not need to understand them: follow the steps below. The process is the same as before: install, place the model, run.

What is new in this version: a model trained on real plus synthetic data, and a few deterministic rules after the model (dates, e-mails, phone numbers, case and ID numbers, age above 89, and masking a name again when it repeats later in the same document).

## Step 0: get the latest code

If you already have a copy of the repo, run this inside its folder:

```text
git pull
```

Otherwise:

```text
git clone git@github.com:golankai/ichilov_anonimizer.git
```

## Step 1: install (once)

Python 3.12 is required. Double-click `setup.bat` (Linux/macOS: `./setup.sh`).
At the end it prints `Setup finished`. If you get an error, send it to me.

## Step 2: place the models

The models are on the remote server in this folder:

**`D:\De ID All\NER_anon_models\`**

It contains one folder per model:

| Folder | What it is |
|---|---|
| `real_only_all_data` | The model you already tested: trained on all the real data only. |
| `real_plus_half_synth_all_data` | **The new model**: the same real data plus synthetic documents (ambiguous words, organisation names, ...). This is the one to test. |

Copy the folder(s) into the repo's `models` folder, keeping the folder names, so that for example this file exists:

```text
models\real_plus_half_synth_all_data\config.json
```

(next to it: `model.safetensors` and the tokenizer files). Copying the old `real_only_all_data` as well lets you compare the two on the same file. If you have an older model folder in `models\` (for example `HeMed_Anonimizer_Model`), leave it there.

With more than one model in `models\`, always give the model name in the command (see the next step).

## Step 3: run

Prepare a UTF-8 CSV with two columns, `id` and `text`, then:

```text
deidentify.bat C:\path\to\your_file.csv label real_plus_half_synth_all_data
```

(`label` is the masking style, see below. Linux/macOS: `./deidentify.sh /path/to/your_file.csv label real_plus_half_synth_all_data`)

The last word of the command is the **name of the model folder** inside `models\`. To try a different model (for example the old one), put its folder in `models\` and change only that name:

```text
deidentify.bat C:\path\to\your_file.csv label real_only_all_data
```

Running the same file with both models and comparing the results is very welcome. If I train an improved model later, I will add it to the same server folder (`D:\De ID All\NER_anon_models\`) and tell you its name: copy it to `models\` the same way, run `git pull`, and use the same command with the new name. Nothing else changes.

The model name is added to the names of the result files, so runs with different models never overwrite each other.

Results are saved next to the input file: a CSV (the original columns plus `de_identified_text` and `identified_entities`) and a JSON file with the exact positions.

While it runs you see a progress bar with the speed and the estimated time left. Large files are safe to interrupt: every finished document is saved to a `*.progress.jsonl` file next to the output. If the run crashes (or you press Ctrl+C), run exactly the same command again and it continues where it stopped. Use `--restart` (with `python -X utf8 inference\de_identify.py --input file.csv --model <name> --restart`) to start from the beginning instead. The progress file is deleted when the run finishes. A document that fails is written as `[PROCESSING_ERROR]` (never left unmasked) and listed at the end; running the same command again retries only those.

The progress file contains the de-identified text and the found entities (which include the original strings), so treat it like the output files and delete it if a run is abandoned.

To try it first on a synthetic example (no real data):

```text
deidentify.bat inference\dummy_data\test.csv label real_plus_half_synth_all_data
```

## Optional

| What | How |
|---|---|
| Masking style: `label` (default, `[FIRST_NAME]`), `mask` (`[REDACTED]`), `x` (`XXXX`) | `deidentify.bat file.csv mask` |
| Compare with the old model on the same file | `deidentify.bat file.csv label real_only_all_data` |
| Lenient mode: do not mask repeated mentions of a name/city/organisation that the model already masked once | `python -X utf8 inference\de_identify.py --input file.csv --model real_plus_half_synth_all_data --no-consistency` |

By default (strict mode) every later mention of a name the model found is masked as well. Words that can be both a name and an ordinary word (for example "אושר" or "דוד") are left to the model alone.

## If something does not work

Run `setup.bat` again: it prints what is missing. If that does not help, send me the full message.

## What I would like back

- Cases where identifying information is still visible. Please describe the type of case only, without the original text.
- Cases where something was masked that should not have been.
- Anything that looks odd in the run or the output.

Thank you!
