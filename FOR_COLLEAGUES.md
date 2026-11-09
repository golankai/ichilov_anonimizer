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

## Step 2: place the model

1. Copy the model folder from here (remote server):

   **`<<< PATH ON THE SERVER: to be filled in before sending >>>`**

2. Put it inside the repo's `models` folder so that this file exists:

   ```text
   models\HeMed_Anonimizer_Model\config.json
   ```

   Next to it should be `model.safetensors` and the tokenizer files.

If you already have an older model called `HeMed_Anonimizer_Model`, do not delete it: rename it first (for example to `HeMed_old`). You can keep several models, each in its own folder, and compare them (see below).

## Step 3: run

Prepare a UTF-8 CSV with two columns, `id` and `text`, then:

```text
deidentify.bat C:\path\to\your_file.csv
```

(Linux/macOS: `./deidentify.sh /path/to/your_file.csv`)

Results are saved next to the input file: a CSV (the original columns plus `de_identified_text` and `identified_entities`) and a JSON file with the exact positions.

While it runs you see a progress bar with the speed and the estimated time left. Large files are safe to interrupt: every finished document is saved to a `*.progress.jsonl` file next to the output. If the run crashes (or you press Ctrl+C), run exactly the same command again and it continues where it stopped. Use `--restart` (with `python -X utf8 inference\de_identify.py`) to start from the beginning instead. The progress file is deleted when the run finishes. A document that fails is written as `[PROCESSING_ERROR]` (never left unmasked) and listed at the end; running the same command again retries only those.

The progress file contains the de-identified text and the found entities (which include the original strings), so treat it like the output files and delete it if a run is abandoned.

To try it first on a synthetic example (no real data):

```text
deidentify.bat inference\dummy_data\test.csv
```

## Optional

| What | How |
|---|---|
| Masking style: `label` (default, `[FIRST_NAME]`), `mask` (`[REDACTED]`), `x` (`XXXX`) | `deidentify.bat file.csv mask` |
| Run another model from `models\` and compare. The model name is added to the output file names | `deidentify.bat file.csv label HeMed_old` |
| Lenient mode: do not mask repeated mentions of a name/city/organisation that the model already masked once | `python -X utf8 inference\de_identify.py --input file.csv --no-consistency` |

By default (strict mode) every later mention of a name the model found is masked as well. Words that can be both a name and an ordinary word (for example "אושר" or "דוד") are left to the model alone.

## If something does not work

Run `setup.bat` again: it prints what is missing. If that does not help, send me the full message.

## What I would like back

- Cases where identifying information is still visible. Please describe the type of case only, without the original text.
- Cases where something was masked that should not have been.
- Anything that looks odd in the run or the output.

Thank you!
