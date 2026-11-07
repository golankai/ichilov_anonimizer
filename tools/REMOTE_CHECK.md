# Remote GPU machine: setup check and smoke-test training

Run in PowerShell on the GPU machine, one block at a time. Stages 1-3 only read things; stage 4 trains for one epoch
into a throwaway folder. The two `.txt` reports contain metadata only (names, sizes, counts, hashes), no document text.

## Stage 1: get the code

```powershell
cd "D:\DeID KAI\ichilov_anonimizer"
git pull
git log --oneline -3
```

## Stage 2: machine, GPU, models and data (read-only)

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\machine_info.ps1 -Search -ModelPath "D:\HeMed All\HeMed Final Model\model_non_anon","D:\HeMed All\HeMed Final Model\model_anon" -DataPath "D:\De ID All\NER_anon_dataset\regular","D:\De ID All\NER_anon_dataset\cons"
```

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\scan_models.ps1 -Hash -Root "D:\HeMed All\HeMed Final Model","D:\DeID KAI\ichilov_anonimizer\HeMed_NER_baseline"
```

This writes `machine_info.txt` and `models_scan.txt` in the current folder. Send both back.
(If `HeMed_NER_baseline` does not exist yet, "PATH DOES NOT EXIST" for it is fine.)

## Stage 3: environment and GPU check

```powershell
uv sync --group training
```

```powershell
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

It must print `True` and the GPU name. If `False`, stop and report (usually a CPU-only torch build).

```powershell
uv run --group training pytest training/tests -q
```

## Stage 4: smoke-test training (1 epoch)

```powershell
uv run python training/train.py train --epochs 1 --output-dir models\smoke_test
```

Expected: loads `model_non_anon` and `...\regular`, trains on the GPU (watch `nvidia-smi`), scores validation and test,
writes `models\smoke_test\training_run.json` plus the model files. If out of memory, add `--batch-size 4`.

Send back the last ~30 console lines and `models\smoke_test\training_run.json` (counts only, no text).

## Optional: run the trained model through the de-identification pipeline

```powershell
uv run python inference/de_identify.py --input inference\dummy_data\test.csv --model models\smoke_test
```

After one epoch the quality is poor; this only checks that the model loads and runs end to end.

## Shortcut (no uv, conda env `deid`, one command)

The GPU machine cannot reach download.pytorch.org (hospital certificate), so `uv sync` fails there. Use the script instead;
it runs all the checks above with the `deid` conda env and writes `remote_report.txt` in the repo root:

```powershell
cd "D:\DeID KAI\ichilov_anonimizer"
git pull
powershell -ExecutionPolicy Bypass -File .\tools\remote_check.ps1
```

Add `-Install` to pip-install missing packages, `-Smoke` to also run the tests and a 1-epoch training.
