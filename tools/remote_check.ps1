<#
.SYNOPSIS
  One-shot check for the GPU machine, using the conda env "deid" (no uv needed).
  Writes everything to remote_report.txt in the repo root (metadata only, no document text).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\tools\remote_check.ps1            # read-only checks
  powershell -ExecutionPolicy Bypass -File .\tools\remote_check.ps1 -Install   # + pip install missing packages
  powershell -ExecutionPolicy Bypass -File .\tools\remote_check.ps1 -Smoke     # + tests and 1-epoch training
#>
param(
    [switch]$Install,
    [switch]$TorchCuda,
    [switch]$Smoke,
    [string]$Py = "C:\ProgramData\anaconda3\envs\deid\python.exe"
)

Set-Location (Split-Path $PSScriptRoot -Parent)
Start-Transcript -Path remote_report.txt -Force | Out-Null

function Section($t) { Write-Host ""; Write-Host "=============== $t ===============" }

Section "Repo"
git log --oneline -3
Write-Host "Python: $Py (exists: $(Test-Path $Py))"

Section "Machine, models, data (machine_info.ps1)"
powershell -ExecutionPolicy Bypass -File .\tools\machine_info.ps1 -Search `
    -ModelPath "D:\HeMed All\HeMed Final Model\model_non_anon,D:\HeMed All\HeMed Final Model\model_anon" `
    -DataPath "D:\De ID All\NER_anon_dataset\regular,D:\De ID All\NER_anon_dataset\cons" | Out-Null
if (Test-Path machine_info.txt) { Get-Content machine_info.txt }

Section "Model folders (scan_models.ps1)"
powershell -ExecutionPolicy Bypass -File .\tools\scan_models.ps1 -Hash `
    -Root "D:\HeMed All\HeMed Final Model,D:\DeID KAI\ichilov_anonimizer\HeMed_NER_baseline" | Out-Null
if (Test-Path models_scan.txt) { Get-Content models_scan.txt }

Section "GPU driver"
nvidia-smi

Section "Python packages in env deid"
& $Py --version
& $Py -m pip list 2>$null | Select-String -Pattern "^(torch|transformers|accelerate|datasets|seqeval|scikit|pandas|nltk|tokenizers|tqdm|pytest|numpy)"

if ($Install) {
    Section "pip install (missing training packages)"
    & $Py -m pip install transformers accelerate datasets seqeval scikit-learn scikit-multilearn pandas nltk tqdm pytest
}

if ($TorchCuda) {
    Section "pip install torch (CUDA 12.4 build; the driver supports CUDA 12.5)"
    & $Py -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu124 --trusted-host download.pytorch.org
}

Section "PyTorch / CUDA"
& $Py -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU VISIBLE TO TORCH')"

if ($Smoke) {
    Section "Tests"
    & $Py -m pytest training/tests -q
    Section "Smoke training (1 epoch)"
    & $Py -X utf8 training\train.py train --epochs 1 --output-dir models\smoke_test
    if (Test-Path models\smoke_test\training_run.json) {
        Section "training_run.json"
        Get-Content models\smoke_test\training_run.json
    }
}

Stop-Transcript | Out-Null
Write-Host ""
Write-Host "Done. Full report: $((Get-Location).Path)\remote_report.txt"
