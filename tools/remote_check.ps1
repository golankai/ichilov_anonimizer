<#
.SYNOPSIS
  One-shot check for the GPU machine, using the conda env "deid" (no uv needed).
  Writes everything to remote_report.txt in the repo root (metadata only, no document text).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\tools\remote_check.ps1            # read-only checks
  powershell -ExecutionPolicy Bypass -File .\tools\remote_check.ps1 -Install   # + pip install missing packages
  powershell -ExecutionPolicy Bypass -File .\tools\remote_check.ps1 -TorchCuda # + replace CPU-only torch by a CUDA build
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

# Run a native command and send ALL its output (stdout + stderr) through the host,
# so that the transcript records it (it does not capture raw native output).
function Run {
    $exe = $args[0]
    $rest = @()
    if ($args.Count -gt 1) { $rest = $args[1..($args.Count - 1)] }
    & $exe @rest 2>&1 | ForEach-Object { "$_" } | Out-Host
}

Section "Repo"
Run git log --oneline -3
Write-Host "Python: $Py (exists: $(Test-Path $Py))"

Section "Machine, models, data (machine_info.ps1)"
Run powershell -ExecutionPolicy Bypass -File .\tools\machine_info.ps1 -ModelPath "D:\HeMed All\HeMed Final Model\model_non_anon,D:\HeMed All\HeMed Final Model\model_anon" -DataPath "D:\De ID All\NER_anon_dataset\regular,D:\De ID All\NER_anon_dataset\cons"

Section "Model folders (scan_models.ps1)"
Run powershell -ExecutionPolicy Bypass -File .\tools\scan_models.ps1 -Hash -Root "D:\HeMed All\HeMed Final Model,D:\DeID KAI\ichilov_anonimizer\HeMed_NER_baseline"

Section "GPU driver"
Run nvidia-smi

Section "Python packages in env deid"
Run $Py --version
& $Py -m pip list 2>&1 | ForEach-Object { "$_" } | Select-String -Pattern "^(torch|transformers|accelerate|datasets|seqeval|scikit|pandas|nltk|tokenizers|tqdm|pytest|numpy)" | Out-Host

Section "PyTorch / CUDA (before any install)"
Run $Py -c "import torch; print('torch', torch.__version__, '| built for CUDA:', torch.version.cuda, '| cuda available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU VISIBLE TO TORCH')"

if ($Install) {
    Section "pip install (missing training packages)"
    Run $Py -m pip install transformers accelerate datasets seqeval scikit-learn scikit-multilearn pandas nltk tqdm pytest
}

if ($TorchCuda) {
    Section "pip install torch (CUDA 12.4 build; the driver supports CUDA 12.5)"
    Run $Py -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu124 --trusted-host download.pytorch.org
    Section "PyTorch / CUDA (after install)"
    Run $Py -c "import torch; print('torch', torch.__version__, '| built for CUDA:', torch.version.cuda, '| cuda available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU VISIBLE TO TORCH')"
}

if ($Smoke) {
    Section "Tests"
    Run $Py -m pytest training/tests -q
    Section "Smoke training (1 epoch)"
    Run $Py -X utf8 training\train.py train --epochs 1 --output-dir models\smoke_test
    if (Test-Path models\smoke_test\training_run.json) {
        Section "training_run.json"
        Get-Content models\smoke_test\training_run.json | Out-Host
    }
}

Stop-Transcript | Out-Null
Write-Host ""
Write-Host "Done. Full report: $((Get-Location).Path)\remote_report.txt"
