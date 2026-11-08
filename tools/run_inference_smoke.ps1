<#
.SYNOPSIS
  Simplest inference test on the GPU machine with an existing model (no training).
  Copies the dummy CSVs into the git-ignored output\smoke\ and runs there, so no tracked file is ever modified
  (de_identify.py writes its results next to the input file).
  Log (metadata only) goes to remote_report.txt (git-ignored).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\tools\run_inference_smoke.ps1
  powershell -ExecutionPolicy Bypass -File .\tools\run_inference_smoke.ps1 -Model "D:\HeMed All\HeMed Final Model\model_anon"
#>
param(
    [string]$Model = "",   # empty = models\HeMed_Anonimizer_Model in the repo (the NER-trained model). The D:\HeMed All base models have NO NER head.
    [string]$Py = "C:\ProgramData\anaconda3\envs\PII\python.exe"
)

Set-Location (Split-Path $PSScriptRoot -Parent)
Start-Transcript -Path remote_report.txt -Force | Out-Null

if (-not $Model) { $Model = Join-Path (Get-Location) "models\HeMed_Anonimizer_Model" }
if (-not (Test-Path "$Model\config.json")) {
    Write-Host "NO NER MODEL at $Model. Copy the trained model folder (config.json, model.safetensors, tokenizer files) there, or pass -Model <folder>."
    Stop-Transcript | Out-Null
    exit 1
}

# Only toolz was missing in the PII env (pure python, no dependencies, PyPI works through the proxy).
& $Py -m pip install toolz 2>&1 | ForEach-Object { "$_" } | Out-Host

$out = "output\smoke"
New-Item -ItemType Directory -Force $out | Out-Null
Copy-Item inference\dummy_data\test.csv, inference\dummy_data\test_long.csv $out -Force

Write-Host "Model: $Model (exists: $(Test-Path $Model))"
Write-Host "GPU:"
& $Py -c "import torch; print(torch.__version__, 'cuda:', torch.cuda.is_available())" 2>&1 | ForEach-Object { "$_" } | Out-Host

foreach ($f in "test.csv", "test_long.csv") {
    Write-Host ""
    Write-Host "=============== $f ==============="
    & $Py -X utf8 inference\de_identify.py --input "$out\$f" --model "$Model" --mode label 2>&1 | ForEach-Object { "$_" } | Out-Host
}

Write-Host ""
Write-Host "Results are in $out (dummy data only). Check git status: it should still be clean."
git status --short | Out-Host
Stop-Transcript | Out-Null
