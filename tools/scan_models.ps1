<#
.SYNOPSIS
  Read-only inventory of trained model folders (anything containing a config.json) so we can tell
  which model is which. Prints metadata only; nothing is modified. With -Hash it also computes a
  SHA-256 of each weights file, so identical copies (e.g. HeMed_Anonimizer_Model vs one of the folders in
  models\) show up as "IDENTICAL".

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\scan_models.ps1 -Hash `
     -Root "D:\DeID NOA\DeIdentification\ICHI\models","D:\DeID KAI\ichilov_anonimizer\HeMed_Anonimizer_Model"

  Writes models_scan.txt in the current folder (and prints it). Paste its content back.
#>
param(
    [string[]]$Root = @("D:\DeID NOA\DeIdentification\ICHI\models"),
    [switch]$Hash
)

# `powershell -File` passes "a,b" as ONE string, so split on commas ourselves.
$Root = @($Root | ForEach-Object { $_ -split ',' } | Where-Object { $_.Trim() } | ForEach-Object { $_.Trim() })

$out = New-Object System.Collections.Generic.List[string]
function Add($line) { $out.Add([string]$line) }
$hashes = @{}

function Summarize-TrainerState($file) {
    try {
        $st = Get-Content -LiteralPath $file -Raw -Encoding UTF8 | ConvertFrom-Json
        $evals = @($st.log_history | Where-Object { $null -ne $_.eval_f1 })
        $f1s = ($evals | ForEach-Object { "{0:N3}" -f [double]$_.eval_f1 }) -join " "
        Add "    trainer_state: epochs=$($st.epoch) steps=$($st.global_step) best_metric=$($st.best_metric)"
        Add "    best_checkpoint: $($st.best_model_checkpoint)"
        if ($f1s) { Add "    eval_f1 per epoch: $f1s" }
    } catch { Add "    trainer_state: unreadable ($($_.Exception.Message))" }
}

foreach ($r in $Root) {
    Add ""
    Add "##### ROOT: $r"
    if (-not (Test-Path -LiteralPath $r)) { Add "  PATH DOES NOT EXIST"; continue }

    $configs = @(Get-ChildItem -LiteralPath $r -Recurse -Filter config.json -File -ErrorAction SilentlyContinue |
        Where-Object { $_.DirectoryName -notmatch '[\\/]checkpoint-\d+$' } | Sort-Object DirectoryName)
    if (-not $configs) {
        # the root itself may be a model folder without nested ones, or contain none
        Add "  no config.json found under this root"
    }
    foreach ($c in $configs) {
        $dir = $c.DirectoryName
        Add ""
        Add "  MODEL: $dir"
        try {
            $cfg = Get-Content -LiteralPath $c.FullName -Raw -Encoding UTF8 | ConvertFrom-Json
            $labels = @()
            if ($cfg.id2label) { $labels = @($cfg.id2label.PSObject.Properties | ForEach-Object { $_.Value }) }
            $bs = ($labels | Where-Object { $_ -like '*\*' }).Count -gt 0
            Add "    type=$($cfg.model_type) vocab=$($cfg.vocab_size) labels=$($labels.Count) backslash_in_labels=$bs transformers=$($cfg.transformers_version)"
            if ($labels.Count -gt 2) {
                $ents = $labels | ForEach-Object { $_ -replace '^[BI]-', '' } | Where-Object { $_ -ne 'O' } | Sort-Object -Unique
                Add "    entity types ($($ents.Count)): $($ents -join ', ')"
            } else { Add "    (no NER head: looks like a base model)" }
        } catch { Add "    config.json unreadable: $($_.Exception.Message)" }

        $w = @("model.safetensors", "pytorch_model.bin") | ForEach-Object { Join-Path $dir $_ } | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
        if ($w) {
            $wi = Get-Item -LiteralPath $w
            Add "    weights: $($wi.Name) $([math]::Round($wi.Length / 1MB, 1)) MB, modified $($wi.LastWriteTime.ToString('yyyy-MM-dd HH:mm'))"
            if ($Hash) {
                $h = (Get-FileHash -LiteralPath $w -Algorithm SHA256).Hash.Substring(0, 16)
                Add "    sha256[0..16]: $h"
                if (-not $hashes.ContainsKey($h)) { $hashes[$h] = @() }
                $hashes[$h] += $dir
            }
        } else { Add "    weights: none in this folder" }

        $ts = Join-Path $dir "trainer_state.json"
        if (-not (Test-Path -LiteralPath $ts)) {
            $ck = Get-ChildItem -LiteralPath $dir -Directory -Filter "checkpoint-*" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime | Select-Object -Last 1
            if ($ck) { $ts = Join-Path $ck.FullName "trainer_state.json" }
        }
        if (Test-Path -LiteralPath $ts) { Summarize-TrainerState $ts } else { Add "    trainer_state: none found" }
        $files = (Get-ChildItem -LiteralPath $dir -File | Select-Object -ExpandProperty Name) -join ", "
        Add "    files: $files"
    }
}

if ($Hash) {
    Add ""
    Add "##### DUPLICATE CHECK (same weights hash)"
    $dups = $hashes.GetEnumerator() | Where-Object { $_.Value.Count -gt 1 }
    if ($dups) { foreach ($d in $dups) { Add "  IDENTICAL: $($d.Value -join '  ==  ')" } } else { Add "  no two scanned models share identical weights" }
}

$text = $out -join "`r`n"
$text | Out-File -FilePath "models_scan.txt" -Encoding utf8
$text
Write-Host ""
Write-Host "Saved to $((Get-Location).Path)\models_scan.txt - paste its content back. (Metadata only.)"
