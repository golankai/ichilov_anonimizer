<#
.SYNOPSIS
  Collects hardware/software info and checks the local Hebrew model (and optionally the data)
  so it can be pasted back to Claude. Prints metadata only - NEVER any patient text.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\machine_info.ps1 -Search `
     -ModelPath "D:\HeMed All\HeMed Final Model\model_anon,D:\HeMed All\HeMed Final Model\model_non_anon" `
     -DataPath "D:\De ID All\NER_anon_dataset\regular,D:\De ID All\NER_anon_dataset\cons"

  Writes machine_info.txt in the current folder (and prints it). Paste that file's content back.
#>
param(
    [string[]]$ModelPath = @(),
    [string[]]$DataPath = @(),
    [switch]$Search
)

# `powershell -File` passes "a,b" as ONE string, so split on commas ourselves.
$ModelPath = @($ModelPath | ForEach-Object { $_ -split ',' } | Where-Object { $_.Trim() } | ForEach-Object { $_.Trim() })
$DataPath = @($DataPath | ForEach-Object { $_ -split ',' } | Where-Object { $_.Trim() } | ForEach-Object { $_.Trim() })

$ErrorActionPreference = "Continue"
$out = New-Object System.Collections.Generic.List[string]
function Add($line) { $out.Add([string]$line) }
function Section($t) { Add ""; Add "=== $t ===" }
function Try-Run($block) { try { & $block } catch { Add "  (failed: $($_.Exception.Message))" } }

Add "machine_info generated $(Get-Date -Format 's')"

Section "OS"
Try-Run {
    $os = Get-CimInstance Win32_OperatingSystem
    Add "  $($os.Caption) $($os.Version) ($($os.OSArchitecture))"
}

Section "CPU / RAM"
Try-Run {
    Get-CimInstance Win32_Processor | ForEach-Object {
        Add "  $($_.Name.Trim()) | cores=$($_.NumberOfCores) logical=$($_.NumberOfLogicalProcessors)"
    }
    $cs = Get-CimInstance Win32_ComputerSystem
    Add "  RAM: $([math]::Round($cs.TotalPhysicalMemory / 1GB, 1)) GB"
}

Section "GPU"
Try-Run {
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version,compute_cap --format=csv | ForEach-Object { Add "  $_" }
        $smi = (nvidia-smi | Select-String "CUDA Version") -replace '.*(CUDA Version: [0-9.]+).*', '$1'
        Add "  nvidia-smi reports $smi"
    } else {
        Add "  nvidia-smi not found. Video controllers:"
        Get-CimInstance Win32_VideoController | ForEach-Object { Add "  $($_.Name) | driver $($_.DriverVersion)" }
    }
}

Section "Disk (free space per drive)"
Try-Run {
    Get-PSDrive -PSProvider FileSystem | Where-Object { $_.Used -ne $null } | ForEach-Object {
        Add "  $($_.Name): free $([math]::Round($_.Free / 1GB, 1)) GB / total $([math]::Round(($_.Used + $_.Free) / 1GB, 1)) GB"
    }
}

Section "Tools"
foreach ($cmd in @("python", "uv", "git")) {
    Try-Run {
        $c = Get-Command $cmd -ErrorAction SilentlyContinue
        if ($c) { Add "  $cmd : $(& $cmd --version 2>&1) [$($c.Source)]" } else { Add "  $cmd : NOT FOUND" }
    }
}

Section "Python ML stack (repo .venv if present, else system python)"
Try-Run {
    $repoRoot = Split-Path -Parent $PSScriptRoot
    $venvPy = Join-Path $repoRoot ".venv\Scripts\python.exe"
    $py = if (Test-Path $venvPy) { $venvPy } else { "python" }
    Add "  using: $py"
    $code = @'
import importlib, sys
print("  python", sys.version.split()[0])
for m in ["torch","transformers","datasets","accelerate","seqeval","sklearn","skmultilearn","tokenizers","nltk"]:
    try:
        mod = importlib.import_module(m); print(f"  {m}", getattr(mod, "__version__", "?"))
    except Exception as e:
        print(f"  {m} MISSING")
try:
    import torch
    print("  torch.cuda.is_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"  cuda device 0: {p.name}, {p.total_memory/1024**3:.1f} GB, torch built for CUDA {torch.version.cuda}")
        print("  bf16 supported:", torch.cuda.is_bf16_supported())
except Exception:
    pass
'@
    $tmp = Join-Path $env:TEMP "machine_info_probe.py"
    Set-Content -LiteralPath $tmp -Value $code -Encoding UTF8
    & $py $tmp 2>&1 | ForEach-Object { Add $_ }
    Remove-Item -LiteralPath $tmp -ErrorAction SilentlyContinue
}

foreach ($mp in $ModelPath) {
    Section "Model: $mp"
    Try-Run {
        if (-not (Test-Path -LiteralPath $mp)) { Add "  PATH DOES NOT EXIST"; return }
        $abs = (Resolve-Path -LiteralPath $mp).Path
        Add "  absolute path: $abs"
        $items = Get-ChildItem -LiteralPath $abs -Force
        $total = ($items | Where-Object { -not $_.PSIsContainer } | Measure-Object Length -Sum).Sum
        Add "  total size: $([math]::Round($total / 1MB, 1)) MB"
        $items | ForEach-Object {
            $size = if ($_.PSIsContainer) { "<dir>" } else { "$([math]::Round($_.Length / 1MB, 2)) MB" }
            Add "    $($_.Name)  $size"
        }
        $cfgPath = Join-Path $abs "config.json"
        if (Test-Path -LiteralPath $cfgPath) {
            $cfg = Get-Content -LiteralPath $cfgPath -Raw -Encoding UTF8 | ConvertFrom-Json
            Add "  config.json: model_type=$($cfg.model_type) architectures=$($cfg.architectures -join ',')"
            Add "    hidden_size=$($cfg.hidden_size) layers=$($cfg.num_hidden_layers) heads=$($cfg.num_attention_heads) vocab_size=$($cfg.vocab_size) max_position_embeddings=$($cfg.max_position_embeddings)"
            $labels = @()
            if ($cfg.id2label) { $labels = $cfg.id2label.PSObject.Properties | ForEach-Object { $_.Value } }
            Add "    id2label entries: $($labels.Count)"
            if ($labels.Count -gt 2) { Add "    labels: $($labels -join ', ')" }
            Add "    transformers_version in config: $($cfg.transformers_version)"
        } else { Add "  NO config.json (not a Hugging Face model dir?)" }
        $hasTok = (Test-Path (Join-Path $abs "tokenizer.json")) -or (Test-Path (Join-Path $abs "vocab.txt")) -or (Test-Path (Join-Path $abs "tokenizer.model"))
        $hasWeights = (Test-Path (Join-Path $abs "model.safetensors")) -or (Test-Path (Join-Path $abs "pytorch_model.bin"))
        Add "  tokenizer files present: $hasTok | weights present: $hasWeights"
    }
}

function Count-Records($file) {
    $n = 0
    foreach ($line in [System.IO.File]::ReadLines($file)) { if ($line.Trim()) { $n++ } }
    return $n
}

foreach ($dp in $DataPath) {
    Section "Data: $dp"
    Try-Run {
        if (-not (Test-Path -LiteralPath $dp)) { Add "  PATH DOES NOT EXIST"; return }
        $f = Get-Item -LiteralPath $dp
        Add "  absolute path: $($f.FullName)"
        Add "  size: $([math]::Round($f.Length / 1MB, 2)) MB"
        if ($f.PSIsContainer) {
            Get-ChildItem -LiteralPath $f.FullName | ForEach-Object {
                $extra = ""
                if (-not $_.PSIsContainer -and $_.Extension -in @('.jsonl', '.json')) { $extra = "  lines=$(Count-Records $_.FullName)" }
                $kind = if ($_.PSIsContainer) { "<dir>" } else { "$([math]::Round($_.Length/1MB,2)) MB" }
                Add "    $($_.Name)  $kind$extra"
            }
        } else {
            $n = 0
            $first = $null
            foreach ($line in [System.IO.File]::ReadLines($f.FullName)) {
                if ($line.Trim()) { $n++; if (-not $first) { $first = $line } }
            }
            Add "  non-empty lines (records): $n"
            if ($first -and $f.Extension -in @('.jsonl', '.json')) {
                $rec = $first | ConvertFrom-Json
                Add "  keys of first record (values NOT shown): $(($rec.PSObject.Properties.Name) -join ', ')"
                if ($rec.label) { Add "  first record has $(@($rec.label).Count) labeled spans; span keys: $((@($rec.label)[0].PSObject.Properties.Name) -join ', ')" }
            }
        }
    }
}

if ($Search) {
    Section "Search: .json/.jsonl files > 100 KB on fixed drives (path + size only)"
    Try-Run {
        Get-PSDrive -PSProvider FileSystem | Where-Object { $_.Root -match '^[A-Z]:\$' } | ForEach-Object {
            Get-ChildItem -Path $_.Root -Recurse -Include *.jsonl, *.json -File -ErrorAction SilentlyContinue |
                Where-Object { $_.Length -gt 100KB -and $_.FullName -notmatch '\(Windows|Program Files|AppData|node_modules|site-packages|\.venv|\.cache)\' } |
                ForEach-Object { Add "  $($_.FullName)  $([math]::Round($_.Length/1MB,2)) MB" }
        }
    }
}

$text = $out -join "`r`n"
$text | Out-File -FilePath "machine_info.txt" -Encoding utf8
$text
Write-Host ""
Write-Host "Saved to $((Get-Location).Path)\machine_info.txt - paste its content back. (No patient text is included.)"
