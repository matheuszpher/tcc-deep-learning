# Rodada completa em duas fases:
#   1) 5 modelos na GPU (WSL2), 2) AEDRNN em CPU (Windows nativo; nao cabe em 4 GB de GPU).
# Uso:
#   powershell -File experimentos\rodada.ps1 -Epochs 5 -Out results/aeon_dl_teste -Datasets "Coffee Beef GunPoint"
#   powershell -File experimentos\rodada.ps1 -Epochs 200 -Out results/aeon_dl -GpuHours 7 -CpuHours 2
param(
    [int]$Epochs = 200,
    [string]$Out = "results/aeon_dl",
    [string]$Datasets = "",          # nomes separados por espaco ou virgula
    [double]$GpuHours = 0,
    [double]$CpuHours = 0,
    [string]$KMode = "both",
    [int]$MaxSize = 0
)
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$logDir = Join-Path $root "$Out/logs"
New-Item -ItemType Directory -Force $logDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$main = "$logDir/rodada_$stamp.log"
function Log($msg) { Add-Content -Encoding utf8 -Path $main -Value "[$(Get-Date -Format s)] $msg" }

$common = "--epochs $Epochs --out $Out --k-mode $KMode"
$names = ($Datasets -split '[,\s]+') | Where-Object { $_ }
if ($names.Count -gt 0) { $common += " --datasets " + ($names -join ' ') }
if ($MaxSize -gt 0) { $common += " --max-size $MaxSize" }

$gpu = "$common --models AEFCNClusterer AEResNetClusterer AEDCNNClusterer AEBiGRUClusterer AEAttentionBiGRUClusterer"
if ($GpuHours -gt 0) { $gpu += " --stop-after-hours $GpuHours" }
Log "fase GPU: $gpu"
cmd.exe /c "wsl.exe -d Ubuntu-20.04 -- bash /mnt/d/dev/tcc/tcc-agrupamento-series/experimentos/run_wsl.sh $gpu > `"$logDir\gpu_$stamp.log`" 2>&1"
Log "fase GPU terminou (codigo $LASTEXITCODE)"

$cpu = "$common --models AEDRNNClusterer --device cpu"
if ($CpuHours -gt 0) { $cpu += " --stop-after-hours $CpuHours" }
Log "fase CPU: $cpu"
$env:TF_CPP_MIN_LOG_LEVEL = "2"
cmd.exe /c "`"$root\.venv\Scripts\python.exe`" -u experimentos\aeon_dl_baseline.py $cpu > `"$logDir\cpu_$stamp.log`" 2>&1"
Log "fase CPU terminou (codigo $LASTEXITCODE)"
Log "FIM"
