# Espera a rodada principal terminar (linha FIM no log) e entao roda o benchmark TF x PyTorch.
$root = "D:\dev\tcc\tcc-agrupamento-series"
$logs = Join-Path $root "results\aeon_dl\logs"
$out = Join-Path $root "results\bench_frameworks"
New-Item -ItemType Directory -Force $out | Out-Null
while ($true) {
    $last = Get-ChildItem "$logs\rodada_*.log" | Sort-Object LastWriteTime | Select-Object -Last 1
    if ($last -and (Select-String -Path $last.FullName -Pattern "FIM" -Quiet)) { break }
    Start-Sleep -Seconds 60
}
Add-Content -Encoding utf8 "$out\cadeia.log" "[$(Get-Date -Format s)] rodada terminou; iniciando benchmark"
cmd.exe /c "wsl.exe -d Ubuntu-20.04 -- bash /mnt/d/dev/tcc/tcc-agrupamento-series/experimentos/bench/run_bench.sh > `"$out\bench_run.log`" 2>&1"
Add-Content -Encoding utf8 "$out\cadeia.log" "[$(Get-Date -Format s)] benchmark terminou (codigo $LASTEXITCODE)"
