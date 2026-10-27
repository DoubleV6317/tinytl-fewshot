# Runs the paper's Table 2 baselines serially (one GPU job at a time).
# Prerequisite: sleep must be disabled, this takes about 5 hours.
#   powercfg /change standby-timeout-ac 0
#   powercfg /change hibernate-timeout-ac 0
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$py = Join-Path $root '.venv\Scripts\python.exe'

$standby = (powercfg /query SCHEME_CURRENT SUB_SLEEP STANDBYIDLE | Select-String 'Current AC Power Setting Index:\s+0x(\w+)').Matches.Groups[1].Value
if ([Convert]::ToInt32($standby, 16) -ne 0) {
    throw "System still sleeps after $([Convert]::ToInt32($standby,16)) s on AC. Run 'powercfg /change standby-timeout-ac 0' first, or this queue will be killed mid-run."
}

if (Get-Process python -ErrorAction SilentlyContinue) {
    throw 'A python process is already running. Do not start a second GPU job.'
}

# Paper Table 2 order, cheapest first so a config error surfaces early.
$methods = @('last', 'tinytl-bias', 'bn+last', 'tinytl-lite_residual', 'full')

foreach ($m in $methods) {
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    $slug = $m -replace '[^a-zA-Z0-9_-]', '_'
    $out = Join-Path $root "baseline-$slug.stdout.log"
    $err = Join-Path $root "baseline-$slug.stderr.log"
    Write-Output "[$stamp] starting $m"
    & $py -u run.py --mode train --epochs 50 --seed 0 --method $m 1> $out 2> $err
    if ($LASTEXITCODE -ne 0) {
        throw "Method $m exited with code $LASTEXITCODE. See $err. Later methods were not started."
    }
    Write-Output "[$(Get-Date -Format 'HH:mm:ss')] finished $m"
    & $py summarize.py
}
Write-Output 'All baselines finished.'
