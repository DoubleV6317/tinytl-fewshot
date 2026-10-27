$ErrorActionPreference = 'Stop'
throw 'Disabled: user switched to a direct 200-epoch run. Do not launch another experiment.'
$taskRoot = $PSScriptRoot
$baseline = Join-Path $taskRoot 'runs\train-tinytl-lite_residual+bias-seed0-20260910-181021'
$done = Join-Path $baseline 'verification.json'
if (-not (Test-Path -LiteralPath $done)) { throw 'The original 50-epoch run has not completed successfully.' }
$report = Get-Content -LiteralPath $done -Raw | ConvertFrom-Json
if ($report.status -ne 'passed' -or $report.mode -ne 'train') { throw 'Baseline verification failed.' }
$baselineArgs = Get-Content -LiteralPath (Join-Path $baseline 'args.txt') -Raw | ConvertFrom-Json
if ($baselineArgs.n_epochs -ne 50) { throw 'Unexpected baseline epoch count.' }
$lockPath = Join-Path $taskRoot 'extension-200.lock'
# CreateNew prevents multiple monitor runs from launching duplicate GPU jobs.
$lock = [System.IO.File]::Open($lockPath, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
try {
    $python = Join-Path $taskRoot '.venv\Scripts\python.exe'
    $job = Start-Process -FilePath $python -ArgumentList '-u run.py --mode train --epochs 200 --seed 0' -WorkingDirectory $taskRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskRoot 'extension-200.stdout.log') -RedirectStandardError (Join-Path $taskRoot 'extension-200.stderr.log') -PassThru
    @{ status='launched'; pid=$job.Id; launched_at=(Get-Date).ToString('o'); epochs=200; baseline=$baseline; protocol='Fresh pretrained initialization, same seed and hyperparameters, cosine schedule over 200 epochs. Exploratory test-set comparison; report final epoch, not best test-selected epoch.' } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskRoot 'extension-200.json') -Encoding UTF8
    Write-Output ('Started 200-epoch experiment. PID: ' + $job.Id)
} finally {
    $lock.Dispose()
}
