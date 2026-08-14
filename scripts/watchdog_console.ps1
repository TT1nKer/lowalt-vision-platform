$ErrorActionPreference = 'SilentlyContinue'
$log = '<PROJECT_ROOT>\lowalt\console_watchdog.log'
$python = '<PROJECT_ROOT>\lowalt\.venv\Scripts\python.exe'
$args = @('-u', '<PROJECT_ROOT>\lowalt\app.py', '--project-dir', '<PROJECT_ROOT>\lowalt', '--config', '<PROJECT_ROOT>\lowalt\config.yaml', '--host', '127.0.0.1', '--port', '7860')
$workdir = '<PROJECT_ROOT>\lowalt'

function Say($message) {
    $stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    Add-Content -Path $log -Value "$stamp $message" -Encoding UTF8
}

function ConsoleUp {
    try {
        $r = Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:7860/api/health' -TimeoutSec 5
        return ($r.StatusCode -ge 200 -and $r.StatusCode -lt 500)
    } catch { return $false }
}

if (ConsoleUp) { Say 'console already healthy, nothing to do'; exit 0 }

for ($attempt = 1; $attempt -le 3; $attempt++) {
    $existing = Get-CimInstance Win32_Process -Filter "name='python.exe'" | Where-Object { $_.CommandLine -like '*app.py*--port*7860*' }
    if ($existing) {
        Say "attempt ${attempt}: stale console python found, killing PIDs $($existing.ProcessId -join ',')"
        $existing | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
        Start-Sleep -Seconds 5
    }
    Say "attempt ${attempt}: starting console"
    Start-Process -FilePath $python -ArgumentList $args -WorkingDirectory $workdir -WindowStyle Hidden
    for ($wait = 0; $wait -lt 12; $wait++) {
        Start-Sleep -Seconds 5
        if (ConsoleUp) { Say "attempt ${attempt}: console healthy after $($wait*5+5)s"; exit 0 }
    }
    Say "attempt ${attempt}: console did not become healthy in time"
}
Say 'all attempts failed; console not running'
exit 1
