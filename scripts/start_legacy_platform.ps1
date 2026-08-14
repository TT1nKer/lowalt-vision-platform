$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$entryPoint = Join-Path $projectRoot "app.py"
$config = Join-Path $projectRoot "config.yaml"
$taskName = "LOWALT-Legacy-7860"
$currentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name

$existing = Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -like "*app.py*" -and $_.CommandLine -like "*--port 7860*"
}
foreach ($process in $existing) {
    Stop-Process -Id $process.ProcessId -Force -ErrorAction SilentlyContinue
}

$action = New-ScheduledTaskAction `
    -Execute $python `
    -Argument "-u `"$entryPoint`" --project-dir `"$projectRoot`" --config `"$config`" --host 127.0.0.1 --port 7860" `
    -WorkingDirectory $projectRoot
$principal = New-ScheduledTaskPrincipal -UserId $currentUser -LogonType Interactive
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Days 30) -MultipleInstances IgnoreNew

Register-ScheduledTask -TaskName $taskName -Action $action -Principal $principal -Settings $settings -Force | Out-Null
Start-ScheduledTask -TaskName $taskName
Get-ScheduledTask -TaskName $taskName | Select-Object TaskName, State
