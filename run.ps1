[CmdletBinding()]
param(
    [switch]$SkipBuild,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $projectRoot "venv\Scripts\python.exe"
$requirements = Join-Path $projectRoot "requirements.txt"
$frontendRoot = Join-Path $projectRoot "frontend"
$runtimeRoot = Join-Path $projectRoot ".run"
$backendLog = Join-Path $runtimeRoot "backend.log"
$backendErrorLog = Join-Path $runtimeRoot "backend-error.log"
$frontendLog = Join-Path $runtimeRoot "frontend.log"
$frontendErrorLog = Join-Path $runtimeRoot "frontend-error.log"
$backendUrl = "http://127.0.0.1:8000"
$frontendUrl = "http://127.0.0.1:5173"
$backendProcess = $null
$frontendProcess = $null

function Write-Step([string]$Message) {
    Write-Host "`n==> $Message" -ForegroundColor Green
}

function Refresh-ProcessPath {
    $machinePath = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = "$machinePath;$userPath"
}

function Resolve-Command([string]$Name) {
    return Get-Command $Name -ErrorAction SilentlyContinue
}

function Wait-ForUrl([string]$Url, [string]$Name, [int]$Attempts = 120) {
    for ($attempt = 1; $attempt -le $Attempts; $attempt++) {
        try {
            $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 2
            if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) {
                Write-Host "$Name is ready: $Url" -ForegroundColor Cyan
                return
            }
        } catch {
            Start-Sleep -Milliseconds 500
        }
    }
    throw "$Name did not become ready. Check its log in $runtimeRoot."
}

function Stop-ChildProcess($Process, [string]$Name) {
    if ($null -ne $Process -and -not $Process.HasExited) {
        Write-Host "Stopping $Name..." -ForegroundColor DarkGray
        Stop-Process -Id $Process.Id -Force -ErrorAction SilentlyContinue
        $Process.WaitForExit(5000) | Out-Null
    }
}

Set-Location -LiteralPath $projectRoot
New-Item -ItemType Directory -Path $runtimeRoot -Force | Out-Null

try {
    Write-Step "Checking Python environment"
    if (-not (Test-Path -LiteralPath $venvPython)) {
        $systemPython = Resolve-Command "python"
        if ($null -eq $systemPython) {
            throw "Python is not installed or is not available in PATH."
        }
        & $systemPython.Source -m venv (Join-Path $projectRoot "venv")
    }
    & $venvPython -m pip install --disable-pip-version-check -r $requirements

    Write-Step "Checking Node.js LTS"
    $nodeCommand = Resolve-Command "node"
    $npmCommand = Resolve-Command "npm.cmd"
    if ($null -eq $nodeCommand -or $null -eq $npmCommand) {
        $wingetCommand = Resolve-Command "winget"
        if ($null -eq $wingetCommand) {
            throw "Node.js is missing and winget is unavailable. Install Node.js LTS, then run ./run again."
        }
        Write-Host "Node.js is missing. Installing the official LTS package..." -ForegroundColor Yellow
        & $wingetCommand.Source install --id OpenJS.NodeJS.LTS --exact --accept-package-agreements --accept-source-agreements
        if ($LASTEXITCODE -ne 0) {
            throw "Node.js LTS installation failed with exit code $LASTEXITCODE."
        }
        Refresh-ProcessPath
        $nodeCommand = Resolve-Command "node"
        $npmCommand = Resolve-Command "npm.cmd"
        if ($null -eq $nodeCommand -or $null -eq $npmCommand) {
            throw "Node.js was installed but is not visible yet. Reopen the terminal and run ./run again."
        }
    }
    Write-Host "Node: $(& $nodeCommand.Source --version)"
    Write-Host "npm:  $(& $npmCommand.Source --version)"

    Write-Step "Preparing React frontend"
    $exampleEnv = Join-Path $frontendRoot ".env.example"
    $localEnv = Join-Path $frontendRoot ".env.local"
    if (-not (Test-Path -LiteralPath $localEnv)) {
        Copy-Item -LiteralPath $exampleEnv -Destination $localEnv
    }
    Push-Location -LiteralPath $frontendRoot
    try {
        & $npmCommand.Source install
        if ($LASTEXITCODE -ne 0) { throw "npm install failed." }
        if (-not $SkipBuild) {
            Write-Step "Type-checking and building React"
            & $npmCommand.Source run build
            if ($LASTEXITCODE -ne 0) { throw "npm run build failed." }
        }
    } finally {
        Pop-Location
    }

    Write-Step "Starting FastAPI and React"
    Remove-Item -LiteralPath $backendLog -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $backendErrorLog -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $frontendLog -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $frontendErrorLog -Force -ErrorAction SilentlyContinue
    $backendProcess = Start-Process -FilePath $venvPython `
        -ArgumentList "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8000" `
        -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $backendLog -RedirectStandardError $backendErrorLog
    $viteScript = Join-Path $frontendRoot "node_modules\vite\bin\vite.js"
    $quotedViteScript = '"' + $viteScript + '"'
    $frontendProcess = Start-Process -FilePath $nodeCommand.Source `
        -ArgumentList $quotedViteScript, "--host", "127.0.0.1", "--port", "5173" `
        -WorkingDirectory $frontendRoot -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $frontendLog -RedirectStandardError $frontendErrorLog

    Wait-ForUrl "$backendUrl/api/health" "FastAPI"
    Wait-ForUrl $frontendUrl "React"

    if (-not $NoBrowser) {
        Start-Process $frontendUrl
    }
    Write-Host "`nCorporate Brain is running." -ForegroundColor Green
    Write-Host "Frontend: $frontendUrl"
    Write-Host "Backend:  $backendUrl"
    Write-Host "Logs:     $runtimeRoot"
    Write-Host "Press Ctrl+C to stop both servers.`n" -ForegroundColor Yellow

    while (-not $backendProcess.HasExited -and -not $frontendProcess.HasExited) {
        Start-Sleep -Seconds 1
    }
    if ($backendProcess.HasExited) { throw "FastAPI stopped unexpectedly. Check $backendLog." }
    if ($frontendProcess.HasExited) { throw "React stopped unexpectedly. Check $frontendLog." }
} finally {
    Stop-ChildProcess $frontendProcess "React"
    Stop-ChildProcess $backendProcess "FastAPI"
}
