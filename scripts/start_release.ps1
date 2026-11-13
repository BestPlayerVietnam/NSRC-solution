$ErrorActionPreference = 'Continue'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding $false
$OutputEncoding = [Console]::OutputEncoding

$releaseRoot = Split-Path -Parent $PSScriptRoot
$releaseExitCode = 1
$releaseError = 'Could not start the release launcher.'

try {
    Set-Location -LiteralPath $releaseRoot -ErrorAction Stop
    $releaseError = 'Install Docker Desktop with Docker Compose, then start Docker Desktop.'
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        throw $releaseError
    }
    & docker info --format '{{.ServerVersion}}' 1>$null 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw $releaseError
    }

    $environmentFile = Join-Path $releaseRoot '.env'
    if (-not (Test-Path -LiteralPath $environmentFile) -and
        ([string]::IsNullOrWhiteSpace($env:LLM_BASE_URL) -or
         [string]::IsNullOrWhiteSpace($env:LLM_API_KEY))) {
        $releaseError = 'Could not create .env from .env.example.'
        Copy-Item -LiteralPath (Join-Path $releaseRoot '.env.example') -Destination $environmentFile -ErrorAction Stop
        $releaseError = 'Created .env. Set TASKS_DIR and LLM_API_KEY there, then open start.cmd again.'
        throw $releaseError
    }

    $composeFile = Join-Path $releaseRoot 'compose.yaml'
    $releaseError = 'Check Docker Compose and the LLM_BASE_URL / LLM_API_KEY values in .env or the environment.'
    # Keep the resolved configuration in memory because it contains credentials.
    $configurationText = & docker compose --file $composeFile config --format json 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw $releaseError
    }
    $configuration = ($configurationText -join "`n") | ConvertFrom-Json -ErrorAction Stop
    $taskMount = @($configuration.services.agent.volumes | Where-Object { $_.target -eq '/data/tasks' })
    if ($taskMount.Count -ne 1) {
        throw $releaseError
    }
    $taskDirectory = $taskMount[0].source
    $releaseError = 'TASKS_DIR must point to an existing directory containing task folders with task.json.'
    if (-not (Test-Path -LiteralPath $taskDirectory -PathType Container)) {
        throw $releaseError
    }
    $hasTasks = Get-ChildItem -LiteralPath $taskDirectory -Directory -ErrorAction Stop | Where-Object {
        Test-Path -LiteralPath (Join-Path $_.FullName 'task.json') -PathType Leaf
    } | Select-Object -First 1
    if (-not $hasTasks) {
        throw $releaseError
    }

    $releaseError = 'The run failed. Check the Docker output above.'
    & docker compose --file $composeFile run --build --rm agent
    $releaseExitCode = $LASTEXITCODE
}
catch {
    Write-Host $releaseError -ForegroundColor Red
}

exit $releaseExitCode
