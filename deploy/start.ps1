# Production-style stack (adjust COMPOSE_FILE if needed)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Error "Docker is not installed or not on PATH."
}

if (-not (Test-Path "$Root\.env")) {
    if (Test-Path "$Root\.env.example") {
        Copy-Item "$Root\.env.example" "$Root\.env"
        Write-Host "Created .env from .env.example — edit before production."
    } else {
        Write-Error "No .env in project root."
    }
}

$compose = if ($env:COMPOSE_FILE) { $env:COMPOSE_FILE } else { "docker-compose.prod.yaml" }
Write-Host "Using compose file: $compose"
docker compose -f $compose --env-file "$Root\.env" up -d --build @args
