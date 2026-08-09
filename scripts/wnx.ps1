[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$WnxArguments
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$wnx = "/opt/weathernext-local/venv/bin/wnx"
$distro = if ($env:WNX_WSL_DISTRO) { $env:WNX_WSL_DISTRO } else { "WeatherNext" }
$wslArguments = @(
    $WnxArguments | ForEach-Object { $_.Replace('\', '\\') }
)

& wsl.exe -d $distro -u weathernext --cd $repoRoot -- `
    env "WNX_PROJECT_ROOT=." "XDG_CACHE_HOME=/opt/weathernext-local/cache" `
    $wnx @wslArguments

exit $LASTEXITCODE
