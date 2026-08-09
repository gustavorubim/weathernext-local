[CmdletBinding()]
param(
    [string]$InstallLocation = "$env:LOCALAPPDATA\WeatherNextWSL"
)

$ErrorActionPreference = "Stop"
$distro = "WeatherNext"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

function Convert-WslOutputToText {
    param([object[]]$Output)

    return ((@($Output) -join "`n") -replace "`0", "").Trim()
}

function New-WslUpdateMessage {
    param(
        [string]$Reason,
        [string]$DetectedVersion = "unavailable"
    )

    return @"
$Reason
Detected WSL version: $DetectedVersion.

From an Administrator PowerShell, update the Microsoft WSL package:
  wsl.exe --update

Restart Windows if requested, reopen PowerShell, and rerun .\scripts\setup.ps1.
If the update reports that WSL is not installed, run
  wsl.exe --install --no-distribution
restart Windows, run 'wsl.exe --update', and then rerun this setup command.
"@
}

$wslCommand = Get-Command wsl.exe -CommandType Application -ErrorAction SilentlyContinue
if (-not $wslCommand) {
    throw (New-WslUpdateMessage -Reason `
        "wsl.exe was not found. Install current Windows updates if the install command is also unavailable.")
}

function Assert-WslInstallCapabilities {
    $wslVersionOutput = @(& wsl.exe --version 2>$null)
    $wslVersionExitCode = $LASTEXITCODE
    $wslVersionText = Convert-WslOutputToText $wslVersionOutput
    $wslVersionMatch = [regex]::Match(
        $wslVersionText,
        '(?m)(?:^|\s)(\d+\.\d+\.\d+(?:\.\d+)?)(?:\s|$)'
    )
    if ($wslVersionExitCode -ne 0 -or -not $wslVersionMatch.Success) {
        throw (New-WslUpdateMessage -Reason `
            "The installed WSL does not expose a modern package version; legacy inbox WSL cannot create this isolated distribution.")
    }
    $wslVersion = $wslVersionMatch.Groups[1].Value

    # These switches are part of the modern distro install interface. Check the
    # install section specifically: --name also appears under unrelated commands.
    $wslHelpText = Convert-WslOutputToText @(& wsl.exe --help 2>$null)
    $installHelpMatch = [regex]::Match(
        $wslHelpText,
        '(?s)--install\b(?<body>.*?)(?=\r?\n\s*--manage\b)'
    )
    $installHelp = if ($installHelpMatch.Success) { $installHelpMatch.Value } else { "" }
    $requiredInstallOptions = @("--name", "--location", "--vhd-size")
    $missingInstallOptions = @(
        $requiredInstallOptions | Where-Object { -not $installHelp.Contains($_) }
    )
    if ($missingInstallOptions.Count -gt 0) {
        $missingText = $missingInstallOptions -join ", "
        throw (New-WslUpdateMessage -DetectedVersion $wslVersion -Reason `
            "This setup requires wsl.exe --install support for $missingText, but the installed WSL help does not advertise those options.")
    }
}

$registeredText = ((& wsl.exe --list --quiet 2>$null) -join "`n") -replace "`0", ""
$registered = @(
    $registeredText -split "`r?`n" |
        ForEach-Object { $_.Trim() } |
        Where-Object { $_ }
)

if ($registered -notcontains $distro) {
    Assert-WslInstallCapabilities
    Write-Host "Installing an isolated Ubuntu WSL2 distribution named $distro..."
    & wsl.exe --install Ubuntu-24.04 `
        --name $distro `
        --location $InstallLocation `
        --version 2 `
        --vhd-size 64GB `
        --no-launch `
        --web-download
    if ($LASTEXITCODE -ne 0) {
        throw "WSL installation failed with exit code $LASTEXITCODE."
    }
}

$probeOutput = @(& wsl.exe -d $distro -u root -- bash -lc `
    'source /etc/os-release; printf "%s|%s|%s" "$ID" "$VERSION_ID" "$(uname -r)"')
$probeExitCode = $LASTEXITCODE
if ($probeExitCode -ne 0) {
    throw "Could not start and inspect the $distro WSL distribution. If WSL was just enabled, restart Windows and rerun this setup command."
}
$probe = (($probeOutput -join "`n") -replace "`0", "").Trim()
$probeParts = $probe -split '\|', 3
if ($probeParts.Count -ne 3 -or $probeParts[0] -ne "ubuntu" -or `
    $probeParts[1] -ne "24.04" -or $probeParts[2] -notmatch "WSL2") {
    throw "The existing $distro registration is not the required Ubuntu 24.04 WSL2 runtime: $probe"
}

Write-Host "Installing the pinned Python/JAX environment inside $distro..."
& wsl.exe -d $distro -u root --cd $repoRoot -- bash scripts/bootstrap_wsl.sh
if ($LASTEXITCODE -ne 0) {
    throw "WeatherNext environment setup failed with exit code $LASTEXITCODE."
}

Write-Host "Setup complete. Run: .\scripts\wnx.cmd doctor"
