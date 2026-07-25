$ErrorActionPreference = "Stop"

$ProjectDir = $PSScriptRoot
$VenvDir = Join-Path $ProjectDir ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

function Write-Status([string] $Message) {
    [Console]::Error.WriteLine($Message)
}

function Invoke-QuietCommand([string] $FilePath, [string[]] $Arguments) {
    $PreviousErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $FilePath @Arguments *> $null
        return $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $PreviousErrorAction
    }
}

function Invoke-SetupCommand([string] $FilePath, [string[]] $Arguments) {
    $PreviousErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $FilePath @Arguments 2>&1 | ForEach-Object {
            [Console]::Error.WriteLine($_.ToString())
        }
        return $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $PreviousErrorAction
    }
}

$NeedsSetup = -not (Test-Path $VenvPython)
if (-not $NeedsSetup) {
    $ProbeExitCode = Invoke-QuietCommand $VenvPython (
        @("-c", "import mcp, vc3d_mcp")
    )
    $NeedsSetup = $ProbeExitCode -ne 0
    if ($NeedsSetup) {
        Write-Status "[vc3d-mcp] Existing $VenvDir is incomplete; rebuilding ..."
        Remove-Item -LiteralPath $VenvDir -Recurse -Force
    }
}

if ($NeedsSetup) {
    Write-Status "[vc3d-mcp] Setting up Python environment in $VenvDir ..."

    if ($env:PYTHON) {
        $Bootstrap = $env:PYTHON
        $BootstrapArgs = @()
    } elseif (Get-Command py -ErrorAction SilentlyContinue) {
        $Bootstrap = "py"
        $BootstrapArgs = @("-3")
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        $Bootstrap = "python"
        $BootstrapArgs = @()
    } else {
        throw "Python 3.10+ was not found. Install Python or set PYTHON to python.exe."
    }

    $SetupExitCode = Invoke-SetupCommand $Bootstrap (
        $BootstrapArgs + @("-m", "venv", $VenvDir)
    )
    if ($SetupExitCode -ne 0) {
        exit $SetupExitCode
    }

    $SetupExitCode = Invoke-SetupCommand $VenvPython (
        @("-m", "pip", "install", "-q", "-e", $ProjectDir)
    )
    if ($SetupExitCode -ne 0) {
        exit $SetupExitCode
    }
    Write-Status "[vc3d-mcp] Setup complete."
}

& $VenvPython -m vc3d_mcp @args
exit $LASTEXITCODE
