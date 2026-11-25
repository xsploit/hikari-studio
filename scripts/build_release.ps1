<#
.SYNOPSIS
  Build a self-contained, zero-Python-required Windows 64-bit Pro Release of Hikari Studio.
.OUTPUTS
  dist\HikariStudio-v<Version>-win64\
  dist\HikariStudio-v<Version>-win64.zip
  dist\SHA256SUMS.txt
#>
param(
  [string]$Version = "1.0.0",
  [string]$SdlDll = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$DistRoot = Join-Path $RepoRoot "dist"
$BuildRoot = Join-Path $RepoRoot "build"
$StageName = "HikariStudio-v$Version-win64"
$StageDir = Join-Path $DistRoot $StageName
$ZipPath = Join-Path $DistRoot "$StageName.zip"

Write-Host "=== Building Hikari Studio v$Version (Win64 Pro Release) ===" -ForegroundColor Cyan

# 1. Locate SDL2.dll
if (-not $SdlDll) {
  foreach ($cand in @(
    $env:HIKARI_SDL_DLL,
    (Join-Path $RepoRoot "engine\SDL2.dll"),
    "D:\Hikari-Rig-Tools\Inochi-Creator-0.8.6\SDL2.dll"
  )) {
    if ($cand -and (Test-Path $cand)) { $SdlDll = $cand; break }
  }
}
if (-not $SdlDll -or -not (Test-Path $SdlDll)) {
  throw "SDL2.dll not found. Pass -SdlDll or place SDL2.dll in engine\SDL2.dll."
}

# 2. Compile standalone Python backend (hikari-server\hikari-server.exe) via PyInstaller --onedir
Write-Host "[1/4] Compiling standalone backend (hikari-server.exe)..." -ForegroundColor Yellow
Push-Location $RepoRoot
try {
  $pyArgs = @(
    "-3.13", "-m", "PyInstaller",
    "--onedir",
    "--noconfirm",
    "--clean",
    "--name", "hikari-server",
    "--distpath", (Join-Path $BuildRoot "pydist"),
    "--workpath", (Join-Path $BuildRoot "pywork"),
    "--specpath", $BuildRoot,
    "--collect-submodules", "core",
    "--collect-submodules", "modules",
    "--collect-submodules", "scipy.spatial",
    "--collect-all", "_sounddevice_data",
    "--collect-all", "cv2",
    "--collect-data", "certifi",
    "--hidden-import", "sounddevice",
    "--hidden-import", "aiohttp",
    "--hidden-import", "httpx",
    "--hidden-import", "msgpack",
    "--hidden-import", "mss",
    "--hidden-import", "PIL.ImageGrab",
    "--hidden-import", "scipy.ndimage",
    "--hidden-import", "scipy.signal",
    "--exclude-module", "torch",
    "--exclude-module", "torchvision",
    "--exclude-module", "torchaudio",
    "--exclude-module", "ultralytics",
    "--exclude-module", "tensorflow",
    "--exclude-module", "matplotlib",
    "--exclude-module", "tkinter",
    "--exclude-module", "PyQt5",
    "--exclude-module", "PySide6",
    "--exclude-module", "IPython",
    "--exclude-module", "pytest",
    "server.py"
  )
  & py @pyArgs
  if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed with exit code $LASTEXITCODE" }
} finally {
  Pop-Location
}

# 3. Compile native Rust + Tauri v2 desktop app (Hikari Studio.exe)
Write-Host "[2/4] Compiling Rust + Tauri v2 desktop app..." -ForegroundColor Yellow
Push-Location (Join-Path $RepoRoot "desktop-tauri\src-tauri")
try {
  cargo build --release
  if ($LASTEXITCODE -ne 0) { throw "cargo build --release failed with exit code $LASTEXITCODE" }
} finally {
  Pop-Location
}

# 4. Assemble clean self-contained release folder (zero secrets, zero caches, zero sleep_lab)
Write-Host "[3/4] Assembling release bundle at $StageDir..." -ForegroundColor Yellow
if (Test-Path $StageDir) { Remove-Item -Recurse -Force $StageDir }
New-Item -ItemType Directory -Force -Path $StageDir | Out-Null

# Desktop exe
$TauriExe = Join-Path $RepoRoot "desktop-tauri\src-tauri\target\release\hikari-desktop.exe"
if (-not (Test-Path $TauriExe)) { throw "Missing compiled $TauriExe" }
Copy-Item $TauriExe (Join-Path $StageDir "Hikari Studio.exe")

# Compiled backend sidecar
Copy-Item -Recurse (Join-Path $BuildRoot "pydist\hikari-server") (Join-Path $StageDir "hikari-server")

# Native DLang GPU engine + SDL2.dll
$EngDst = Join-Path $StageDir "engine"
New-Item -ItemType Directory -Force -Path $EngDst | Out-Null
Copy-Item (Join-Path $RepoRoot "engine\hikari-live-engine.exe") $EngDst
Copy-Item $SdlDll (Join-Path $EngDst "SDL2.dll")

# Base tracked HD puppet (rebuilt into Hikari-v28-Custom.inp on first launch)
$ModDst = Join-Path $StageDir "models"
New-Item -ItemType Directory -Force -Path $ModDst | Out-Null
Copy-Item (Join-Path $RepoRoot "models\Hikari-v28-BaseHD-v5.inp") $ModDst

# Frontend UI, plugins, and clean assets (exclude transient caches)
Copy-Item -Recurse (Join-Path $RepoRoot "web") (Join-Path $StageDir "web")
$PlugDst = Join-Path $StageDir "plugins"
New-Item -ItemType Directory -Force -Path $PlugDst | Out-Null
Get-ChildItem (Join-Path $RepoRoot "plugins") -Filter "*.py" | Copy-Item -Destination $PlugDst

$ToolsDst = Join-Path $StageDir "tools"
New-Item -ItemType Directory -Force -Path $ToolsDst | Out-Null
Copy-Item (Join-Path $RepoRoot "tools\build_midori_deform_arms.py") $ToolsDst

$AssetDst = Join-Path $StageDir "assets"
robocopy (Join-Path $RepoRoot "assets") $AssetDst /E /XD _vfx_cache _midori_cache __pycache__ /NFL /NDL /NJH /NJS /NC /NS | Out-Null
if ($LASTEXITCODE -gt 7) { throw "robocopy assets failed ($LASTEXITCODE)" }

# Clean empty runtime dirs + docs + license + .env.example
New-Item -ItemType Directory -Force -Path (Join-Path $StageDir "motions") | Out-Null
foreach ($f in @("README.md", "LICENSE", "THIRD_PARTY_NOTICES.md", ".env.example")) {
  Copy-Item (Join-Path $RepoRoot $f) $StageDir
}

# 5. Zip + SHA-256 checksums
Write-Host "[4/4] Packing $ZipPath and SHA256SUMS.txt..." -ForegroundColor Yellow
if (Test-Path $ZipPath) { Remove-Item -Force $ZipPath }
Compress-Archive -Path $StageDir -DestinationPath $ZipPath -CompressionLevel Optimal
$hash = (Get-FileHash -Algorithm SHA256 $ZipPath).Hash.ToLower()
"$hash  $StageName.zip" | Set-Content -Encoding UTF8 (Join-Path $DistRoot "SHA256SUMS.txt")

$zipMb = [math]::Round((Get-Item $ZipPath).Length / 1MB, 2)
Write-Host "RELEASE READY: $ZipPath ($zipMb MB) | SHA256: $hash" -ForegroundColor Green
