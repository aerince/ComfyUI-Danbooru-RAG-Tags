# Local preparation only unless the user explicitly supplies -Publish.
param(
    [switch]$Publish,
    [switch]$Public,
    [string]$Repository = 'aerince/ComfyUI-Danbooru-RAG-Tags'
)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$archive = Join-Path $root 'output/ComfyUI-Danbooru-RAG-Tags-v1.0.0-full.zip'
$visibility = if ($Public) { '--public' } else { '--private' }
if (-not $Publish) {
    Write-Host "Ready: $Repository ($visibility), tag v1.0.0"
    Write-Host "Release asset: $archive"
    Write-Host 'Run this script with -Publish to create/push the repo and upload the release.'
    Write-Host 'Add -Public only when public visibility is intended.'
    exit 0
}
if (-not (Test-Path $archive -PathType Leaf)) { throw "Missing full package: $archive" }
Push-Location $root
try {
    $dirty = & git status --porcelain
    if ($LASTEXITCODE -ne 0) { throw 'Initialize and commit this standalone repository first.' }
    if ($dirty) { throw 'Commit the standalone source changes before publishing.' }
    & gh auth status
    if ($LASTEXITCODE -ne 0) { throw 'GitHub CLI authentication is required.' }
    $existing = & gh repo view $Repository --json name 2>$null
    if ($LASTEXITCODE -ne 0) {
        & gh repo create $Repository $visibility --source . --remote origin --push
        if ($LASTEXITCODE -ne 0) { throw 'Repository creation/push failed.' }
    } else {
        & git push "https://github.com/$Repository.git" HEAD:main
        if ($LASTEXITCODE -ne 0) { throw 'Source push failed.' }
    }
    & gh release create v1.0.0 $archive --repo $Repository --target main --title 'v1.0.0' --notes-file docs/RELEASE.md
    if ($LASTEXITCODE -ne 0) { throw 'Release upload failed. Existing releases are not overwritten automatically.' }
    Write-Host "Published: https://github.com/$Repository/releases/tag/v1.0.0"
} finally {
    Pop-Location
}
