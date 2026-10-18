# Packs the data the server needs but git does not carry (documents, index, extracted texts) into one file,
# to copy to the server once. Run on the Windows development machine from the repository folder:
#   powershell -ExecutionPolicy Bypass -File deploy\veri_paketle.ps1
# Result: darg-veri.tar.gz (~2.5 GB) in the repository folder (git-ignored). Copy it with scp (see KURULUM.md).
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$parts = @(
    "data/raw/documents",                                   # library: the AFAD documents themselves
    "data/indexes/mobile_v5",                               # search index
    "data/processed/structured_v5",                         # library: page texts for offline search
    "data/processed/text_v3/publications/toplanma_alanlari.json"  # assembly-area brochure for local answers
)
foreach ($part in $parts) {
    if (-not (Test-Path $part)) { throw "Eksik: $part" }
}
Write-Host "Paketleniyor (birkaç dakika sürebilir)..."
tar -czf darg-veri.tar.gz @parts
$size = (Get-Item darg-veri.tar.gz).Length / 1GB
Write-Host ("Hazır: darg-veri.tar.gz ({0:N2} GB)" -f $size)
