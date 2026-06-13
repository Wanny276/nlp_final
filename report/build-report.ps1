$ErrorActionPreference = "Stop"

$reportDir = $PSScriptRoot
$buildDir = Join-Path $reportDir "build"
$outputPdf = Join-Path $reportDir "main.pdf"

$invalidHeadingPattern = '\\(part|chapter|paragraph|subparagraph)\*?\{'
$invalidHeadings = Get-ChildItem `
    -LiteralPath (Join-Path $reportDir "sections") `
    -Filter "*.tex" |
    Select-String -Pattern $invalidHeadingPattern

if ($invalidHeadings) {
    $details = $invalidHeadings |
        ForEach-Object { "$($_.Path):$($_.LineNumber): $($_.Line.Trim())" }
    throw "报告标题不得超过三级：`n$($details -join "`n")"
}

Push-Location $reportDir
try {
    & latexmk `
        -xelatex `
        -interaction=nonstopmode `
        -halt-on-error `
        -outdir=build `
        main.tex

    if ($LASTEXITCODE -ne 0) {
        throw "LaTeX compilation failed with exit code $LASTEXITCODE."
    }

    Copy-Item `
        -LiteralPath (Join-Path $buildDir "main.pdf") `
        -Destination $outputPdf `
        -Force

    Write-Host "Updated tracked report: $outputPdf"
}
finally {
    Pop-Location
}
