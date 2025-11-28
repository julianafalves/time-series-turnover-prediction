# Remove synthetic data files from the project
$files = @("data/synthetic_turnover.csv", "data/synthetic_turnover_ts.csv")
foreach ($f in $files) {
    if (Test-Path $f) {
        try {
            Remove-Item -Force -Path $f
            Write-Host "Removed $f"
        } catch {
            Write-Host "Could not remove $f — maybe file is locked or permission denied" -ForegroundColor Yellow
        }
    } else {
        Write-Host "Not found: $f"
    }
}
Write-Host "Cleanup complete. Verify data directory contents."