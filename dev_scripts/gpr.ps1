git pull --rebase origin main
if ($LASTEXITCODE -eq 0) {
    git push
} else {
    Write-Error "git pull --rebase failed; aborting git push."
    exit $LASTEXITCODE
}

