Push-Location $PSScriptRoot
try {
    & python scripts/rsl_rl/train.py --num_envs 1024 --headless @args
    $trainExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $trainExitCode
