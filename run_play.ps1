Push-Location $PSScriptRoot
try {
    & python scripts/rsl_rl/play.py --num_envs 1 @args
    $playExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $playExitCode
