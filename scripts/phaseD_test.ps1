# Phase D test: the top-down student drives Town02 (unseen) with traffic, seeds 123 + 456, 10 routes each
# (same routes as the camera-only students I / J in rounds 7-8).  Real-car camera (--v2), no jitter, day weather.
#   powershell -ExecutionPolicy Bypass -File scripts\phaseD_test.ps1 -Student <best.pt> -Tag K
# CARLA is shut down at the end.  ~1 h (the segmenter runs live, ~0.1 s per step).
# -Gps "2,0": fixed GPS error for a route-curve student (curve drawn 2 m right) - robustness test.
param([Parameter(Mandatory = $true)][string]$Student, [string]$Tag = "K", [string]$Gps = "")
$ErrorActionPreference = "Continue"
$here = (Resolve-Path "$PSScriptRoot\..\src").Path
$py = "python"
Set-Location $here
foreach ($seed in 123, 456) {
    "== test $Tag Town02 traffic seed $seed  $(Get-Date -Format HH:mm)"
    $log = "runs\eval_student_town02_phaseD_$($Tag)_$seed.log"
    $extra = if ($Gps) { "--gps-offset $Gps" } else { "" }
    cmd /c "`"$py`" -u vision_collect.py --v2 --driver student --student `"$Student`" --town Town02 --episodes 10 --no-save --no-jitter --seed $seed --weather-set day $extra > $log 2>&1"
    Select-String -Path $log -Pattern "^episode|summary|mean route|smoothness|Error" | ForEach-Object { $_.Line }
}
& $py -c "import carla_server as s; s.shutdown_if_configured(); print('CARLA running:', bool(s.processes_running()))"
"== done  $(Get-Date -Format HH:mm)"
