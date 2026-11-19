# Where does a student leave the route?  Runs traced test drives, then summarises every off-route / collision ending.
#   powershell -ExecutionPolicy Bypass -File scripts\diag_student.ps1 -Tag I
# Shuts CARLA down at the end.
param(
    [string]$Tag = "I",
    [string]$Student = "$PSScriptRoot\..\data\vision_runs\student_20261116_131620\best.pt",
    [string]$Runs = "Town02:123,Town02:456,Town03:123,Town05:123",
    [string]$Teacher = "$PSScriptRoot\..\src\runs\ppo_20261115_100321\final_model.zip",
    [int]$Episodes = 10,
    [switch]$V2                       # top-down student: real-car camera at 640x480
)
$ErrorActionPreference = "Continue"
$here = (Resolve-Path "$PSScriptRoot\..\src").Path
$py = "python"
Set-Location $here
$traces = @()
foreach ($r in $Runs.Split(",")) {
    $town, $seed = $r.Split(":")
    $log = "runs\diag_student_$($Tag)_$($town)_$seed.log"
    $tr = "runs\trace_student_$($Tag)_$($town)_$seed.csv"
    "== $town seed $seed  $(Get-Date -Format HH:mm)"
    cmd /c "`"$py`" -u vision_collect.py $(if ($V2) { '--v2' }) --driver student --steady --teacher `"$Teacher`" --student `"$Student`" --town $town --episodes $Episodes --no-save --no-jitter --seed $seed --weather-set day --trace $tr > $log 2>&1"
    Select-String -Path $log -Pattern "^summary|mean route" | ForEach-Object { $_.Line }
    $traces += "$tr=$log"
}
& $py -u diag_student_report.py @traces
& $py -c "import carla_server as s; s.shutdown_if_configured(); print('CARLA running:', bool(s.processes_running()))"
"== done  $(Get-Date -Format HH:mm)"
