# Teacher round 5: traffic in every town + ideal-line reward.
#   powershell -ExecutionPolicy Bypass -File scripts\teacher_traffic.ps1      (add -Resume to continue)
# 1) milestone tour clip of the current teacher (v4 speed + lane tracker steering), 2 routes per town, with traffic
# 2) PPO fine-tune v4 -> v5 in all towns WITH traffic and the new reward (~600k steps, ~7+ h; supervisor restarts CARLA)
# 3) score v5 (own PPO steering) and v5track (lane tracker steering) in every town, 10 routes, traffic on
# 4) tour clip of v5track.  CARLA is shut down at the end.
param([switch]$Resume, [switch]$SkipFirstClip)
$ErrorActionPreference = "Continue"
$here = (Resolve-Path "$PSScriptRoot\..\src").Path
$py = "python"
$v4 = "$PSScriptRoot\..\src\runs\ppo_20261114_093825\final_model.zip"
Set-Location $here
if (-not $Resume -and -not $SkipFirstClip) {
    "== tour clip teacher v4 + lane tracker  $(Get-Date -Format HH:mm)"
    & $py -u record_tour.py --model $v4 --steady --label "Teacher v4 lane tracker traffic"
}
if ($Resume) {
    "== fine-tune RESUMED  $(Get-Date -Format HH:mm)"
    cmd /c "`"$py`" -u run_training.py >> runs\teacher_traffic.log 2>&1"
} else {
    Remove-Item -ErrorAction SilentlyContinue runs\train_town_state.json
    "== fine-tune v4 -> v5 (traffic, ideal-line reward)  $(Get-Date -Format HH:mm)"
    cmd /c "`"$py`" -u run_training.py --finetune > runs\teacher_traffic.log 2>&1"
}
$run = (Get-ChildItem runs -Directory -Filter "ppo_*" | Sort-Object Name | Select-Object -Last 1).FullName
"run: $run"
Get-Content runs\carla_events.log -Tail 400 | Select-String "town switch|SKIPPING|FAILURE|FROZEN|restart" | Select-Object -Last 15 | ForEach-Object { $_.Line }
foreach ($town in "Town01", "Town02", "Town03", "Town04", "Town05", "Town10HD") {
    foreach ($pair in @(@("v5", ""), @("v5track", "--steady"))) {
        "== teacher $($pair[0]) $town  $(Get-Date -Format HH:mm)"
        $log = "runs\eval_teacher_$($pair[0])_$town.log"
        cmd /c "`"$py`" -u vision_collect.py --driver teacher $($pair[1]) --teacher `"$run\final_model.zip`" --town $town --episodes 10 --no-save --no-jitter --no-noise --seed 123 --weather-set day --trace runs\trace_teacher_$($pair[0])_$town.csv > $log 2>&1"
        Select-String -Path $log -Pattern "summary|mean route|smoothness" | ForEach-Object { $_.Line }
    }
}
$tr = Get-ChildItem runs\trace_teacher_v5*_Town*.csv | ForEach-Object { $_.FullName }
& $py turn_report.py $tr
& $py steady_report.py $tr
"== tour clip v5 + lane tracker  $(Get-Date -Format HH:mm)"
& $py -u record_tour.py --model "$run\final_model.zip" --steady --label "Teacher v5 lane tracker traffic"
& $py -c "import carla_server as s; s.shutdown_if_configured(); print('CARLA running:', bool(s.processes_running()))"
"== done  $(Get-Date -Format HH:mm)"
