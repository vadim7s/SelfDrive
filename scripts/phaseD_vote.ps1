# Phase D round 6: K6 = K5's recipe (route curve twice, from scratch) on maps built with the EVIDENCE-VOTING memory
# (VoteFuser: slow fade + stickiness; real-road map flicker 3.3% -> 0.9%, CARLA accuracy unchanged).
#   powershell -ExecutionPolicy Bypass -File scripts\phaseD_vote.ps1
# 1) vote map caches (bev_vote.npz, old caches kept)  2) train K6 (20 epochs, best.pt)  3) route reliance (offline)
# 4) Town02 test seeds 123/456  5) GPS 2 m test  6) milestone tour  7) real Tesla clip (open loop, no CARLA)
# CARLA shut down at the end.  ~5-6 h
$ErrorActionPreference = "Continue"
$here = (Resolve-Path "$PSScriptRoot\..\src").Path
$py = "python"
$v2 = "$PSScriptRoot\..\data\vision_data_v2"
$dag = "$PSScriptRoot\..\data\vision_data_v2_dagger"
Set-Location $here
"== vote map caches  $(Get-Date -Format HH:mm)"
& $py -u bev_dataset.py $v2 vote
& $py -u bev_dataset.py $dag vote
"== train K6  $(Get-Date -Format HH:mm)"
cmd /c "`"$py`" -u bev_train.py --data `"$v2`" --data `"$dag`" --epochs 20 --lr 0.0003 --turn-weight 4 --route --fuser vote > runs\bev_train_K6.log 2>&1"
Select-String -Path runs\bev_train_K6.log -Pattern "map cache|episodes|epoch|student:" | ForEach-Object { $_.Line }
$m = Select-String -Path runs\bev_train_K6.log -Pattern "^student: (.*)$"
if (-not $m) { "TRAINING FAILED - see runs\bev_train_K6.log"; exit 1 }
$k6 = $m.Matches[0].Groups[1].Value
"K6 = $k6"
"== route reliance  $(Get-Date -Format HH:mm)"
cmd /c "`"$py`" -u route_reliance.py `"$k6`" > runs\route_reliance_K6.log 2>&1"
Get-Content runs\route_reliance_K6.log | Select-String -NotMatch "Warning" | ForEach-Object { $_.Line }
& powershell -ExecutionPolicy Bypass -File "$here\phaseD_test.ps1" -Student $k6 -Tag K6
& powershell -ExecutionPolicy Bypass -File "$here\phaseD_test.ps1" -Student $k6 -Tag K6_gps2m -Gps "2,0"
"== tour clip K6  $(Get-Date -Format HH:mm)"
& $py -u record_tour.py --student $k6 --label "Student K6"
& $py -c "import carla_server as s; s.shutdown_if_configured(); print('CARLA running:', bool(s.processes_running()))"
"== real Tesla clip  $(Get-Date -Format HH:mm)"
cmd /c "`"$py`" -u tesla_bev_test.py --student `"$k6`" --name K6_bev > runs\tesla_bev_K6.log 2>&1"
Get-Content runs\tesla_bev_K6.log | Select-String -NotMatch "Warning|warn" | Select-Object -Last 3 | ForEach-Object { $_.Line }
"== done  $(Get-Date -Format HH:mm)"
