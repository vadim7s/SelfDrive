# Phase D round 5: K4 ignored the route curve (thin map line only; fine-tuned from a model that had memorised its data).
# K5 = FROM SCRATCH with the curve twice: map layer + 8 route points as numbers into the memory and steering heads.
#   powershell -ExecutionPolicy Bypass -File scripts\phaseD_route2.ps1
# 1) train K5 (20 epochs, best.pt)  2) route reliance check (offline)  3) Town02 test seeds 123/456
# 4) same with GPS 2 m off to the right  5) milestone tour.  CARLA shut down at the end.  ~4 h
$ErrorActionPreference = "Continue"
$here = (Resolve-Path "$PSScriptRoot\..\src").Path
$py = "python"
$v2 = "$PSScriptRoot\..\data\vision_data_v2"
$dag = "$PSScriptRoot\..\data\vision_data_v2_dagger"
Set-Location $here
"== train K5  $(Get-Date -Format HH:mm)"
cmd /c "`"$py`" -u bev_train.py --data `"$v2`" --data `"$dag`" --epochs 20 --lr 0.0003 --turn-weight 4 --route > runs\bev_train_K5.log 2>&1"
Select-String -Path runs\bev_train_K5.log -Pattern "episodes|epoch|student:" | ForEach-Object { $_.Line }
$m = Select-String -Path runs\bev_train_K5.log -Pattern "^student: (.*)$"
if (-not $m) { "TRAINING FAILED - see runs\bev_train_K5.log"; exit 1 }
$k5 = $m.Matches[0].Groups[1].Value
"K5 = $k5"
"== route reliance  $(Get-Date -Format HH:mm)"
cmd /c "`"$py`" -u route_reliance.py `"$k5`" > runs\route_reliance_K5.log 2>&1"
Get-Content runs\route_reliance_K5.log | Select-String -NotMatch "Warning" | ForEach-Object { $_.Line }
& powershell -ExecutionPolicy Bypass -File "$here\phaseD_test.ps1" -Student $k5 -Tag K5
& powershell -ExecutionPolicy Bypass -File "$here\phaseD_test.ps1" -Student $k5 -Tag K5_gps2m -Gps "2,0"
"== tour clip K5  $(Get-Date -Format HH:mm)"
& $py -u record_tour.py --student $k5 --label "Student K5"
& $py -c "import carla_server as s; s.shutdown_if_configured(); print('CARLA running:', bool(s.processes_running()))"
"== done  $(Get-Date -Format HH:mm)"
