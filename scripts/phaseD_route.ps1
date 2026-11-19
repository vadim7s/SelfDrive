# Phase D round 4: satnav route curve (GPS-grade, bev_route.py) as an extra map layer -> K4 (from K3, same turn weight).
#   powershell -ExecutionPolicy Bypass -File scripts\phaseD_route.ps1
# 1) train K4 (8 epochs, last.pt)  2) Town02 test seeds 123/456 (clean GPS)
# 3) same with GPS 2 m off to the right (must still hold its lane)  4) milestone tour.  CARLA shut down at the end.  ~2.5 h
$ErrorActionPreference = "Continue"
$here = (Resolve-Path "$PSScriptRoot\..\src").Path
$py = "python"
$v2 = "$PSScriptRoot\..\data\vision_data_v2"
$dag = "$PSScriptRoot\..\data\vision_data_v2_dagger"
$k3 = "$PSScriptRoot\..\data\vision_runs\bev_20261118_072014\last.pt"
Set-Location $here
"== train K4  $(Get-Date -Format HH:mm)"
cmd /c "`"$py`" -u bev_train.py --data `"$v2`" --data `"$dag`" --init `"$k3`" --epochs 8 --lr 0.0002 --turn-weight 4 --route > runs\bev_train_K4.log 2>&1"
Select-String -Path runs\bev_train_K4.log -Pattern "episodes|epoch|student:" | ForEach-Object { $_.Line }
$m = Select-String -Path runs\bev_train_K4.log -Pattern "^student: (.*)$"
if (-not $m) { "TRAINING FAILED - see runs\bev_train_K4.log"; exit 1 }
$k4 = Join-Path (Split-Path $m.Matches[0].Groups[1].Value) "last.pt"
"K4 = $k4"
& powershell -ExecutionPolicy Bypass -File "$here\phaseD_test.ps1" -Student $k4 -Tag K4
& powershell -ExecutionPolicy Bypass -File "$here\phaseD_test.ps1" -Student $k4 -Tag K4_gps2m -Gps "2,0"
"== tour clip K4  $(Get-Date -Format HH:mm)"
& $py -u record_tour.py --student $k4 --label "Student K4"
& $py -c "import carla_server as s; s.shutdown_if_configured(); print('CARLA running:', bool(s.processes_running()))"
"== done  $(Get-Date -Format HH:mm)"
