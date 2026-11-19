# Phase D round 2: diagnose student K, then DAgger with K driving, then train K2 and test it.
#   powershell -ExecutionPolicy Bypass -File scripts\phaseD_dagger.ps1
# Re-running continues (collection counts frames on disk; caches skip finished episodes).
# 1) diagnosis: traced drives Town02 seeds 123 + 456 and Town03 seed 123 -> where/why K leaves the route or crashes
# 2) DAgger: K drives, the teacher drives 20% of steps and takes over beyond 1 half-lane; every frame is labelled by
#    the teacher; K's own segmentation is saved with each frame. 10k frames per town -> vision_data_v2_dagger
# 3) top-down cache  4) train K2 from K on v2 + dagger, 12 epochs  5) Town02 test (seeds 123, 456)  6) milestone tour
# Town02 stays unseen.  CARLA is shut down at the end.  ~7 h.
$ErrorActionPreference = "Continue"
$here = (Resolve-Path "$PSScriptRoot\..\src").Path
$py = "python"
$t5 = "$PSScriptRoot\..\src\runs\ppo_20261115_100321\final_model.zip"
$k = "$PSScriptRoot\..\data\vision_runs\bev_20261117_120600\best.pt"
$v2 = "$PSScriptRoot\..\data\vision_data_v2"
$dag = "$PSScriptRoot\..\data\vision_data_v2_dagger"
Set-Location $here
"== diagnosis K  $(Get-Date -Format HH:mm)"
& powershell -ExecutionPolicy Bypass -File "$here\diag_student.ps1" -Tag K -Student $k -Runs "Town02:123,Town02:456,Town03:123" -V2
foreach ($t in "Town01", "Town03", "Town04", "Town05", "Town10HD") {
    "== DAgger collect $t  $(Get-Date -Format HH:mm)"
    cmd /c "`"$py`" -u vision_collect.py --v2 --driver mix --beta 0.2 --takeover 1.0 --steady --teacher `"$t5`" --student `"$k`" --town $t --target 10000 --out `"$dag`" >> runs\v2_dagger_$t.log 2>&1"
    Select-String -Path "runs\v2_dagger_$t.log" -Pattern "target reached|giving up" | Select-Object -Last 1 | ForEach-Object { $_.Line }
}
& $py -c "import carla_server as s; s.shutdown_if_configured(); print('CARLA running:', bool(s.processes_running()))"
"== segment leftovers + top-down cache  $(Get-Date -Format HH:mm)"
cmd /c "`"$py`" -u seg_dataset.py `"$dag`" > runs\v2_dagger_segment.log 2>&1"
cmd /c "`"$py`" -u bev_dataset.py `"$dag`" > runs\v2_dagger_bev.log 2>&1"
Select-String -Path runs\v2_dagger_segment.log, runs\v2_dagger_bev.log -Pattern "segmented|cache done|Error" | ForEach-Object { $_.Line }
"== train K2  $(Get-Date -Format HH:mm)"
cmd /c "`"$py`" -u bev_train.py --data `"$v2`" --data `"$dag`" --init `"$k`" --epochs 12 --lr 0.00015 > runs\bev_train_K2.log 2>&1"
Select-String -Path runs\bev_train_K2.log -Pattern "episodes|epoch|student:" | ForEach-Object { $_.Line }
$m = Select-String -Path runs\bev_train_K2.log -Pattern "^student: (.*)$"
if (-not $m) { "TRAINING FAILED - see runs\bev_train_K2.log" }
else {
    $k2 = $m.Matches[0].Groups[1].Value
    & powershell -ExecutionPolicy Bypass -File "$here\phaseD_test.ps1" -Student $k2 -Tag K2
    "== tour clip K2  $(Get-Date -Format HH:mm)"
    & $py -u record_tour.py --student $k2 --label "Student K2"
}
& $py -c "import carla_server as s; s.shutdown_if_configured(); print('CARLA running:', bool(s.processes_running()))"
"== done  $(Get-Date -Format HH:mm)"
