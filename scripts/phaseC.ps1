# Phase C: data for the segmented top-down student.
#   powershell -ExecutionPolicy Bypass -File scripts\phaseC.ps1
# Re-running continues (collection counts frames on disk; segmentation skips done frames).
# 1) teacher v5 (--steady, bigger pushes) with the REAL-CAR camera (--v2: pitch -2, height 1.45, 640x480),
#    recording pose + traffic state + route per episode: Town01/03 15k, Town04/05/10HD 10k = ~60k frames.
#    Town02 stays unseen.  -> $PSScriptRoot\..\data\vision_data_v2
# 2) segment every frame with the Vistas segmenter (~0.1 s/frame)   3) situation label counts + timelines
# CARLA is shut down after collection.  ~3-4 h.
$ErrorActionPreference = "Continue"
$here = (Resolve-Path "$PSScriptRoot\..\src").Path
$py = "python"
$t5 = "$PSScriptRoot\..\src\runs\ppo_20261115_100321\final_model.zip"
$data = "$PSScriptRoot\..\data\vision_data_v2"
Set-Location $here
foreach ($pair in @(@("Town01", 15000), @("Town03", 15000), @("Town04", 10000), @("Town05", 10000), @("Town10HD", 10000))) {
    "== collect $($pair[0])  $(Get-Date -Format HH:mm)"
    cmd /c "`"$py`" -u vision_collect.py --v2 --steady --teacher `"$t5`" --town $($pair[0]) --target $($pair[1]) --noise-steer 0.10,0.25 --noise-steps 6,15 --out `"$data`" >> runs\v2_collect_$($pair[0]).log 2>&1"
    Select-String -Path "runs\v2_collect_$($pair[0]).log" -Pattern "target reached|giving up" | Select-Object -Last 1 | ForEach-Object { $_.Line }
}
& $py -c "import carla_server as s; s.shutdown_if_configured(); print('CARLA running:', bool(s.processes_running()))"
"== segment  $(Get-Date -Format HH:mm)"
cmd /c "`"$py`" -u seg_dataset.py `"$data`" > runs\v2_segment.log 2>&1"
Select-String -Path runs\v2_segment.log -Pattern "segmented|Error" | ForEach-Object { $_.Line }
"== situations  $(Get-Date -Format HH:mm)"
& $py -u situation.py $data 8
"== done  $(Get-Date -Format HH:mm)"
