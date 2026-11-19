@echo off
if "%CARLA_ROOT%"=="" set CARLA_ROOT=C:\CARLA
rem Training: CARLA without a window (fastest). Leave this running while train.py runs.
start "" "%CARLA_ROOT%\CarlaUE4.exe" -RenderOffScreen -quality-level=Low -carla-rpc-port=2000
