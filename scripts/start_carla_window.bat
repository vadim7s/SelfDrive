@echo off
if "%CARLA_ROOT%"=="" set CARLA_ROOT=C:\CARLA
rem Evaluation / filming: CARLA with a window, Epic quality.
start "" "%CARLA_ROOT%\CarlaUE4.exe" -quality-level=Epic -carla-rpc-port=2000 -ResX=1920 -ResY=1080 -windowed
