@echo off
cd /d D:\workspace\python\counting_app
C:\Users\PorNe\miniconda3\envs\yolov8\python.exe vehicle_counter.py "rtsp://root:pass@axis-b8a44fe03000-rama3-10-207-200-7.tail8176dd.ts.net/axis-media/media.amp?resolution=1280x960" --config "config\scene_config.json" --stats live_stats.json
pause
