@echo off
echo ===========================================
echo DL_Test - GPU 이상 탐지 시스템 GUI 실행
echo ===========================================

REM 가상환경 활성화
call conda activate dl_test

REM DL_Test 디렉토리로 이동
cd /d "D:\Image_Processing\DL_Test"

REM GPU 상태 확인
echo 🔍 GPU 상태 확인 중...
python check_gpu.py

echo.
echo ✅ GPU 확인 완료! GUI를 시작합니다...
echo.

REM GPU GUI 실행
python gui/gpu_anomaly_detector_gui.py

pause
