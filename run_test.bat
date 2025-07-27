@echo off
echo ===========================================
echo DL_Test - 모듈식 시스템 테스트 및 GPU 검증
echo ===========================================

REM 가상환경 활성화
call conda activate dl_test

REM DL_Test 디렉토리로 이동
cd /d "D:\Image_Processing\DL_Test"

REM GPU 상태 확인
echo 🔍 GPU 상태 확인 중...
python check_gpu.py

echo.
echo ✅ GPU 확인 완료! 시스템 테스트를 시작합니다...
echo.

REM 테스트 실행
python tests/test_modular_system.py

pause
