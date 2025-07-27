"""
DL_Test 프로젝트 - GPU 최적화 이상 탐지 시스템
모듈러 딥러닝 모델 아키텍처
"""

# 프로젝트 루트 경로 설정
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT / "src"))
sys.path.append(str(PROJECT_ROOT / "models"))
sys.path.append(str(PROJECT_ROOT / "gui"))

# 버전 정보
__version__ = "1.0.0"
__author__ = "DL_Test Team"
__description__ = "GPU 최적화 이상 탐지 시스템"
