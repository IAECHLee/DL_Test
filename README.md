# DL_Test - GPU 최적화 이상 탐지 시스템

모듈러 딥러닝 모델 아키텍처를 사용한 GPU 가속 이상 탐지 시스템입니다.

## 🚀 프로젝트 구조

```
DL_Test/
├── models/                 # 딥러닝 모델 모듈
│   ├── __init__.py        # 모델 패키지 초기화
│   ├── base_model.py      # 기본 모델 인터페이스
│   ├── patchcore_model.py # PatchCore 모델 구현
│   └── model_factory.py   # 모델 팩토리
├── src/                   # 핵심 소스 코드
│   ├── gpu_optimized_detector.py  # GPU 최적화 검출기
│   └── image_utils.py     # 이미지 처리 유틸리티
├── gui/                   # GUI 인터페이스
│   └── gpu_anomaly_detector_gui.py  # PyQt5 기반 GUI
├── tests/                 # 테스트 코드
│   └── test_modular_system.py  # 시스템 테스트
├── config/                # 설정 파일들
├── docs/                  # 문서
├── requirements.txt       # Python 패키지 의존성
├── environment.yml        # Conda 환경 설정
└── README.md             # 프로젝트 문서
```

## 🔧 환경 설정

### 1. Conda 가상환경 생성
```bash
conda env create -f environment.yml
conda activate dl_test
```

### 2. 또는 pip 설치
```bash
conda create -n dl_test python=3.11
conda activate dl_test
pip install -r requirements.txt
```

## 🚀 실행 방법

### 🔍 GPU 상태 확인 (추천)
```bash
conda activate dl_test
cd DL_Test
python check_gpu.py
```

### GUI 실행
```bash
conda activate dl_test
cd DL_Test
python gui/gpu_anomaly_detector_gui.py
```

### 배치 파일 실행 (Windows)
```bash
# GUI 실행
run_gui.bat

# 테스트 실행  
run_test.bat
```

### 테스트 실행
```bash
conda activate dl_test
cd DL_Test
python tests/test_modular_system.py
```

## 🧪 주요 기능

- ✅ GPU 가속 PatchCore 모델 (CUDA 11.8)
- ✅ 모듈러 모델 아키텍처
- ✅ 동적 모델 교체
- ✅ 실시간 이상 탐지
- ✅ 히트맵 시각화
- ✅ 배치 처리
- ✅ ROI 자동 감지
- ✅ GPU 상태 자동 점검

## 🔧 문제 해결

### GPU 인식 문제가 있는 경우
1. **GPU 상태 확인**
   ```bash
   conda activate dl_test
   cd DL_Test
   python check_gpu.py
   ```

2. **CUDA 드라이버 확인**
   ```bash
   nvidia-smi
   ```

3. **가상환경 재설치** (필요시)
   ```bash
   conda env remove -n dl_test
   conda env create -f environment.yml
   ```

## 💡 사용법

```python
from src.gpu_optimized_detector import create_detector

# 검출기 생성
detector = create_detector(
    model_name='patchcore',
    resolution_ratio=0.4
)

# 훈련
detector.fit(normal_images)

# 예측
is_anomaly, score = detector.predict(test_image)
```

## 🔮 확장 가능한 모델

현재 지원:
- PatchCore

향후 추가 예정:
- PADIM
- FastFlow
- STFPM
