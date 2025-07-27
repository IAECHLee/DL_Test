# WSL 환경 설정 가이드

## 🐧 WSL Ubuntu에서 DL_Test 설정

### 1️⃣ 프로젝트 클론
```bash
# 홈 디렉토리로 이동
cd ~

# GitHub에서 클론
git clone https://github.com/IAECHLee/DL_Test.git
cd DL_Test
```

### 2️⃣ Conda 설치 (필요시)
```bash
# Miniconda 설치
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

### 3️⃣ CUDA 설정 확인
```bash
# NVIDIA 드라이버 확인
nvidia-smi

# CUDA 버전 확인  
nvcc --version
```

### 4️⃣ 환경 설정
```bash
# Conda 환경 생성
conda env create -f environment.yml
conda activate dl_test

# 또는 pip 사용
# conda create -n dl_test python=3.11
# conda activate dl_test  
# pip install -r requirements.txt
```

### 5️⃣ GPU 상태 확인
```bash
python check_gpu.py
```

### 6️⃣ 실행
```bash
# GUI 실행 (X11 포워딩 필요)
export DISPLAY=:0
python gui/gpu_anomaly_detector_gui.py

# 또는 테스트만
python tests/test_modular_system.py
```

## 🔧 WSL GPU 설정

### NVIDIA CUDA on WSL2
```bash
# WSL2에서 CUDA 지원 확인
nvidia-smi

# PyTorch CUDA 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### X11 포워딩 (GUI 사용시)
```bash
# WSL에서 GUI 사용을 위한 설정
sudo apt update
sudo apt install x11-apps

# Windows에서 VcXsrv 또는 X410 설치 필요
export DISPLAY=:0
```

## 📁 개발 워크플로우

### 노트북에서 작업 후 동기화
```bash
# 변경사항 커밋
git add .
git commit -m "Update: [작업 내용]"
git push origin main
```

### WSL에서 최신 변경사항 받기
```bash
git pull origin main
```

## 🚀 성능 비교

- **노트북**: RTX 3070 Laptop (8GB) - 모바일 GPU
- **회사PC**: 더 강력한 GPU 기대 (RTX 4080/4090 등)
- **WSL2**: 네이티브에 가까운 성능 (DirectML 지원)

## 💡 팁

1. **대용량 데이터**: Git LFS 사용 고려
2. **모델 파일**: `.gitignore`에 추가됨 (별도 관리)
3. **환경 동기화**: `environment.yml` 사용
4. **개발 환경**: VSCode Remote-WSL 확장 사용 권장
