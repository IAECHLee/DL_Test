#!/usr/bin/env python3
"""
GPU 상태 확인 스크립트
"""

import sys
from pathlib import Path
import torch

# 경로 설정
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / "src"))
sys.path.insert(0, str(current_dir / "models"))

def check_gpu_status():
    """GPU 상태 전체 점검"""
    print("=" * 60)
    print("🔍 GPU 상태 종합 점검")
    print("=" * 60)
    
    # PyTorch GPU 상태
    print(f"🐍 PyTorch 버전: {torch.__version__}")
    print(f"🔥 CUDA 사용 가능: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"⚡ CUDA 버전: {torch.version.cuda}")
        print(f"🖥️ GPU 이름: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"📊 GPU 개수: {torch.cuda.device_count()}")
    else:
        print("❌ CUDA를 사용할 수 없습니다")
        return False
    
    # 모듈 import 테스트
    print("\n📦 모듈 import 테스트")
    try:
        from gpu_optimized_detector import create_detector
        print("✅ gpu_optimized_detector import 성공")
    except Exception as e:
        print(f"❌ gpu_optimized_detector import 실패: {e}")
        return False
    
    # 검출기 생성 테스트
    print("\n🤖 검출기 생성 테스트")
    try:
        detector = create_detector()
        print("✅ 검출기 생성 성공")
        print(f"🎯 디바이스: {detector.device}")
        print(f"🧠 모델: {detector.model_name}")
        
        # 모델 정보
        info = detector.get_model_info()
        print(f"📊 검출기 정보: {info}")
        
    except Exception as e:
        print(f"❌ 검출기 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("🎉 모든 GPU 기능이 정상 작동합니다!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    check_gpu_status()
