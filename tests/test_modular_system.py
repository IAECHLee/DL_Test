#!/usr/bin/env python3
"""
모듈러 딥러닝 모델 시스템 테스트 스크립트
새로운 모델 아키텍처의 동작을 확인하고 모델 교체 기능을 테스트
"""

import sys
import os
import numpy as np
import cv2
from pathlib import Path

# 프로젝트 경로 추가
current_dir = Path(__file__).parent
# 프로젝트 경로 추가
current_dir = Path(__file__).parent
project_root = current_dir.parent
models_dir = project_root / "models"
src_dir = project_root / "src"

sys.path.insert(0, str(models_dir))
sys.path.insert(0, str(src_dir))

# 모듈 임포트
from model_factory import ModelFactory, get_model_factory
from base_model import BaseAnomalyModel
from patchcore_model import PatchCoreModel
from gpu_optimized_detector import GPUDeepAnomalyDetector, create_detector, get_available_models


def test_model_factory():
    """모델 팩토리 기능 테스트"""
    print("=" * 60)
    print("🏭 모델 팩토리 테스트")
    print("=" * 60)
    
    # 사용 가능한 모델 목록
    available_models = ModelFactory.get_available_models()
    print(f"📋 사용 가능한 모델: {available_models}")
    
    # 모델 정보 조회
    for model_name in available_models:
        info = ModelFactory.get_model_info(model_name)
        print(f"\n🔍 {model_name} 모델 정보:")
        for key, value in info.items():
            if key != 'default_config':
                print(f"   {key}: {value}")
    
    return available_models


def test_model_creation():
    """모델 생성 테스트"""
    print("\n" + "=" * 60)
    print("🤖 모델 생성 테스트")
    print("=" * 60)
    
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 사용 디바이스: {device}")
    
    # PatchCore 모델 생성
    try:
        config = {
            'resolution_ratio': 0.5,
            'batch_size': 4
        }
        
        model = ModelFactory.create_model(
            model_name='patchcore',
            device=device,
            config=config
        )
        
        print(f"✅ 모델 생성 성공: {model.model_name}")
        print(f"📊 모델 정보: {model.get_model_info()}")
        
        return model
        
    except Exception as e:
        print(f"❌ 모델 생성 실패: {e}")
        return None


def test_detector_creation():
    """검출기 생성 테스트"""
    print("\n" + "=" * 60)
    print("🕵️ 검출기 생성 테스트")
    print("=" * 60)
    
    try:
        # create_detector 함수 테스트
        detector = create_detector(
            model_name='patchcore',
            resolution_ratio=0.4,
            batch_size=6
        )
        
        print(f"✅ 검출기 생성 성공")
        print(f"🔧 검출기 정보: {detector.get_model_info()}")
        
        return detector
        
    except Exception as e:
        print(f"❌ 검출기 생성 실패: {e}")
        return None


def test_model_switching(detector):
    """모델 교체 테스트"""
    print("\n" + "=" * 60)
    print("🔄 모델 교체 테스트")
    print("=" * 60)
    
    if detector is None:
        print("❌ 검출기가 없어 모델 교체 테스트를 건너뜁니다.")
        return
    
    current_model = detector.model_name
    print(f"🎯 현재 모델: {current_model}")
    
    # 사용 가능한 다른 모델 목록
    available_models = detector.get_available_models()
    print(f"📋 사용 가능한 모델: {available_models}")
    
    # 같은 모델로 다시 교체 (설정 변경 테스트)
    new_config = {
        'resolution_ratio': 0.6,
        'batch_size': 2
    }
    
    success = detector.switch_model(current_model, new_config)
    
    if success:
        print(f"✅ 모델 교체 성공: {current_model} (설정 변경)")
        print(f"📊 새 모델 정보: {detector.get_model_info()}")
    else:
        print(f"❌ 모델 교체 실패")


def create_dummy_images(num_images=5):
    """테스트용 더미 이미지 생성"""
    print("\n" + "=" * 60)
    print("🖼️ 테스트 이미지 생성")
    print("=" * 60)
    
    images = []
    for i in range(num_images):
        # 랜덤 노이즈가 있는 이미지 생성
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # 일부 패턴 추가
        cv2.rectangle(image, (50, 50), (150, 150), (255, 255, 255), -1)
        cv2.circle(image, (100, 100), 30, (0, 0, 255), -1)
        
        images.append(image)
    
    print(f"✅ {num_images}개 테스트 이미지 생성 완료")
    return images


def test_training(detector, images):
    """모델 훈련 테스트"""
    print("\n" + "=" * 60)
    print("🎓 모델 훈련 테스트")
    print("=" * 60)
    
    if detector is None or not images:
        print("❌ 검출기나 이미지가 없어 훈련 테스트를 건너뜁니다.")
        return False
    
    try:
        print(f"📚 {len(images)}개 이미지로 훈련 시작...")
        
        # 진행률 콜백 함수
        def progress_callback(progress):
            print(f"📈 훈련 진행률: {progress}%")
        
        result = detector.fit(images, progress_callback)
        
        if result.get('status') == 'success':
            print(f"✅ 훈련 성공!")
            print(f"🧠 메모리 뱅크 크기: {result.get('memory_bank_size', 'N/A')}")
            print(f"📐 특징 차원: {result.get('feature_dim', 'N/A')}")
            return True
        else:
            print(f"❌ 훈련 실패: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ 훈련 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_prediction(detector, images):
    """예측 테스트"""
    print("\n" + "=" * 60)
    print("🔮 예측 테스트")
    print("=" * 60)
    
    if detector is None or not images:
        print("❌ 검출기나 이미지가 없어 예측 테스트를 건너뜁니다.")
        return
    
    if not detector.model.is_trained:
        print("❌ 모델이 훈련되지 않아 예측 테스트를 건너뜁니다.")
        return
    
    try:
        # 첫 번째 이미지로 예측 테스트
        test_image = images[0]
        print(f"🖼️ 테스트 이미지 크기: {test_image.shape}")
        
        is_anomaly, score = detector.predict(test_image)
        
        print(f"🎯 예측 결과:")
        print(f"   이상 여부: {'⚠️ 이상' if is_anomaly else '✅ 정상'}")
        print(f"   이상 점수: {score:.4f}")
        
        # 히트맵 생성 테스트
        print(f"\n🔥 히트맵 생성 테스트...")
        heatmap = detector.create_heatmap(test_image)
        
        if heatmap is not None:
            print(f"✅ 히트맵 생성 성공: {heatmap.shape}")
        else:
            print(f"❌ 히트맵 생성 실패")
            
    except Exception as e:
        print(f"❌ 예측 중 오류: {e}")
        import traceback
        traceback.print_exc()


def test_save_load(detector):
    """모델 저장/로드 테스트"""
    print("\n" + "=" * 60)
    print("💾 모델 저장/로드 테스트")
    print("=" * 60)
    
    if detector is None:
        print("❌ 검출기가 없어 저장/로드 테스트를 건너뜁니다.")
        return
    
    if not detector.model.is_trained:
        print("❌ 모델이 훈련되지 않아 저장/로드 테스트를 건너뜁니다.")
        return
    
    try:
        # 임시 디렉토리에 모델 저장
        save_dir = current_dir / "temp_model_test"
        save_dir.mkdir(exist_ok=True)
        
        print(f"💾 모델 저장 중: {save_dir}")
        success = detector.save_model(str(save_dir))
        
        if success:
            print(f"✅ 모델 저장 성공")
            
            # 새 검출기 생성하여 모델 로드 테스트
            print(f"📂 새 검출기로 모델 로드 테스트...")
            
            new_detector = create_detector(model_name='patchcore')
            
            # 저장된 모델 파일 찾기
            model_files = list(save_dir.glob("*.pkl"))
            if model_files:
                model_path = model_files[0]
                load_success = new_detector.load_model(str(model_path))
                
                if load_success:
                    print(f"✅ 모델 로드 성공")
                    print(f"🔍 로드된 모델 정보: {new_detector.get_model_info()}")
                else:
                    print(f"❌ 모델 로드 실패")
            else:
                print(f"❌ 저장된 모델 파일을 찾을 수 없음")
        else:
            print(f"❌ 모델 저장 실패")
            
    except Exception as e:
        print(f"❌ 저장/로드 중 오류: {e}")
        import traceback
        traceback.print_exc()


def main():
    """메인 테스트 실행"""
    print("🚀 모듈러 딥러닝 모델 시스템 종합 테스트")
    print("=" * 80)
    
    # 1. 모델 팩토리 테스트
    available_models = test_model_factory()
    
    # 2. 모델 생성 테스트
    model = test_model_creation()
    
    # 3. 검출기 생성 테스트
    detector = test_detector_creation()
    
    # 4. 모델 교체 테스트
    test_model_switching(detector)
    
    # 5. 테스트 이미지 생성
    images = create_dummy_images(3)
    
    # 6. 훈련 테스트
    training_success = test_training(detector, images)
    
    # 7. 예측 테스트 (훈련이 성공한 경우에만)
    if training_success:
        test_prediction(detector, images)
        
        # 8. 저장/로드 테스트
        test_save_load(detector)
    
    print("\n" + "=" * 80)
    print("🎉 모든 테스트 완료!")
    print("=" * 80)


if __name__ == "__main__":
    main()
