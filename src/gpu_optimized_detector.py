"""
GPU 최적화된 딥러닝 기반 이상 감지 모듈 (완전 리팩토링됨)
- 모듈러 모델 아키텍처
- 동적 모델 교체 지원
- GPU 최적화 및 메모리 관리
"""

import torch
import numpy as np
import cv2
from pathlib import Path
import os
from typing import Optional, List, Tuple, Dict, Any
import logging
from tqdm import tqdm
import time
import pickle

# 현재 디렉토리의 image_utils import
try:
    from image_utils import get_target_roi_info
except ImportError:
    # image_utils가 없어도 동작하도록 fallback 함수 제공
    def get_target_roi_info(image_path):
        """ROI 자동 감지 함수 (fallback)"""
        return None, None, "image_utils not available"

# 새로운 모듈러 모델 시스템 import
import sys
from pathlib import Path

# 모델 디렉토리를 sys.path에 추가
current_dir = Path(__file__).parent
project_root = current_dir.parent
models_dir = project_root / "models"
sys.path.insert(0, str(models_dir))

from model_factory import ModelFactory
from base_model import BaseAnomalyModel

# GPU 사용 가능 여부 확인
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 [GPU Optimized] Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"🔥 [GPU Info] GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"💾 [GPU Info] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")


class GPUDeepAnomalyDetector:
    """GPU 최적화된 딥러닝 이상 감지 검출기 (모듈러 구조)"""
    
    def __init__(self, model_name: str = "patchcore", config: Optional[Dict] = None):
        """
        Args:
            model_name: 사용할 모델 이름 ('patchcore', 'padim', 'fastflow' 등)
            config: 모델별 설정 딕셔너리
        """
        # 디바이스 설정
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 기본 설정
        self.model_name = model_name
        self.config = config or {}
        
        # 해상도 비율 설정 (기본값: 35%)
        self.resolution_ratio = self.config.get('resolution_ratio', 0.35)
        
        print(f"🤖 [GPU Detector] Using {model_name} model")
        print(f"💻 [Device] {self.device}")
        print(f"📐 [Resolution] Initial ratio: {self.resolution_ratio*100:.0f}%")
        
        # 모델 팩토리를 통한 모델 생성
        try:
            self.model = ModelFactory.create_model(
                model_name=model_name,
                device=self.device,
                config=self.config
            )
            print(f"✅ [GPU Detector] Model created successfully")
        except ValueError as e:
            print(f"❌ [GPU Detector] Model creation failed: {e}")
            # 기본값으로 PatchCore 사용
            print(f"🔄 [GPU Detector] Falling back to PatchCore")
            self.model = ModelFactory.create_model(
                model_name="patchcore",
                device=self.device,
                config=self.config
            )
            self.model_name = "patchcore"
    
    def get_available_models(self) -> List[str]:
        """사용 가능한 모델 목록 반환"""
        return ModelFactory.get_available_models()
    
    def switch_model(self, model_name: str, config: Optional[Dict] = None) -> bool:
        """
        모델 교체
        
        Args:
            model_name: 새 모델 이름
            config: 새 모델 설정
            
        Returns:
            교체 성공 여부
        """
        try:
            # 기존 모델 메모리 정리
            if hasattr(self.model, 'clear_memory'):
                self.model.clear_memory()
            
            # 새 모델 생성
            new_config = self.config.copy()
            if config:
                new_config.update(config)
            
            self.model = ModelFactory.create_model(
                model_name=model_name,
                device=self.device,
                config=new_config
            )
            
            self.model_name = model_name
            self.config = new_config
            
            print(f"🔄 [GPU Detector] Model switched to {model_name}")
            return True
            
        except Exception as e:
            print(f"❌ [GPU Detector] Model switch failed: {e}")
            return False
    
    def set_resolution_ratio(self, ratio: float):
        """해상도 비율 설정"""
        if 0.2 <= ratio <= 1.0:
            self.resolution_ratio = ratio
            # 모델의 해상도 비율도 업데이트
            if hasattr(self.model, 'resolution_ratio'):
                self.model.resolution_ratio = ratio
            if hasattr(self.model, 'config') and 'resolution_ratio' in self.model.config:
                self.model.config['resolution_ratio'] = ratio
            print(f"📐 [Resolution] Updated ratio to {ratio*100:.0f}%")
        else:
            print(f"❌ [Resolution] Invalid ratio {ratio}, must be between 0.2 and 1.0")
            
    def fit(self, normal_images: List[np.ndarray], progress_callback: Optional[callable] = None):
        """모델 훈련"""
        print(f"🎯 [Training] Starting training with {len(normal_images)} normal images")
        
        # 이미지 배치로 분할
        batch_size = self.config.get('batch_size', 8)
        image_batches = []
        
        for i in range(0, len(normal_images), batch_size):
            batch = normal_images[i:i + batch_size]
            image_batches.append(batch)
        
        print(f"📦 [Training] Created {len(image_batches)} batches (batch_size={batch_size})")
        
        # 모델 훈련
        start_time = time.time()
        result = self.model.train(image_batches, progress_callback)
        training_time = time.time() - start_time
        
        print(f"⏱️ [Training] Completed in {training_time:.2f} seconds")
        return result
        
    def predict(self, image_path: str) -> Tuple[bool, float]:
        """이상 감지 예측"""
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
        else:
            image = image_path
            
        if image is None:
            return False, 0.0
        
        # 단일 이미지를 배치로 변환
        images_batch = [[image]]
        
        try:
            scores, _ = self.model.predict(images_batch)
            score = scores[0] if len(scores) > 0 else 0.0
            
            # 임계값 기반 이상 판정 (임시로 평균 + 2*표준편차 사용)
            # TODO: 더 정교한 임계값 설정 방법 구현
            is_anomaly = score > 0.5  # 임시 임계값
            
            return is_anomaly, float(score)
            
        except Exception as e:
            print(f"❌ [Prediction] Error: {e}")
            return False, 0.0
        
    def create_heatmap(self, image_path: str, patch_stride: int = 16) -> Optional[np.ndarray]:
        """GPU 최적화된 히트맵 생성"""
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
        else:
            image = image_path
            
        if image is None:
            return None
            
        if not self.model.is_trained:
            raise ValueError("Model not fitted")
            
        print(f"🔥 [GPU Heatmap] Creating heatmap for {image.shape} image")
        
        try:
            # 단일 이미지를 배치로 변환
            images_batch = [[image]]
            
            # 이상 점수와 맵 계산
            scores, anomaly_maps = self.model.predict(images_batch)
            
            if len(anomaly_maps) > 0:
                anomaly_map = anomaly_maps[0]
                
                # 히트맵을 원본 크기로 리사이즈
                h, w = image.shape[:2]
                heatmap = cv2.resize(anomaly_map, (w, h))
                
                # 0-255 범위로 정규화
                heatmap = ((heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8) * 255).astype(np.uint8)
                
                print(f"✅ [GPU Heatmap] Heatmap created successfully")
                return heatmap
            else:
                print(f"❌ [GPU Heatmap] No anomaly map generated")
                return None
                
        except Exception as e:
            print(f"❌ [GPU Heatmap] Error: {e}")
            return None
    
    def save_model(self, save_dir: str) -> bool:
        """모델 저장"""
        try:
            save_path = self.model.save_model(Path(save_dir))
            print(f"💾 [Save] Model saved to: {save_path}")
            return True
        except Exception as e:
            print(f"❌ [Save] Failed to save model: {e}")
            return False
    
    def load_model(self, model_path: str) -> bool:
        """모델 로드"""
        try:
            # 파일에서 모델 로드
            self.model = ModelFactory.load_model_from_file(
                model_path=Path(model_path),
                device=self.device
            )
            
            # 해상도 비율 동기화
            if hasattr(self.model, 'resolution_ratio'):
                self.resolution_ratio = self.model.resolution_ratio
            
            print(f"📂 [Load] Model loaded from: {model_path}")
            return True
            
        except Exception as e:
            print(f"❌ [Load] Failed to load model: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        base_info = {
            'detector_name': 'GPUDeepAnomalyDetector',
            'current_model': self.model_name,
            'device': str(self.device),
            'resolution_ratio': self.resolution_ratio,
            'available_models': self.get_available_models()
        }
        
        # 모델별 상세 정보 추가
        if hasattr(self.model, 'get_model_info'):
            model_info = self.model.get_model_info()
            base_info.update(model_info)
        
        return base_info
    
    def clear_memory(self):
        """메모리 정리"""
        if hasattr(self.model, 'clear_memory'):
            self.model.clear_memory()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


# 편의 함수들
def create_detector(model_name: str = "patchcore", 
                   resolution_ratio: float = 0.35,
                   **kwargs) -> GPUDeepAnomalyDetector:
    """
    검출기 생성 편의 함수
    
    Args:
        model_name: 모델 이름
        resolution_ratio: 해상도 비율 (0.2-1.0)
        **kwargs: 추가 모델 설정
    
    Returns:
        GPU 딥러닝 검출기 인스턴스
    """
    config = {'resolution_ratio': resolution_ratio}
    config.update(kwargs)
    
    detector = GPUDeepAnomalyDetector(model_name=model_name, config=config)
    return detector


def get_available_models() -> List[str]:
    """사용 가능한 모델 목록 반환"""
    return ModelFactory.get_available_models()


def get_model_info(model_name: str) -> Dict[str, Any]:
    """모델 정보 반환"""
    return ModelFactory.get_model_info(model_name)
