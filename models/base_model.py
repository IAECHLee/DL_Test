"""
기본 이상 탐지 모델 인터페이스
모든 이상 탐지 모델이 구현해야 하는 공통 인터페이스 정의
"""

from abc import ABC, abstractmethod
import torch
import numpy as np
from typing import Tuple, Dict, Optional, Any
from pathlib import Path


class BaseAnomalyModel(ABC):
    """
    이상 탐지 모델 기본 클래스
    모든 이상 탐지 모델은 이 클래스를 상속받아 구현해야 함
    """
    
    def __init__(self, device: torch.device):
        """
        모델 초기화
        
        Args:
            device: 연산에 사용할 디바이스 (cuda/cpu)
        """
        self.device = device
        self.is_trained = False
        self.model_name = "base_model"
        
    @abstractmethod
    def build_model(self) -> None:
        """
        모델 아키텍처 구성
        """
        pass
    
    @abstractmethod
    def train(self, 
             normal_images: torch.Tensor, 
             progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        모델 학습
        
        Args:
            normal_images: 정상 이미지 텐서
            progress_callback: 진행상황 콜백 함수
            
        Returns:
            학습 결과 딕셔너리
        """
        pass
    
    @abstractmethod
    def predict(self, images: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        이상 탐지 예측
        
        Args:
            images: 예측할 이미지 텐서
            
        Returns:
            (anomaly_scores, anomaly_maps) 튜플
        """
        pass
    
    @abstractmethod
    def save_model(self, save_dir: Path) -> str:
        """
        모델 저장
        
        Args:
            save_dir: 저장할 디렉토리 경로
            
        Returns:
            저장된 파일 경로
        """
        pass
    
    @abstractmethod
    def load_model(self, model_path: Path) -> bool:
        """
        모델 로드
        
        Args:
            model_path: 모델 파일 경로
            
        Returns:
            로드 성공 여부
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        모델 정보 반환
        
        Returns:
            모델 정보 딕셔너리
        """
        return {
            'model_name': self.model_name,
            'device': str(self.device),
            'is_trained': self.is_trained,
            'parameters': self.get_parameter_count() if hasattr(self, 'model') else 0
        }
    
    def get_parameter_count(self) -> int:
        """
        모델 파라미터 수 반환
        
        Returns:
            총 파라미터 수
        """
        if hasattr(self, 'model') and self.model is not None:
            return sum(p.numel() for p in self.model.parameters())
        return 0
    
    def clear_memory(self) -> None:
        """
        메모리 정리
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        텐서를 모델 디바이스로 이동
        
        Args:
            tensor: 이동할 텐서
            
        Returns:
            디바이스로 이동된 텐서
        """
        return tensor.to(self.device)
