"""
모델 팩토리 - 동적 모델 생성 및 관리
다양한 이상 탐지 모델을 통합 인터페이스로 제공
"""

import torch
from typing import Dict, Type, Optional, Any
from pathlib import Path

from base_model import BaseAnomalyModel
from patchcore_model import PatchCoreModel


class ModelFactory:
    """
    이상 탐지 모델 팩토리 클래스
    모델 등록, 생성, 관리를 담당
    """
    
    # 등록된 모델들
    _models: Dict[str, Type[BaseAnomalyModel]] = {
        'patchcore': PatchCoreModel,
    }
    
    # 모델별 기본 설정
    _default_configs = {
        'patchcore': {
            'backbone': 'wide_resnet50_2',
            'layers': ['layer3'],
            'input_size': (224, 224),
            'normalization_mean': [0.485, 0.456, 0.406],
            'normalization_std': [0.229, 0.224, 0.225],
            'k_nearest_neighbors': 9,
            'coreset_sampling_ratio': 0.1,
            'batch_size': 8,
            'resolution_ratio': 0.35
        }
    }
    
    @classmethod
    def register_model(cls, 
                      name: str, 
                      model_class: Type[BaseAnomalyModel], 
                      default_config: Optional[Dict] = None) -> None:
        """
        새로운 모델 등록
        
        Args:
            name: 모델 이름
            model_class: 모델 클래스
            default_config: 기본 설정
        """
        cls._models[name.lower()] = model_class
        if default_config:
            cls._default_configs[name.lower()] = default_config
        
        print(f"📝 [ModelFactory] 모델 등록: {name}")
    
    @classmethod
    def get_available_models(cls) -> list:
        """
        사용 가능한 모델 목록 반환
        
        Returns:
            등록된 모델 이름 리스트
        """
        return list(cls._models.keys())
    
    @classmethod
    def create_model(cls, 
                    model_name: str, 
                    device: torch.device,
                    config: Optional[Dict] = None) -> BaseAnomalyModel:
        """
        모델 생성
        
        Args:
            model_name: 생성할 모델 이름
            device: 연산 디바이스
            config: 모델 설정 (None이면 기본 설정 사용)
            
        Returns:
            생성된 모델 인스턴스
            
        Raises:
            ValueError: 지원하지 않는 모델 이름인 경우
        """
        model_name = model_name.lower()
        
        if model_name not in cls._models:
            available = ', '.join(cls._models.keys())
            raise ValueError(f"지원하지 않는 모델: {model_name}. 사용 가능한 모델: {available}")
        
        # 기본 설정과 사용자 설정 병합
        model_config = cls._default_configs.get(model_name, {}).copy()
        if config:
            model_config.update(config)
        
        # 모델 생성
        model_class = cls._models[model_name]
        model = model_class(device=device, config=model_config)
        
        print(f"🏭 [ModelFactory] 모델 생성: {model_name}")
        return model
    
    @classmethod
    def load_model_from_file(cls, 
                            model_path: Path, 
                            device: torch.device) -> BaseAnomalyModel:
        """
        파일에서 모델 로드
        
        Args:
            model_path: 모델 파일 경로
            device: 연산 디바이스
            
        Returns:
            로드된 모델 인스턴스
            
        Raises:
            ValueError: 모델 로드 실패 시
        """
        model_path = Path(model_path)
        
        # 파일명으로 모델 타입 추정
        if 'patchcore' in model_path.name.lower():
            model_name = 'patchcore'
        else:
            # 기본값으로 patchcore 사용
            model_name = 'patchcore'
        
        # 모델 생성
        model = cls.create_model(model_name, device)
        
        # 모델 로드
        if model.load_model(model_path):
            print(f"📂 [ModelFactory] 모델 로드 성공: {model_path}")
            return model
        else:
            raise ValueError(f"모델 로드 실패: {model_path}")
    
    @classmethod
    def get_model_info(cls, model_name: str) -> Dict[str, Any]:
        """
        모델 정보 반환
        
        Args:
            model_name: 모델 이름
            
        Returns:
            모델 정보 딕셔너리
        """
        model_name = model_name.lower()
        
        if model_name not in cls._models:
            return {}
        
        model_class = cls._models[model_name]
        default_config = cls._default_configs.get(model_name, {})
        
        return {
            'name': model_name,
            'class': model_class.__name__,
            'description': model_class.__doc__ or "설명 없음",
            'default_config': default_config
        }
    
    @classmethod
    def update_model_config(cls, 
                           model_name: str, 
                           config_updates: Dict) -> None:
        """
        모델 기본 설정 업데이트
        
        Args:
            model_name: 모델 이름
            config_updates: 업데이트할 설정
        """
        model_name = model_name.lower()
        
        if model_name in cls._default_configs:
            cls._default_configs[model_name].update(config_updates)
            print(f"⚙️ [ModelFactory] {model_name} 설정 업데이트: {config_updates}")
        else:
            cls._default_configs[model_name] = config_updates
            print(f"⚙️ [ModelFactory] {model_name} 새 설정 등록: {config_updates}")


# 추가 모델 등록 예시 (향후 확장용)
def register_additional_models():
    """
    추가 모델들을 등록하는 함수
    새로운 모델을 개발했을 때 여기에 등록
    """
    
    # 예시: PADIM 모델 등록 (구현되면)
    # from .padim_model import PADIMModel
    # ModelFactory.register_model(
    #     'padim',
    #     PADIMModel,
    #     {
    #         'backbone': 'resnet18',
    #         'layers': ['layer1', 'layer2', 'layer3'],
    #         'input_size': (256, 256)
    #     }
    # )
    
    # 예시: FastFlow 모델 등록 (구현되면)
    # from .fastflow_model import FastFlowModel
    # ModelFactory.register_model(
    #     'fastflow',
    #     FastFlowModel,
    #     {
    #         'backbone': 'resnet18',
    #         'flow_steps': 8,
    #         'input_size': (256, 256)
    #     }
    # )
    
    pass


# 모듈 로드 시 추가 모델 등록
register_additional_models()


def get_model_factory() -> ModelFactory:
    """
    모델 팩토리 인스턴스 반환 (편의 함수)
    
    Returns:
        ModelFactory 클래스
    """
    return ModelFactory
