"""
PatchCore 이상 탐지 모델 구현
torchvision ResNet 백본을 사용한 PatchCore 모델
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
import cv2
import os
from typing import List, Tuple, Dict, Optional, Any
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
import pickle

from base_model import BaseAnomalyModel


class FeatureExtractor(nn.Module):
    """ResNet 백본에서 특징 추출"""
    
    def __init__(self, backbone_name: str = 'wide_resnet50_2', layers: List[str] = None):
        super().__init__()
        
        if layers is None:
            layers = ['layer3']
            
        # ResNet 백본 로드
        if backbone_name == 'wide_resnet50_2':
            import torchvision.models as models
            self.backbone = models.wide_resnet50_2(pretrained=True)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        self.layers = layers
        self.features = {}
        
        # 특징 추출 후크 등록
        self._register_hooks()
        self.backbone.eval()
        
    def _register_hooks(self):
        """특징 추출을 위한 후크 등록"""
        def hook_fn(name):
            def hook(module, input, output):
                self.features[name] = output
            return hook
        
        for name in self.layers:
            if hasattr(self.backbone, name):
                getattr(self.backbone, name).register_forward_hook(hook_fn(name))
    
    def forward(self, x):
        """순전파 및 특징 추출"""
        self.features.clear()
        with torch.no_grad():
            _ = self.backbone(x)
        return self.features.copy()


class PatchCoreModel(BaseAnomalyModel):
    """GPU 최적화된 PatchCore 모델"""
    
    def __init__(self, device: torch.device, config: Optional[Dict] = None):
        super().__init__(device)
        self.model_name = "PatchCore"
        
        # 기본 설정
        default_config = {
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
        
        if config:
            default_config.update(config)
        self.config = default_config
        
        # 해상도 비율 설정
        self.resolution_ratio = self.config['resolution_ratio']
        
        print(f"⚡ [PatchCore] Device: {self.device}")
        print(f"📐 [Resolution] Target ratio: {self.resolution_ratio*100:.0f}%")
        
        self.feature_extractor = None
        self.transform = None
        self.memory_bank_gpu = None
        self.memory_bank_cpu = None
        self.nn_index = None
        
    def build_model(self) -> None:
        """모델 아키텍처 구성"""
        # 특징 추출기 초기화
        self.feature_extractor = FeatureExtractor(
            self.config['backbone'], 
            self.config['layers']
        ).to(self.device)
        
        # 데이터 전처리 파이프라인
        self.transform = T.Compose([
            T.Resize(self.config['input_size']),
            T.ToTensor(),
            T.Normalize(
                mean=self.config['normalization_mean'], 
                std=self.config['normalization_std']
            )
        ])
        
        print(f"🧠 [PatchCore] Built with {self.config['backbone']}")
        print(f"🎯 [PatchCore] Target layers: {self.config['layers']}")
    
    def preprocess_images_batch(self, images: List[np.ndarray]) -> torch.Tensor:
        """배치 이미지 전처리"""
        batch_tensors = []
        
        for image in images:
            # RGB 변환
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 해상도 보존 리사이즈
            patches = self.resize_with_aspect_ratio(image, self.config['input_size'])
            
            # 각 패치를 텐서로 변환
            for patch in patches:
                tensor_img = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
                
                # 정규화
                for i, (mean, std) in enumerate(zip(
                    self.config['normalization_mean'], 
                    self.config['normalization_std']
                )):
                    tensor_img[i] = (tensor_img[i] - mean) / std
                    
                batch_tensors.append(tensor_img)
        
        return torch.stack(batch_tensors).to(self.device)
    
    def resize_with_aspect_ratio(self, image: np.ndarray, target_size: Tuple[int, int]) -> List[np.ndarray]:
        """종횡비를 유지하면서 원본 해상도를 보존하는 패치 방식"""
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # 설정된 해상도 비율 사용
        min_resolution_ratio = self.resolution_ratio
        original_pixels = h * w
        target_total_pixels = original_pixels * min_resolution_ratio
        
        # 필요한 패치 개수 계산
        min_patches = max(1, int(np.ceil(target_total_pixels / (target_w * target_h))))
        
        # 종횡비를 고려한 패치 배치
        aspect_ratio = w / h
        
        if aspect_ratio > 1:  # 가로가 긴 이미지
            patch_cols = max(1, int(np.ceil(np.sqrt(min_patches * aspect_ratio))))
            patch_rows = max(1, int(np.ceil(min_patches / patch_cols)))
        else:  # 세로가 긴 이미지  
            patch_rows = max(1, int(np.ceil(np.sqrt(min_patches / aspect_ratio))))
            patch_cols = max(1, int(np.ceil(min_patches / patch_rows)))
        
        # 패치 생성
        patches = []
        patch_w = w // patch_cols
        patch_h = h // patch_rows
        
        for i in range(patch_rows):
            for j in range(patch_cols):
                y1 = i * patch_h
                y2 = min((i + 1) * patch_h, h)
                x1 = j * patch_w  
                x2 = min((j + 1) * patch_w, w)
                
                patch = image[y1:y2, x1:x2]
                resized_patch = cv2.resize(patch, target_size)
                patches.append(resized_patch)
        
        return patches
    
    def train(self, 
             normal_images: torch.Tensor, 
             progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """모델 학습 (정상 이미지로 메모리 뱅크 구성)"""
        if self.feature_extractor is None:
            self.build_model()
            
        self.feature_extractor.eval()
        all_features = []
        
        total_batches = len(normal_images)
        
        with torch.no_grad():
            for i, batch in enumerate(normal_images):
                if isinstance(batch, list):
                    # 리스트 형태의 이미지들을 배치로 변환
                    batch_tensor = self.preprocess_images_batch(batch)
                else:
                    batch_tensor = batch.to(self.device)
                
                # 특징 추출
                features_dict = self.feature_extractor(batch_tensor)
                
                # 모든 레이어의 특징을 결합
                batch_features = []
                for layer_name in self.config['layers']:
                    if layer_name in features_dict:
                        feat = features_dict[layer_name]
                        # 특징을 평탄화
                        feat_flat = feat.reshape(feat.size(0), feat.size(1), -1)
                        feat_flat = feat_flat.permute(0, 2, 1)  # [B, H*W, C]
                        batch_features.append(feat_flat.reshape(-1, feat.size(1)))
                
                if batch_features:
                    combined_features = torch.cat(batch_features, dim=0)
                    all_features.append(combined_features.cpu())
                
                # 메모리 정리
                self.clear_memory()
                
                # 진행률 업데이트
                if progress_callback:
                    progress = int((i + 1) / total_batches * 100)
                    progress_callback(progress)
        
        # 메모리 뱅크 구성
        if all_features:
            memory_bank = torch.cat(all_features, dim=0)
            
            # 코어셋 샘플링
            if len(memory_bank) > 1000:  # 너무 많은 특징이 있으면 샘플링
                sample_size = min(len(memory_bank), 
                                int(len(memory_bank) * self.config['coreset_sampling_ratio']))
                indices = torch.randperm(len(memory_bank))[:sample_size]
                memory_bank = memory_bank[indices]
            
            self.memory_bank_cpu = memory_bank
            self.memory_bank_gpu = memory_bank.to(self.device)
            
            # k-NN 인덱스 구성
            self.nn_index = NearestNeighbors(
                n_neighbors=self.config['k_nearest_neighbors'],
                metric='cosine'
            )
            self.nn_index.fit(memory_bank.numpy())
            
            self.is_trained = True
            
            print(f"✅ [PatchCore] 학습 완료 - 메모리 뱅크 크기: {len(memory_bank)}")
            
            return {
                'status': 'success',
                'memory_bank_size': len(memory_bank),
                'feature_dim': memory_bank.shape[1]
            }
        else:
            return {'status': 'failed', 'error': 'No features extracted'}
    
    def predict(self, images: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """이상 탐지 예측"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
            
        self.feature_extractor.eval()
        anomaly_scores = []
        anomaly_maps = []
        
        with torch.no_grad():
            for batch in images:
                if isinstance(batch, list):
                    batch_tensor = self.preprocess_images_batch(batch)
                else:
                    batch_tensor = batch.to(self.device)
                
                # 특징 추출
                features_dict = self.feature_extractor(batch_tensor)
                
                # 이상 점수 계산
                for layer_name in self.config['layers']:
                    if layer_name in features_dict:
                        feat = features_dict[layer_name]
                        
                        # 특징 평탄화
                        B, C, H, W = feat.shape
                        feat_flat = feat.reshape(B, C, -1).permute(0, 2, 1)  # [B, H*W, C]
                        
                        for b in range(B):
                            single_feat = feat_flat[b].cpu().numpy()  # [H*W, C]
                            
                            # k-NN 거리 계산
                            distances, _ = self.nn_index.kneighbors(single_feat)
                            scores = distances.mean(axis=1)  # [H*W]
                            
                            # 이상 맵 복원
                            anomaly_map = scores.reshape(H, W)
                            anomaly_maps.append(anomaly_map)
                            
                            # 전체 이상 점수
                            anomaly_scores.append(scores.max())
                
                self.clear_memory()
        
        return np.array(anomaly_scores), np.array(anomaly_maps)
    
    def save_model(self, save_dir: Path) -> str:
        """모델 저장"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'config': self.config,
            'memory_bank': self.memory_bank_cpu,
            'is_trained': self.is_trained,
            'model_name': self.model_name
        }
        
        if self.nn_index is not None:
            model_data['nn_index'] = self.nn_index
        
        save_path = save_dir / 'patchcore_model.pkl'
        
        with open(save_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 [PatchCore] 모델 저장 완료: {save_path}")
        return str(save_path)
    
    def load_model(self, model_path: Path) -> bool:
        """모델 로드"""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.config.update(model_data['config'])
            self.memory_bank_cpu = model_data['memory_bank']
            self.is_trained = model_data['is_trained']
            
            if self.memory_bank_cpu is not None:
                self.memory_bank_gpu = self.memory_bank_cpu.to(self.device)
            
            if 'nn_index' in model_data:
                self.nn_index = model_data['nn_index']
            
            # 모델 재구성
            self.build_model()
            
            print(f"📂 [PatchCore] 모델 로드 완료: {model_path}")
            return True
            
        except Exception as e:
            print(f"❌ [PatchCore] 모델 로드 실패: {e}")
            return False
