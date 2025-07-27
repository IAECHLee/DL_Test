#!/usr/bin/env python3
"""
GPU 가속 이상 감지 GUI (PyQt5 기반)
- PatchCore GPU 최적화 알고리즘 적용
- ROI 영역 선택 기능
- 실시간 히트맵 시각화
- 배치 처리 및 결과 분석
"""

import os
import sys
import json
import time
import glob
import traceback
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

# PyQt5 imports
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                            QWidget, QPushButton, QLabel, QTextEdit, QProgressBar,
                            QFileDialog, QMessageBox, QComboBox, QSpinBox, QCheckBox,
                            QGroupBox, QGridLayout, QSlider, QTabWidget, QSplitter,
                            QScrollArea, QFrame, QButtonGroup, QRadioButton)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer, QRect
from PyQt5.QtGui import QFont, QPalette, QColor, QPixmap, QPainter, QPen, QImage

# Image processing and visualization
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib.figure import Figure

# GPU 최적화 모듈 import
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # 프로젝트 내 모듈 경로 추가
    models_dir = os.path.join(project_root, 'models')
    src_dir = os.path.join(project_root, 'src')
    sys.path.insert(0, models_dir)
    sys.path.insert(0, src_dir)
    
    from gpu_optimized_detector import GPUDeepAnomalyDetector, create_detector, get_available_models
    try:
        from image_utils import get_target_roi_info
        print("✂️ [Import] ROI 자동 감지 기능 로드 성공")
    except ImportError:
        # Fallback: 직접 구현
        def get_target_roi_info(image_path):
            """ROI 자동 감지 함수 (내장)"""
            import cv2
            import numpy as np
            
            img = cv2.imread(image_path)
            if img is None:
                return None, None, None
            
            H, W = img.shape[:2]
            N = max(1, int(W * 0.02))  # 샘플 구간 폭: 전체 가로의 2%
            
            # 1. 샘플 구간 설정
            left_bg = img[:, :N]            # 왼쪽 샘플(주로 배경)
            center = img[:, N:-N] if N < W//2 else img  # 중앙(제품)
            
            # 2. 밝기 평균값 산출
            left_mean = np.mean(left_bg)
            product_mean = np.mean(center)
            
            # 3. 임계값 계산 (배경과 제품의 중간값)
            threshold = (left_mean + product_mean) / 2
            
            # 4. 가로 projection (세로 평균)
            proj = np.mean(img, axis=0)
            
            # 5. 임계값보다 큰 x영역 찾기
            mask = proj > threshold
            x_indices = np.where(mask)[0]
            
            # 6. ROI 산출
            if len(x_indices) > 0:
                x1, x2 = int(x_indices[0]), int(x_indices[-1])
                y1, y2 = 0, H-1
                return x1, x2, (x1, y1, x2 - x1, y2 - y1)
            else:
                return None, None, None
        
        print("✂️ [Import] ROI 자동 감지 기능 내장 구현 사용")
    
    GPU_AVAILABLE = True
    print("🚀 [Import] GPU 최적화 모듈 로드 성공")
except ImportError as e:
    GPU_AVAILABLE = False
    print(f"❌ [Import Error] GPU 모듈 사용 불가: {e}")
    # Fallback function
    def get_target_roi_info(image_path):
        return None, None, None

# 한글 폰트 설정
def setup_korean_font():
    """한글 폰트 설정"""
    try:
        font_candidates = ['Malgun Gothic', 'Gulim', 'Dotum', 'Batang', 'DejaVu Sans']
        for font_name in font_candidates:
            try:
                plt.rcParams['font.family'] = font_name
                break
            except:
                continue
        plt.rcParams['axes.unicode_minus'] = False
    except:
        plt.rcParams['font.family'] = 'DejaVu Sans'

setup_korean_font()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gpu_anomaly_detector.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class ROISelector(QWidget):
    """ROI 영역 선택 위젯"""
    roi_changed = pyqtSignal(tuple)  # (x, y, width, height)
    
    def __init__(self):
        super().__init__()
        self.image_path = None
        self.image = None
        self.roi_rect = None
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.scale_factor = 1.0
        
        self.setMinimumSize(400, 300)
        self.setStyleSheet("border: 1px solid gray;")
        
    def load_image(self, image_path):
        """이미지 로드"""
        try:
            self.image_path = image_path
            self.image = cv2.imread(image_path)
            if self.image is not None:
                self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
                self.roi_rect = None
                self.update()
                return True
        except Exception as e:
            print(f"❌ 이미지 로드 실패: {e}")
        return False
        
    def paintEvent(self, event):
        """화면 그리기"""
        if self.image is None:
            return
            
        painter = QPainter(self)
        
        # 이미지 크기에 맞게 스케일 조정
        widget_size = self.size()
        image_height, image_width = self.image.shape[:2]
        
        scale_x = widget_size.width() / image_width
        scale_y = widget_size.height() / image_height
        self.scale_factor = min(scale_x, scale_y)
        
        new_width = int(image_width * self.scale_factor)
        new_height = int(image_height * self.scale_factor)
        
        # 이미지 중앙 정렬
        x_offset = (widget_size.width() - new_width) // 2
        y_offset = (widget_size.height() - new_height) // 2
        
        # 이미지 그리기
        image_resized = cv2.resize(self.image, (new_width, new_height))
        height, width, channel = image_resized.shape
        bytes_per_line = 3 * width
        q_image = QPixmap.fromImage(
            QImage(image_resized.data, width, height, bytes_per_line, QImage.Format_RGB888))
        
        painter.drawPixmap(x_offset, y_offset, q_image)
        
        # ROI 사각형 그리기
        if self.roi_rect:
            painter.setPen(QPen(Qt.red, 2))
            scaled_rect = QRect(
                int(self.roi_rect[0] * self.scale_factor) + x_offset,
                int(self.roi_rect[1] * self.scale_factor) + y_offset,
                int(self.roi_rect[2] * self.scale_factor),
                int(self.roi_rect[3] * self.scale_factor)
            )
            painter.drawRect(scaled_rect)
            
    def mousePressEvent(self, event):
        """마우스 클릭 시작"""
        if self.image is None:
            return
            
        self.drawing = True
        self.start_point = event.pos()
        
    def mouseMoveEvent(self, event):
        """마우스 드래그"""
        if self.drawing and self.start_point:
            self.end_point = event.pos()
            self.update()
            
    def mouseReleaseEvent(self, event):
        """마우스 클릭 종료"""
        if self.drawing and self.start_point:
            self.end_point = event.pos()
            self.drawing = False
            
            # ROI 좌표 계산
            widget_size = self.size()
            image_height, image_width = self.image.shape[:2]
            
            x_offset = (widget_size.width() - int(image_width * self.scale_factor)) // 2
            y_offset = (widget_size.height() - int(image_height * self.scale_factor)) // 2
            
            x1 = max(0, min(self.start_point.x(), self.end_point.x()) - x_offset)
            y1 = max(0, min(self.start_point.y(), self.end_point.y()) - y_offset)
            x2 = max(0, max(self.start_point.x(), self.end_point.x()) - x_offset)
            y2 = max(0, max(self.start_point.y(), self.end_point.y()) - y_offset)
            
            # 원본 이미지 좌표로 변환
            x1 = int(x1 / self.scale_factor)
            y1 = int(y1 / self.scale_factor)
            width = int((x2 - x1) / self.scale_factor)
            height = int((y2 - y1) / self.scale_factor)
            
            if width > 10 and height > 10:  # 최소 크기 확인
                self.roi_rect = (x1, y1, width, height)
                self.roi_changed.emit(self.roi_rect)
                self.update()

class GPUTrainingThread(QThread):
    """GPU 모델 훈련 스레드"""
    progress_update = pyqtSignal(int, str)
    training_complete = pyqtSignal(bool, str, dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, config, image_paths, model_save_path, roi_rect=None):
        super().__init__()
        self.config = config
        self.image_paths = image_paths
        self.model_save_path = model_save_path
        self.roi_rect = roi_rect
        self.detector = None
        
    def run(self):
        try:
            # GPU 메모리 정리
            if self.config.use_gpu:
                try:
                    import torch
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    self.progress_update.emit(2, "🧹 GPU 메모리 정리 완료")
                except:
                    pass
            
            self.progress_update.emit(5, "🤖 GPU 검출기 초기화 중...")
            
            # 해상도 비율 가져오기
            resolution_ratio = self.config.get('resolution_ratio', 0.35)
            
            # GPU 검출기 생성 (새로운 모듈러 구조 사용)
            self.detector = create_detector(
                model_name='patchcore',
                **self.config
            )
            
            self.progress_update.emit(15, f"📂 {len(self.image_paths)}개 훈련 이미지 로드 중...")
            
            # 메모리 절약을 위한 배치 크기 추가 제한
            use_gpu = self.config.get('use_gpu', True)
            batch_size = self.config.get('batch_size', 8)
            
            if use_gpu:
                # 메모리 최적화를 기본값으로 활성화
                if len(self.image_paths) > 50:
                    batch_size = min(batch_size, 6)
                elif len(self.image_paths) > 100:
                    batch_size = min(batch_size, 4)
                else:
                    batch_size = min(batch_size, 8)
                
                # 설정 업데이트
                self.detector.config['batch_size'] = batch_size
                self.progress_update.emit(18, f"⚡ 메모리 최적화: 배치 크기 {batch_size}로 설정")
            
            # 해상도 비율 정보 표시
            resolution_ratio = getattr(self.config, 'resolution_ratio', 0.35)
            self.progress_update.emit(20, f"📐 해상도 비율: {resolution_ratio*100:.0f}% (원본 대비 해상도 유지)")
            
            # ROI 영역이 있으면 이미지 전처리
            processed_paths = []
            if self.roi_rect:
                self.progress_update.emit(25, "✂️ ROI 영역 처리 중...")
                for i, img_path in enumerate(self.image_paths):
                    try:
                        image = cv2.imread(img_path)
                        if image is not None:
                            x, y, w, h = self.roi_rect
                            roi_image = image[y:y+h, x:x+w]
                            
                            # 임시 ROI 이미지 저장
                            temp_path = f"temp_roi_{i}.jpg"
                            cv2.imwrite(temp_path, roi_image)
                            processed_paths.append(temp_path)
                    except Exception as e:
                        print(f"❌ ROI 처리 실패: {img_path}, {e}")
                        
                training_paths = processed_paths if processed_paths else self.image_paths
            else:
                training_paths = self.image_paths
                
            self.progress_update.emit(40, "🎓 GPU 가속 훈련 시작...")
            
            # 모델 훈련 (메모리 관리 포함)
            start_time = time.time()
            
            # 훈련 중 메모리 모니터링
            if self.config.use_gpu:
                try:
                    import torch
                    initial_memory = torch.cuda.memory_allocated() / 1024**3
                    self.progress_update.emit(45, f"🔍 초기 GPU 메모리: {initial_memory:.2f}GB")
                except:
                    pass
            
            self.detector.fit(training_paths)
            training_time = time.time() - start_time
            
            # 훈련 후 메모리 정리
            if self.config.use_gpu:
                try:
                    import torch
                    final_memory = torch.cuda.memory_allocated() / 1024**3
                    torch.cuda.empty_cache()
                    cleaned_memory = torch.cuda.memory_allocated() / 1024**3
                    self.progress_update.emit(75, f"🧹 GPU 메모리 정리: {final_memory:.2f}GB → {cleaned_memory:.2f}GB")
                except:
                    pass
            
            self.progress_update.emit(80, "💾 모델 자동 저장 중...")
            
            # 모델 저장 경로 확인 및 생성
            save_dir = os.path.dirname(self.model_save_path)
            if save_dir and not os.path.exists(save_dir):
                try:
                    os.makedirs(save_dir, exist_ok=True)
                    self.progress_update.emit(82, f"📁 저장 폴더 생성: {save_dir}")
                except Exception as e:
                    self.progress_update.emit(82, f"❌ 폴더 생성 실패: {e}")
                    # 현재 디렉토리에 저장
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    self.model_save_path = os.path.join(os.getcwd(), f"deep_model_{timestamp}.pkl")
            
            # 모델 저장 실행
            try:
                self.detector.save_model(self.model_save_path)
                self.progress_update.emit(90, f"✅ 모델 저장 완료: {os.path.basename(self.model_save_path)}")
                self.progress_update.emit(92, f"📁 저장 위치: {os.path.dirname(self.model_save_path)}")
            except Exception as e:
                self.progress_update.emit(90, f"❌ 모델 저장 실패: {e}")
                # 오류 발생 시 현재 디렉토리에 재시도
                try:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    fallback_path = os.path.join(os.getcwd(), f"deep_model_{timestamp}.pkl")
                    self.detector.save_model(fallback_path)
                    self.model_save_path = fallback_path
                    self.progress_update.emit(92, f"✅ 대체 경로 저장 완료: {fallback_path}")
                except Exception as e2:
                    self.progress_update.emit(92, f"❌ 대체 저장도 실패: {e2}")
                
            # 임시 파일 정리
            if self.roi_rect:
                for temp_path in processed_paths:
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                        
            self.progress_update.emit(100, "✅ GPU 훈련 완료!")
            
            # 결과 정보
            results = {
                'model_type': self.config.model_name,
                'backbone': 'wide_resnet50_2',
                'num_training_images': len(self.image_paths),
                'training_time': training_time,
                'use_gpu': self.config.use_gpu,
                'roi_applied': self.roi_rect is not None,
                'roi_rect': self.roi_rect,
                'model_saved_path': self.model_save_path  # 저장된 모델 경로 추가
            }
            
            self.training_complete.emit(True, "GPU 훈련이 성공적으로 완료되었습니다!", results)
            
        except Exception as e:
            error_msg = f"GPU 훈련 실패: {str(e)}\n{traceback.format_exc()}"
            logging.error(error_msg)
            self.error_occurred.emit(error_msg)

class GPUAnalysisThread(QThread):
    """GPU 분석 스레드"""
    progress_update = pyqtSignal(int, str)
    analysis_complete = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, detector, test_image_paths, create_heatmaps=True, roi_rect=None):
        super().__init__()
        self.detector = detector
        self.test_image_paths = test_image_paths
        self.create_heatmaps = create_heatmaps
        self.roi_rect = roi_rect
        
    def run(self):
        try:
            results = {
                'total_images': len(self.test_image_paths),
                'anomaly_images': [],
                'normal_images': [],
                'analysis_details': [],
                'roi_applied': self.roi_rect is not None
            }
            
            for i, img_path in enumerate(self.test_image_paths):
                progress = int((i / len(self.test_image_paths)) * 100)
                progress_msg = f"🔍 GPU 분석 중 {i+1}/{len(self.test_image_paths)}"
                self.progress_update.emit(progress, progress_msg)
                
                try:
                    # ROI 영역 처리
                    if self.roi_rect:
                        image = cv2.imread(img_path)
                        x, y, w, h = self.roi_rect
                        roi_image = image[y:y+h, x:x+w]
                        temp_path = f"temp_analysis_{i}.jpg"
                        cv2.imwrite(temp_path, roi_image)
                        analysis_path = temp_path
                    else:
                        analysis_path = img_path
                    
                    # GPU 이상 감지
                    start_time = time.time()
                    is_anomaly, score = self.detector.predict(analysis_path)
                    inference_time = time.time() - start_time
                    
                    # 히트맵 생성
                    heatmap = None
                    heatmap_time = 0
                    if self.create_heatmaps:
                        try:
                            heatmap_start = time.time()
                            heatmap = self.detector.create_heatmap(analysis_path)
                            heatmap_time = time.time() - heatmap_start
                        except Exception as e:
                            print(f"❌ 히트맵 생성 실패: {e}")
                    
                    analysis_result = {
                        'image_path': img_path,
                        'is_anomaly': is_anomaly,
                        'anomaly_score': score,
                        'inference_time': inference_time,
                        'heatmap': heatmap,
                        'heatmap_time': heatmap_time,
                        'roi_rect': self.roi_rect
                    }
                    
                    results['analysis_details'].append(analysis_result)
                    
                    if is_anomaly:
                        results['anomaly_images'].append(img_path)
                    else:
                        results['normal_images'].append(img_path)
                        
                    # ROI 임시 파일 정리
                    if self.roi_rect and os.path.exists(temp_path):
                        os.remove(temp_path)
                        
                except Exception as e:
                    print(f"❌ 분석 실패: {img_path}, {e}")
                    continue
                    
            self.progress_update.emit(100, "✅ GPU 분석 완료!")
            self.analysis_complete.emit(results)
            
        except Exception as e:
            error_msg = f"GPU 분석 실패: {str(e)}\n{traceback.format_exc()}"
            logging.error(error_msg)
            self.error_occurred.emit(error_msg)

class GPUVisualizationWidget(QWidget):
    """GPU 분석 결과 시각화 위젯"""
    
    def __init__(self):
        super().__init__()
        self.figure = Figure(figsize=(16, 12))
        self.canvas = FigureCanvas(self.figure)
        
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
    def plot_gpu_analysis_results(self, analysis_results):
        """GPU 분석 결과 시각화"""
        self.figure.clear()
        
        if not analysis_results or not analysis_results.get('analysis_details'):
            ax = self.figure.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, 'GPU 분석 결과가 없습니다', 
                   ha='center', va='center', fontsize=16)
            self.canvas.draw()
            return
            
        # 2x3 레이아웃
        gs = self.figure.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 1])
        
        details = analysis_results['analysis_details']
        scores = [d['anomaly_score'] for d in details]
        anomaly_flags = [d['is_anomaly'] for d in details]
        inference_times = [d['inference_time'] for d in details]
        
        # 1. 이상도 점수 분포
        ax1 = self.figure.add_subplot(gs[0, 0])
        ax1.hist(scores, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(np.mean(scores), color='red', linestyle='--', 
                   label=f'평균: {np.mean(scores):.3f}')
        ax1.set_xlabel('이상도 점수')
        ax1.set_ylabel('빈도')
        ax1.set_title('🔍 GPU 이상도 점수 분포')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 정상/이상 비율
        ax2 = self.figure.add_subplot(gs[0, 1])
        normal_count = len(analysis_results['normal_images'])
        anomaly_count = len(analysis_results['anomaly_images'])
        
        colors = ['lightgreen', 'salmon']
        labels = [f'정상 ({normal_count})', f'이상 ({anomaly_count})']
        sizes = [normal_count, anomaly_count]
        
        if sum(sizes) > 0:
            wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors, 
                                              autopct='%1.1f%%', startangle=90)
            ax2.set_title('🎯 GPU 감지 결과')
        
        # 3. GPU 추론 시간 분석
        ax3 = self.figure.add_subplot(gs[0, 2])
        ax3.hist(inference_times, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        ax3.axvline(np.mean(inference_times), color='blue', linestyle='--',
                   label=f'평균: {np.mean(inference_times):.3f}초')
        ax3.set_xlabel('추론 시간 (초)')
        ax3.set_ylabel('빈도')
        ax3.set_title('⚡ GPU 추론 시간 분포')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4-6. 상위 3개 이상 히트맵
        if any(d.get('heatmap') is not None for d in details):
            top_anomalies = sorted(details, key=lambda x: x['anomaly_score'], reverse=True)[:3]
            
            for i, result in enumerate(top_anomalies):
                if result.get('heatmap') is not None:
                    ax = self.figure.add_subplot(gs[1, i])
                    
                    heatmap = result['heatmap']
                    im = ax.imshow(heatmap, cmap='hot', alpha=0.8)
                    
                    # 원본 이미지 오버레이
                    try:
                        img = cv2.imread(result['image_path'])
                        if img is not None:
                            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            
                            # ROI 적용된 경우
                            if result.get('roi_rect'):
                                x, y, w, h = result['roi_rect']
                                img_rgb = img_rgb[y:y+h, x:x+w]
                                
                            img_resized = cv2.resize(img_rgb, (heatmap.shape[1], heatmap.shape[0]))
                            ax.imshow(img_resized, alpha=0.4)
                    except:
                        pass
                    
                    filename = Path(result['image_path']).name
                    roi_text = " (ROI)" if result.get('roi_rect') else ""
                    ax.set_title(f'🔥 히트맵: {filename}{roi_text}\n점수: {result["anomaly_score"]:.3f} '
                               f'({result["inference_time"]:.3f}초)', fontsize=9)
                    ax.axis('off')
                    
                    # 컬러바
                    self.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # ROI 정보 표시
        roi_info = " (ROI 적용)" if analysis_results.get('roi_applied') else ""
        avg_inference = np.mean(inference_times) if inference_times else 0
        
        self.figure.suptitle(f'🚀 GPU 가속 이상 감지 분석 결과{roi_info}\n'
                           f'⚡ 평균 추론 시간: {avg_inference:.3f}초', 
                           fontsize=14, fontweight='bold')
        self.figure.tight_layout(rect=[0, 0.03, 1, 0.92])
        self.canvas.draw()

class GPUAnomalyDetectorUI(QMainWindow):
    """GPU 가속 이상 감지 메인 UI"""
    
    def __init__(self):
        super().__init__()
        self.gpu_detector = None
        self.analysis_results = None
        self.current_config = None
        self.roi_rect = None
        
        # 저장 경로 관련 변수 초기화
        self.selected_save_folder = None
        
        # GPU 가용성 확인
        self.gpu_available = False
        if GPU_AVAILABLE:
            try:
                import torch
                self.gpu_available = torch.cuda.is_available()
            except ImportError:
                self.gpu_available = False
        
        self.init_ui()
        self.setup_logging()
        
        # 초기 상태 메시지
        if GPU_AVAILABLE and self.gpu_available:
            try:
                import torch
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                self.log_message(f"🚀 GPU 가속 활성화: {gpu_name} ({gpu_memory:.1f} GB)")
            except:
                self.log_message("🚀 GPU 가속 활성화")
        else:
            self.log_message("⚠️ GPU 사용 불가 - CPU 모드로 실행")
            
        # GPU 상태 주기적 업데이트 (30초마다)
        if self.gpu_available:
            self.gpu_status_timer = QTimer()
            self.gpu_status_timer.timeout.connect(self.update_gpu_status)
            self.gpu_status_timer.start(30000)  # 30초
            
    def init_ui(self):
        """UI 초기화"""
        self.setWindowTitle("🚀 GPU 가속 이상 감지 시스템 v2.0")
        self.setGeometry(100, 100, 1800, 1200)
        
        # 중앙 위젯
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 메인 레이아웃 (좌우 분할)
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # 좌측 제어 패널
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel, 1)
        
        # 우측 결과 표시
        result_panel = self.create_result_panel()
        main_layout.addWidget(result_panel, 2)
        
        # 다크 테마 적용
        self.apply_dark_theme()
        
    def create_control_panel(self):
        """제어 패널 생성"""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # 탭 위젯
        tab_widget = QTabWidget()
        layout.addWidget(tab_widget)
        
        # 1. GPU 설정 탭
        gpu_tab = self.create_gpu_config_tab()
        tab_widget.addTab(gpu_tab, "🔧 GPU 설정")
        
        # 2. ROI 설정 탭
        roi_tab = self.create_roi_tab()
        tab_widget.addTab(roi_tab, "✂️ ROI 설정")
        
        # 3. 훈련 탭
        training_tab = self.create_training_tab()
        tab_widget.addTab(training_tab, "🎓 모델 훈련")
        
        # 4. 분석 탭
        analysis_tab = self.create_analysis_tab()
        tab_widget.addTab(analysis_tab, "🔍 이상 감지")
        
        # 하단 로그
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(180)
        self.log_text.setFont(QFont("Consolas", 9))
        layout.addWidget(QLabel("📋 시스템 로그:"))
        layout.addWidget(self.log_text)
        
        return panel
        
    def create_result_panel(self):
        """결과 패널 생성"""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # 제목
        title_label = QLabel("📊 GPU 분석 결과")
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # 시각화 위젯
        self.visualization_widget = GPUVisualizationWidget()
        layout.addWidget(self.visualization_widget)
        
        return panel
        
    def create_gpu_config_tab(self):
        """GPU 설정 탭"""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        # GPU 모델 설정
        gpu_group = QGroupBox("🚀 GPU 가속 설정")
        gpu_layout = QGridLayout()
        gpu_group.setLayout(gpu_layout)
        
        # 모델 타입
        gpu_layout.addWidget(QLabel("모델 타입:"), 0, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["PatchCore"])
        gpu_layout.addWidget(self.model_combo, 0, 1)
        
        # GPU 사용
        self.gpu_checkbox = QCheckBox("GPU 가속 사용")
        self.gpu_checkbox.setChecked(self.gpu_available)
        self.gpu_checkbox.setEnabled(self.gpu_available)
        gpu_layout.addWidget(self.gpu_checkbox, 1, 0, 1, 2)
        
        # 배치 크기 (메모리 절약을 위해 기본값 감소)
        gpu_layout.addWidget(QLabel("배치 크기:"), 2, 0)
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 32)
        self.batch_size_spin.setValue(8 if self.gpu_available else 4)  # 메모리 절약
        gpu_layout.addWidget(self.batch_size_spin, 2, 1)
        
        # 메모리 관리 옵션
        self.memory_optimize_checkbox = QCheckBox("메모리 최적화 모드 (대용량 데이터셋용)")
        self.memory_optimize_checkbox.setChecked(True)
        gpu_layout.addWidget(self.memory_optimize_checkbox, 3, 0, 1, 2)
        
        # 해상도 비율 설정
        gpu_layout.addWidget(QLabel("원본 해상도 비율:"), 4, 0)
        self.resolution_ratio_spin = QSpinBox()
        self.resolution_ratio_spin.setRange(20, 100)  # 20% ~ 100%
        self.resolution_ratio_spin.setValue(35)  # 기본값 35%
        self.resolution_ratio_spin.setSuffix("%")
        self.resolution_ratio_spin.valueChanged.connect(self.on_resolution_ratio_changed)
        gpu_layout.addWidget(self.resolution_ratio_spin, 4, 1)
        
        # 비율 안내 레이블
        self.resolution_info = QLabel("💡 35%: 고해상도 유지, 메모리 사용량 중간")
        self.resolution_info.setStyleSheet("color: #90EE90; font-size: 10px; font-weight: bold;")
        gpu_layout.addWidget(self.resolution_info, 5, 0, 1, 2)
        
        # 해상도 미리보기 버튼
        self.preview_resolution_btn = QPushButton("🔍 해상도 효과 미리보기")
        self.preview_resolution_btn.clicked.connect(self.preview_resolution_effect)
        gpu_layout.addWidget(self.preview_resolution_btn, 6, 0, 1, 2)
        
        # 메모리 정리 버튼
        self.clear_memory_btn = QPushButton("🧹 GPU 메모리 정리")
        self.clear_memory_btn.clicked.connect(self.clear_gpu_memory)
        gpu_layout.addWidget(self.clear_memory_btn, 7, 0, 1, 2)
        
        layout.addWidget(gpu_group)
        
        # GPU 상태 정보
        status_group = QGroupBox("📊 GPU 상태")
        status_layout = QVBoxLayout()
        status_group.setLayout(status_layout)
        
        self.gpu_status_label = QLabel("GPU 정보 로딩 중...")
        status_layout.addWidget(self.gpu_status_label)
        
        # GPU 정보 업데이트
        self.update_gpu_status()
        
        layout.addWidget(status_group)
        layout.addStretch()
        return tab
        
    def create_roi_tab(self):
        """ROI 설정 탭"""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        # 자동 ROI 안내 메시지
        auto_roi_info = QGroupBox("🤖 자동 ROI 처리 정보")
        info_layout = QVBoxLayout()
        auto_roi_info.setLayout(info_layout)
        
        auto_info_label = QLabel("""
        ✅ ROI (관심 영역) 자동 감지가 활성화되었습니다!
        
        🔍 모든 이미지는 훈련 및 분석 전에 자동으로 ROI 영역이 감지되어 처리됩니다:
        • 이미지의 배경과 제품 영역을 자동으로 구분
        • 제품 영역만 추출하여 모델 훈련 및 이상 감지 수행
        • 종횡비를 유지하면서 표준 크기로 자동 리사이즈
        • 불규칙한 이미지 크기 문제 자동 해결
        
        📐 처리 과정:
        1. 이미지 로드 → 2. ROI 자동 감지 → 3. 제품 영역 크롭 → 4. 종횡비 유지 리사이즈 → 5. 모델 처리
        
        ⚡ 별도의 ROI 설정 없이 자동으로 최적화된 분석을 수행합니다!
        """)
        auto_info_label.setWordWrap(True)
        auto_info_label.setStyleSheet("""
            QLabel {
                background-color: #2d4739;
                color: #90EE90;
                padding: 15px;
                border-radius: 8px;
                border: 2px solid #4a7c59;
                font-size: 11px;
                line-height: 1.4;
            }
        """)
        info_layout.addWidget(auto_info_label)
        
        # ROI 활성화 (참조용 - 실제로는 자동 처리됨)
        roi_group = QGroupBox("✂️ ROI 설정 (참조용)")
        roi_layout = QVBoxLayout()
        roi_group.setLayout(roi_layout)
        
        # ROI 사용 여부 (항상 활성화됨을 표시)
        self.roi_checkbox = QCheckBox("ROI 영역 사용 (자동 활성화됨)")
        self.roi_checkbox.setChecked(True)
        self.roi_checkbox.setEnabled(False)  # 비활성화 (자동 처리됨)
        roi_layout.addWidget(self.roi_checkbox)
        
        # 이미지 선택
        img_layout = QHBoxLayout()
        self.roi_image_label = QLabel("ROI 확인용 이미지 선택 (옵션)")
        self.select_roi_image_btn = QPushButton("📁 이미지 선택")
        self.auto_roi_btn = QPushButton("🎯 ROI 미리보기")
        img_layout.addWidget(self.roi_image_label)
        img_layout.addWidget(self.select_roi_image_btn)
        img_layout.addWidget(self.auto_roi_btn)
        roi_layout.addLayout(img_layout)
        
        # ROI 선택기
        self.roi_selector = ROISelector()
        roi_layout.addWidget(self.roi_selector)
        
        # ROI 좌표 정보
        self.roi_info_label = QLabel("ROI 영역: 모든 이미지에 자동 적용됨")
        self.roi_info_label.setStyleSheet("color: #90EE90; font-weight: bold;")
        roi_layout.addWidget(self.roi_info_label)
        
        # 이벤트 연결
        self.select_roi_image_btn.clicked.connect(self.select_roi_image)
        self.auto_roi_btn.clicked.connect(self.auto_detect_roi)
        self.roi_selector.roi_changed.connect(self.on_roi_changed)
        
        layout.addWidget(auto_roi_info)
        layout.addWidget(roi_group)
        layout.addStretch()
        return tab
        
    def create_training_tab(self):
        """훈련 탭"""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        # 훈련 데이터
        data_group = QGroupBox("📂 훈련 데이터")
        data_layout = QVBoxLayout()
        data_group.setLayout(data_layout)
        
        # 폴더 선택
        folder_layout = QHBoxLayout()
        self.folder_label = QLabel("훈련 폴더가 선택되지 않음")
        self.select_folder_btn = QPushButton("📁 훈련 폴더 선택")
        folder_layout.addWidget(self.folder_label)
        folder_layout.addWidget(self.select_folder_btn)
        data_layout.addLayout(folder_layout)
        
        # 이미지 개수 제한
        limit_layout = QHBoxLayout()
        self.limit_checkbox = QCheckBox("훈련 이미지 개수 제한:")
        self.limit_spin = QSpinBox()
        self.limit_spin.setRange(10, 10000)
        self.limit_spin.setValue(500)
        self.limit_spin.setEnabled(False)
        limit_layout.addWidget(self.limit_checkbox)
        limit_layout.addWidget(self.limit_spin)
        data_layout.addLayout(limit_layout)
        
        self.limit_checkbox.toggled.connect(self.limit_spin.setEnabled)
        
        layout.addWidget(data_group)
        
        # 훈련 제어
        training_group = QGroupBox("🎓 GPU 훈련 제어")
        training_layout = QVBoxLayout()
        training_group.setLayout(training_layout)
        
        # 모델 저장 경로
        save_layout = QHBoxLayout()
        self.model_save_label = QLabel("자동 생성: deep_model_YYYYMMDD_HHMMSS.pkl")
        self.select_save_btn = QPushButton("💾 저장 경로 변경")
        save_layout.addWidget(QLabel("모델 저장:"))
        save_layout.addWidget(self.model_save_label)
        save_layout.addWidget(self.select_save_btn)
        training_layout.addLayout(save_layout)
        
        # 진행률
        self.training_progress = QProgressBar()
        self.training_status = QLabel("GPU 훈련 대기 중")
        training_layout.addWidget(self.training_status)
        training_layout.addWidget(self.training_progress)
        
        # 훈련 버튼
        self.train_btn = QPushButton("🚀 GPU 가속 훈련 시작")
        self.train_btn.setMinimumHeight(50)
        self.train_btn.setStyleSheet("font-size: 14px; font-weight: bold; background-color: #4CAF50;")
        training_layout.addWidget(self.train_btn)
        
        layout.addWidget(training_group)
        
        # 이벤트 연결
        self.select_folder_btn.clicked.connect(self.select_training_folder)
        self.select_save_btn.clicked.connect(self.select_model_save_path)
        self.train_btn.clicked.connect(self.start_gpu_training)
        
        layout.addStretch()
        return tab
        
    def create_analysis_tab(self):
        """분석 탭"""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        # 모델 로드
        model_group = QGroupBox("📂 훈련된 모델 로드")
        model_layout = QVBoxLayout()
        model_group.setLayout(model_layout)
        
        load_layout = QHBoxLayout()
        self.model_path_label = QLabel("모델이 로드되지 않음")
        self.load_model_btn = QPushButton("📂 모델 로드")
        load_layout.addWidget(self.model_path_label)
        load_layout.addWidget(self.load_model_btn)
        model_layout.addLayout(load_layout)
        
        layout.addWidget(model_group)
        
        # 테스트 데이터
        test_group = QGroupBox("🔍 테스트 데이터")
        test_layout = QVBoxLayout()
        test_group.setLayout(test_layout)
        
        # 테스트 폴더
        test_folder_layout = QHBoxLayout()
        self.test_folder_label = QLabel("테스트 폴더가 선택되지 않음")
        self.select_test_folder_btn = QPushButton("📁 테스트 폴더 선택")
        test_folder_layout.addWidget(self.test_folder_label)
        test_folder_layout.addWidget(self.select_test_folder_btn)
        test_layout.addLayout(test_folder_layout)
        
        # 분석 옵션
        options_layout = QHBoxLayout()
        self.heatmap_checkbox = QCheckBox("히트맵 생성")
        self.heatmap_checkbox.setChecked(True)
        options_layout.addWidget(self.heatmap_checkbox)
        test_layout.addLayout(options_layout)
        
        layout.addWidget(test_group)
        
        # 분석 제어
        analysis_group = QGroupBox("📊 GPU 분석 제어")
        analysis_layout = QVBoxLayout()
        analysis_group.setLayout(analysis_layout)
        
        # 진행률
        self.analysis_progress = QProgressBar()
        self.analysis_status = QLabel("GPU 분석 대기 중")
        analysis_layout.addWidget(self.analysis_status)
        analysis_layout.addWidget(self.analysis_progress)
        
        # 분석 버튼
        analysis_btn_layout = QHBoxLayout()
        self.analyze_btn = QPushButton("🔍 GPU 가속 분석 시작")
        self.clear_results_btn = QPushButton("🧹 결과 지우기")
        analysis_btn_layout.addWidget(self.analyze_btn)
        analysis_btn_layout.addWidget(self.clear_results_btn)
        analysis_layout.addLayout(analysis_btn_layout)
        
        layout.addWidget(analysis_group)
        
        # 결과 요약
        summary_group = QGroupBox("📈 결과 요약")
        summary_layout = QVBoxLayout()
        summary_group.setLayout(summary_layout)
        
        self.summary_label = QLabel("분석이 수행되지 않음")
        self.summary_label.setWordWrap(True)
        summary_layout.addWidget(self.summary_label)
        
        layout.addWidget(summary_group)
        
        # 이벤트 연결
        self.load_model_btn.clicked.connect(self.load_trained_model)
        self.select_test_folder_btn.clicked.connect(self.select_test_folder)
        self.analyze_btn.clicked.connect(self.start_gpu_analysis)
        self.clear_results_btn.clicked.connect(self.clear_results)
        
        layout.addStretch()
        return tab
        
    def update_gpu_status(self):
        """GPU 상태 업데이트"""
        if not GPU_AVAILABLE:
            self.gpu_status_label.setText("❌ GPU 모듈 사용 불가")
            return
            
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                gpu_memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
                gpu_memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
                gpu_memory_free = gpu_memory_total - gpu_memory_reserved
                
                # 메모리 사용률 계산
                memory_usage_percent = (gpu_memory_allocated / gpu_memory_total) * 100
                
                status_text = f"""✅ GPU 사용 가능
🔥 이름: {gpu_name}
💾 전체 메모리: {gpu_memory_total:.1f} GB
📊 사용 메모리: {gpu_memory_allocated:.2f} GB ({memory_usage_percent:.1f}%)
📦 예약 메모리: {gpu_memory_reserved:.2f} GB
🆓 여유 메모리: {gpu_memory_free:.2f} GB
⚡ CUDA 버전: {torch.version.cuda}"""
                self.gpu_status_label.setText(status_text)
                
                # 메모리 사용량이 80% 이상이면 경고
                if memory_usage_percent > 80:
                    self.gpu_status_label.setStyleSheet("color: orange; font-weight: bold;")
                elif memory_usage_percent > 50:
                    self.gpu_status_label.setStyleSheet("color: yellow;")
                else:
                    self.gpu_status_label.setStyleSheet("color: lightgreen;")
            else:
                self.gpu_status_label.setText("❌ CUDA GPU 사용 불가")
        except Exception as e:
            self.gpu_status_label.setText(f"❌ GPU 상태 확인 실패: {e}")
            
    def preview_resolution_effect(self):
        """해상도 효과 미리보기"""
        current_ratio = self.resolution_ratio_spin.value()
        
        # 예상 메모리 사용량 계산 (대략적인 추정)
        base_memory = 2.0  # 기본 모델 메모리 (GB)
        ratio_factor = (current_ratio / 35.0)  # 35%를 기준으로 함
        estimated_memory = base_memory * (ratio_factor ** 1.5)  # 비선형 증가
        
        # 예상 패치 개수 계산
        base_patches = 15  # 35%일 때 평균 패치 개수
        estimated_patches = int(base_patches * ratio_factor)
        
        # 예상 처리 시간 (패치 개수와 해상도에 비례)
        base_time = 0.05  # 35%일 때 기준 시간 (초/이미지)
        estimated_time = base_time * estimated_patches * ratio_factor
        
        # 해상도 품질 설명
        if current_ratio >= 70:
            quality = "최고 품질 - 미세한 결함까지 감지 가능"
        elif current_ratio >= 50:
            quality = "고품질 - 대부분의 결함 감지 가능"
        elif current_ratio >= 35:
            quality = "중고품질 - 일반적인 결함 감지에 최적"
        elif current_ratio >= 25:
            quality = "중품질 - 큰 결함 위주로 감지"
        else:
            quality = "저품질 - 명확한 결함만 감지"
            
        preview_message = f"""📐 해상도 비율 {current_ratio}% 효과 미리보기
        
🎯 검출 품질: {quality}
📊 예상 메모리 사용량: {estimated_memory:.1f} GB
🔢 예상 패치 개수: {estimated_patches}개 (이미지당)
⏱️ 예상 처리 시간: {estimated_time:.3f}초 (이미지당)

💡 추천 사용 시나리오:
• 20-30%: 빠른 스크리닝, 메모리 제한 환경
• 35-50%: 일반적인 품질 검사 (권장)
• 50-70%: 정밀 검사, 미세 결함 탐지
• 70-100%: 최고 정밀도, 연구용"""

        QMessageBox.information(self, "해상도 효과 미리보기", preview_message)
        self.log_message(f"🔍 해상도 {current_ratio}% 효과 미리보기: {quality}")
        
    def on_resolution_ratio_changed(self, value):
        """해상도 비율 변경 시 호출"""
        if value >= 70:
            info_text = f"💡 {value}%: 최고 해상도, 높은 메모리 사용량"
            color = "#FFB366"  # 주황색
        elif value >= 50:
            info_text = f"💡 {value}%: 고해상도 유지, 메모리 사용량 높음"
            color = "#FFFF66"  # 노란색
        elif value >= 35:
            info_text = f"💡 {value}%: 고해상도 유지, 메모리 사용량 중간"
            color = "#90EE90"  # 연녹색
        elif value >= 25:
            info_text = f"💡 {value}%: 중간 해상도, 메모리 절약"
            color = "#87CEEB"  # 하늘색
        else:
            info_text = f"💡 {value}%: 낮은 해상도, 최대 메모리 절약"
            color = "#DDA0DD"  # 연보라
            
        self.resolution_info.setText(info_text)
        self.resolution_info.setStyleSheet(f"color: {color}; font-size: 10px; font-weight: bold;")
        
        # 이미 로드된 모델이 있으면 해상도 비율 업데이트
        if hasattr(self, 'gpu_detector') and self.gpu_detector:
            resolution_ratio = value / 100.0
            if hasattr(self.gpu_detector, 'set_resolution_ratio'):
                self.gpu_detector.set_resolution_ratio(resolution_ratio)
            else:
                self.gpu_detector.resolution_ratio = resolution_ratio
            self.log_message(f"📐 해상도 비율 변경: {value}%")
            
    def clear_gpu_memory(self):
        """GPU 메모리 강제 정리"""
        if self.gpu_available:
            try:
                import torch
                import gc
                
                # 현재 메모리 사용량 확인
                before_allocated = torch.cuda.memory_allocated() / 1024**3
                before_reserved = torch.cuda.memory_reserved() / 1024**3
                
                # 메모리 정리 수행
                if hasattr(self, 'gpu_detector') and self.gpu_detector:
                    del self.gpu_detector
                    self.gpu_detector = None
                    
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                
                # 정리 후 메모리 확인
                after_allocated = torch.cuda.memory_allocated() / 1024**3
                after_reserved = torch.cuda.memory_reserved() / 1024**3
                
                message = f"""🧹 GPU 메모리 정리 완료!
                
📊 정리 전: {before_allocated:.2f}GB 사용, {before_reserved:.2f}GB 예약
📊 정리 후: {after_allocated:.2f}GB 사용, {after_reserved:.2f}GB 예약
💾 절약량: {before_allocated - after_allocated:.2f}GB"""
                
                self.log_message("🧹 GPU 메모리 강제 정리 완료")
                self.update_gpu_status()
                QMessageBox.information(self, "메모리 정리", message)
                
            except Exception as e:
                self.log_message(f"❌ GPU 메모리 정리 실패: {e}")
                QMessageBox.warning(self, "오류", f"GPU 메모리 정리 실패: {e}")
        else:
            QMessageBox.information(self, "안내", "GPU가 사용 불가능합니다.")
            
    def get_current_config(self) -> Dict:
        """현재 UI 설정을 딕셔너리로 변환"""
        batch_size = self.batch_size_spin.value()
        
        # 메모리 최적화 모드일 때 배치 크기 추가 제한
        if self.memory_optimize_checkbox.isChecked():
            batch_size = max(2, min(6, batch_size))
        
        # 해상도 비율 가져오기
        resolution_ratio = self.resolution_ratio_spin.value() / 100.0  # 퍼센트를 소수로 변환
            
        config = {
            'model_name': 'patchcore',
            'use_gpu': self.gpu_checkbox.isChecked() and self.gpu_available,
            'batch_size': batch_size,
            'resolution_ratio': resolution_ratio
        }
        
        return config
        
    def select_roi_image(self):
        """ROI 설정용 이미지 선택"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "ROI 설정용 이미지 선택", "", 
            "Image Files (*.jpg *.jpeg *.png *.bmp *.tiff);;All Files (*)")
        if file_path:
            if self.roi_selector.load_image(file_path):
                self.roi_image_label.setText(Path(file_path).name)
                self.log_message(f"📁 ROI 이미지 로드: {Path(file_path).name}")
            else:
                QMessageBox.warning(self, "오류", "이미지 로드에 실패했습니다.")
                
    def auto_detect_roi(self):
        """자동 ROI 감지"""
        if not hasattr(self, 'roi_selector') or not self.roi_selector.image_path:
            QMessageBox.warning(self, "경고", "먼저 ROI 설정용 이미지를 선택해주세요.")
            return
            
        try:
            # get_target_roi_info 함수 사용하여 자동 ROI 감지
            x1, x2, roi_rect = get_target_roi_info(self.roi_selector.image_path)
            
            if roi_rect is not None:
                # ROI 정보 설정
                self.roi_rect = roi_rect
                self.roi_selector.roi_rect = roi_rect
                self.roi_selector.update()
                
                # ROI 정보 표시 업데이트
                self.roi_info_label.setText(f"자동 감지 ROI: ({roi_rect[0]}, {roi_rect[1]}, {roi_rect[2]}, {roi_rect[3]})")
                self.roi_checkbox.setChecked(True)  # 자동으로 ROI 사용 활성화
                
                self.log_message(f"🎯 자동 ROI 감지 성공: {roi_rect}")
                QMessageBox.information(self, "성공", 
                    f"ROI 영역이 자동으로 감지되었습니다!\n"
                    f"위치: ({roi_rect[0]}, {roi_rect[1]})\n"
                    f"크기: {roi_rect[2]} × {roi_rect[3]}")
            else:
                self.log_message("❌ 자동 ROI 감지 실패: 적절한 영역을 찾을 수 없음")
                QMessageBox.warning(self, "실패", 
                    "자동 ROI 감지에 실패했습니다.\n"
                    "수동으로 ROI 영역을 선택해주세요.")
                
        except Exception as e:
            error_msg = f"자동 ROI 감지 중 오류 발생: {str(e)}"
            self.log_message(f"❌ {error_msg}")
            QMessageBox.critical(self, "오류", error_msg)
            
    def on_roi_changed(self, roi_rect):
        """ROI 변경 시 호출"""
        self.roi_rect = roi_rect
        self.roi_info_label.setText(f"수동 설정 ROI: ({roi_rect[0]}, {roi_rect[1]}, {roi_rect[2]}, {roi_rect[3]})")
        self.log_message(f"✂️ 수동 ROI 설정: {roi_rect}")
        
    def select_training_folder(self):
        """훈련 폴더 선택"""
        folder = QFileDialog.getExistingDirectory(self, "훈련 폴더 선택")
        if folder:
            self.folder_label.setText(folder)
            
            # 이미지 개수 확인
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
                image_files.extend(glob.glob(os.path.join(folder, ext)))
                image_files.extend(glob.glob(os.path.join(folder, ext.upper())))
                
            self.log_message(f"📁 훈련 폴더: {Path(folder).name} ({len(image_files)}개 이미지)")
            
    def select_model_save_path(self):
        """모델 저장 경로 선택"""
        # 현재 작업 디렉토리를 기본 경로로 설정
        default_path = os.getcwd()
        
        # 폴더 선택 대화상자
        folder = QFileDialog.getExistingDirectory(
            self, "모델 저장 폴더 선택", default_path)
        
        if folder:
            # 폴더 쓰기 권한 확인
            test_file = os.path.join(folder, "test_write_permission.tmp")
            try:
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
                
                # 선택된 폴더 저장
                self.selected_save_folder = folder
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"deep_model_{timestamp}.pkl"
                full_path = os.path.join(folder, filename)
                
                # 라벨 업데이트 (폴더 경로와 파일명 표시)
                self.model_save_label.setText(f"📁 {Path(folder).name}\\{filename}")
                self.model_save_label.setToolTip(full_path)  # 전체 경로를 툴팁으로 표시
                
                self.log_message(f"💾 저장 경로 설정: {full_path}")
                
            except Exception as e:
                QMessageBox.warning(self, "경고", f"선택한 폴더에 쓰기 권한이 없습니다:\n{folder}\n\n오류: {e}")
                self.log_message(f"❌ 저장 경로 설정 실패: 쓰기 권한 없음 - {folder}")
            
    def start_gpu_training(self):
        """GPU 훈련 시작"""
        if not GPU_AVAILABLE:
            QMessageBox.warning(self, "오류", "GPU 모듈이 사용 불가합니다.")
            return
            
        # 훈련 폴더 확인
        folder_path = self.folder_label.text()
        if folder_path == "훈련 폴더가 선택되지 않음":
            QMessageBox.warning(self, "오류", "훈련 폴더를 선택해주세요.")
            return
            
        # 이미지 파일 수집
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
            image_files.extend(glob.glob(os.path.join(folder_path, ext)))
            image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
            
        if not image_files:
            QMessageBox.warning(self, "오류", "훈련 이미지가 없습니다.")
            return
            
        # 이미지 개수 제한
        if self.limit_checkbox.isChecked():
            image_files = image_files[:self.limit_spin.value()]
            
        # 모델 저장 경로 결정
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 사용자가 폴더를 선택한 경우
        if hasattr(self, 'selected_save_folder') and self.selected_save_folder:
            filename = f"deep_model_{timestamp}.pkl"
            save_path = os.path.join(self.selected_save_folder, filename)
        else:
            # 기본 경로 (현재 작업 디렉토리)
            save_path = os.path.join(os.getcwd(), f"deep_model_{timestamp}.pkl")
            
        # 설정 가져오기
        config = self.get_current_config()
        self.current_config = config
        
        # ROI 설정
        roi_rect = self.roi_rect if self.roi_checkbox.isChecked() else None
        
        # GPU 메모리 정리
        if self.gpu_available:
            try:
                import torch
                torch.cuda.empty_cache()
            except:
                pass
                
        # 훈련 스레드 시작
        self.training_thread = GPUTrainingThread(config, image_files, save_path, roi_rect)
        self.training_thread.progress_update.connect(self.update_training_progress)
        self.training_thread.training_complete.connect(self.on_training_complete)
        self.training_thread.error_occurred.connect(self.on_training_error)
        
        # UI 상태 변경
        self.train_btn.setEnabled(False)
        self.training_progress.setValue(0)
        
        self.training_thread.start()
        
        # 해상도 비율 정보 로그
        resolution_ratio = getattr(config, 'resolution_ratio', 0.35)
        self.log_message(f"🚀 GPU 훈련 시작: {len(image_files)}개 이미지")
        self.log_message(f"📐 해상도 비율: {resolution_ratio*100:.0f}% (원본 대비)")
        self.log_message(f"✂️ ROI 적용: {'예' if roi_rect else '아니오'}")
        self.log_message(f"🧠 배치 크기: {config.batch_size}")
        self.log_message(f"🧹 메모리 최적화: {'활성화' if self.memory_optimize_checkbox.isChecked() else '비활성화'}")
        
    def update_training_progress(self, value, message):
        """훈련 진행률 업데이트"""
        self.training_progress.setValue(value)
        self.training_status.setText(message)
        self.log_message(message)
        
    def on_training_complete(self, success, message, results):
        """훈련 완료 처리"""
        self.train_btn.setEnabled(True)
        
        if success:
            self.gpu_detector = self.training_thread.detector
            
            # 저장된 모델 경로 정보 표시
            saved_path = results.get('model_saved_path', '알 수 없음')
            saved_dir = os.path.dirname(saved_path)
            saved_filename = os.path.basename(saved_path)
            
            success_msg = f"{message}\n\n"
            success_msg += f"📁 저장 폴더: {saved_dir}\n"
            success_msg += f"📄 파일명: {saved_filename}\n"
            success_msg += f"💾 전체 경로: {saved_path}"
            
            QMessageBox.information(self, "훈련 성공", success_msg)
            
            # 결과 정보 로그
            training_time = results.get('training_time', 0)
            num_images = results.get('num_training_images', 0)
            roi_applied = results.get('roi_applied', False)
            
            self.log_message(f"✅ GPU 훈련 완료: {training_time:.2f}초, "
                           f"{num_images}개 이미지, ROI: {'적용' if roi_applied else '미적용'}")
            self.log_message(f"💾 모델 자동 저장: {saved_filename}")
            self.log_message(f"📁 저장 위치: {saved_dir}")
            
            # 분석 탭의 모델 경로 레이블 업데이트
            try:
                self.model_path_label.setText(saved_filename)
                self.model_path_label.setToolTip(saved_path)  # 전체 경로를 툴팁으로
            except AttributeError:
                pass  # 분석 탭이 아직 생성되지 않은 경우
                
        else:
            QMessageBox.critical(self, "훈련 실패", message)
            
    def on_training_error(self, error_message):
        """훈련 오류 처리"""
        self.train_btn.setEnabled(True)
        QMessageBox.critical(self, "훈련 오류", error_message)
        self.log_message(f"❌ GPU 훈련 오류: {error_message}")
        
    def load_trained_model(self):
        """훈련된 모델 로드"""
        if not GPU_AVAILABLE:
            QMessageBox.warning(self, "오류", "GPU 모듈이 사용 불가합니다.")
            return
            
        path, _ = QFileDialog.getOpenFileName(
            self, "훈련된 모델 로드", "", "Pickle Files (*.pkl);;All Files (*)")
        if path:
            try:
                # 새로운 모듈러 구조를 사용한 모델 로드
                config = {
                    'resolution_ratio': self.resolution_ratio_spin.value() / 100.0
                }
                
                self.gpu_detector = create_detector(
                    model_name='patchcore',
                    **config
                )
                
                # 모델 파일 로드
                if self.gpu_detector.load_model(path):
                    self.model_path_label.setText(Path(path).name)
                    self.log_message(f"📂 모델 로드 완료: {Path(path).name}")
                    self.log_message(f"📐 해상도 비율 설정: {self.resolution_ratio_spin.value()}%")
                    QMessageBox.information(self, "성공", "모델이 성공적으로 로드되었습니다.")
                else:
                    QMessageBox.warning(self, "오류", "모델 로드에 실패했습니다.")
                
            except Exception as e:
                QMessageBox.critical(self, "오류", f"모델 로드 실패: {str(e)}")
                self.log_message(f"❌ 모델 로드 실패: {e}")
                
    def select_test_folder(self):
        """테스트 폴더 선택"""
        folder = QFileDialog.getExistingDirectory(self, "테스트 폴더 선택")
        if folder:
            self.test_folder_label.setText(folder)
            
            # 이미지 개수 확인
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
                image_files.extend(glob.glob(os.path.join(folder, ext)))
                image_files.extend(glob.glob(os.path.join(folder, ext.upper())))
                
            self.log_message(f"📁 테스트 폴더: {Path(folder).name} ({len(image_files)}개 이미지)")
            
    def start_gpu_analysis(self):
        """GPU 분석 시작"""
        if not self.gpu_detector or not hasattr(self.gpu_detector, 'model'):
            QMessageBox.warning(self, "오류", "훈련된 모델을 먼저 로드해주세요.")
            return
            
        # 테스트 폴더 확인
        folder_path = self.test_folder_label.text()
        if folder_path == "테스트 폴더가 선택되지 않음":
            QMessageBox.warning(self, "오류", "테스트 폴더를 선택해주세요.")
            return
            
        # 테스트 이미지 수집
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
            image_files.extend(glob.glob(os.path.join(folder_path, ext)))
            image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
            
        if not image_files:
            QMessageBox.warning(self, "오류", "테스트 이미지가 없습니다.")
            return
            
        # ROI 설정
        roi_rect = self.roi_rect if self.roi_checkbox.isChecked() else None
        
        # GPU 메모리 정리
        if self.gpu_available:
            try:
                import torch
                torch.cuda.empty_cache()
            except:
                pass
                
        # 분석 스레드 시작
        create_heatmaps = self.heatmap_checkbox.isChecked()
        
        # 해상도 비율 설정 (현재 GUI 설정 사용)
        if hasattr(self.gpu_detector, 'set_resolution_ratio'):
            resolution_ratio = self.resolution_ratio_spin.value() / 100.0
            self.gpu_detector.set_resolution_ratio(resolution_ratio)
        elif self.gpu_detector:
            resolution_ratio = self.resolution_ratio_spin.value() / 100.0
            self.gpu_detector.resolution_ratio = resolution_ratio
        
        self.analysis_thread = GPUAnalysisThread(
            self.gpu_detector, image_files, create_heatmaps, roi_rect)
        
        self.analysis_thread.progress_update.connect(self.update_analysis_progress)
        self.analysis_thread.analysis_complete.connect(self.on_analysis_complete)
        self.analysis_thread.error_occurred.connect(self.on_analysis_error)
        
        # UI 상태 변경
        self.analyze_btn.setEnabled(False)
        self.analysis_progress.setValue(0)
        
        self.analysis_thread.start()
        
        # 분석 시작 로그 (상세 정보 포함)
        resolution_ratio = self.resolution_ratio_spin.value()
        self.log_message(f"🔍 GPU 분석 시작: {len(image_files)}개 이미지")
        self.log_message(f"📐 해상도 비율: {resolution_ratio}% (원본 대비)")
        self.log_message(f"🔥 히트맵 생성: {'활성화' if create_heatmaps else '비활성화'}")
        self.log_message(f"✂️ ROI 적용: {'예' if roi_rect else '아니오'}")
        
    def update_analysis_progress(self, value, message):
        """분석 진행률 업데이트"""
        self.analysis_progress.setValue(value)
        self.analysis_status.setText(message)
        
    def on_analysis_complete(self, results):
        """분석 완료 처리"""
        self.analyze_btn.setEnabled(True)
        self.analysis_results = results
        
        # 결과 요약
        total = results['total_images']
        anomalies = len(results['anomaly_images'])
        normal = len(results['normal_images'])
        
        # 평균 추론 시간 계산
        details = results['analysis_details']
        avg_inference_time = np.mean([d['inference_time'] for d in details]) if details else 0
        
        roi_text = " (ROI 적용)" if results.get('roi_applied') else ""
        
        summary = f"""📊 GPU 분석 결과{roi_text}:
• 전체 이미지: {total:,}개
• 정상 이미지: {normal:,}개 ({normal/total*100:.1f}%)
• 이상 이미지: {anomalies:,}개 ({anomalies/total*100:.1f}%)
• 평균 추론 시간: {avg_inference_time:.3f}초
• GPU 가속: {'활성화' if self.gpu_available else '비활성화'}"""
        
        self.summary_label.setText(summary)
        self.log_message("✅ GPU 분석 완료!")
        
        # 결과 시각화
        self.visualization_widget.plot_gpu_analysis_results(results)
        
    def on_analysis_error(self, error_message):
        """분석 오류 처리"""
        self.analyze_btn.setEnabled(True)
        QMessageBox.critical(self, "분석 오류", error_message)
        self.log_message(f"❌ GPU 분석 오류: {error_message}")
        
    def clear_results(self):
        """결과 지우기"""
        self.analysis_results = None
        self.summary_label.setText("분석이 수행되지 않음")
        self.visualization_widget.figure.clear()
        self.visualization_widget.canvas.draw()
        self.log_message("🧹 결과 지움")
        
    def setup_logging(self):
        """로깅 설정"""
        self.log_timer = QTimer()
        self.log_timer.timeout.connect(self.flush_log)
        self.log_timer.start(1000)  # 1초마다 갱신
        
    def log_message(self, message):
        """로그 메시지 추가"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        
        self.log_text.append(formatted_message)
        self.log_text.ensureCursorVisible()
        
        # 파일 로깅
        logging.info(message)
        
    def flush_log(self):
        """로그 정리"""
        # 로그가 너무 길면 일부 제거
        if self.log_text.document().blockCount() > 1000:
            cursor = self.log_text.textCursor()
            cursor.movePosition(cursor.Start)
            cursor.movePosition(cursor.Down, cursor.KeepAnchor, 200)
            cursor.removeSelectedText()
            
    def apply_dark_theme(self):
        """다크 테마 적용"""
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
        dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ToolTipBase, QColor(0, 0, 0))
        dark_palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.Text, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
        dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
        
        self.setPalette(dark_palette)
        
    def closeEvent(self, event):
        """프로그램 종료 시 정리"""
        # GPU 메모리 정리
        if self.gpu_available:
            try:
                import torch
                torch.cuda.empty_cache()
            except:
                pass
                
        self.log_message("🛑 GPU 이상 감지 시스템 종료")
        event.accept()

def main():
    """메인 함수"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # 모던한 스타일
    
    # 메인 윈도우 생성
    window = GPUAnomalyDetectorUI()
    window.show()
    
    print("🚀 GPU 가속 이상 감지 시스템 시작!")
    print(f"💻 GPU 사용 가능: {window.gpu_available}")
    
    try:
        sys.exit(app.exec_())
    except KeyboardInterrupt:
        print("\n🛑 사용자에 의해 종료됨")

if __name__ == "__main__":
    main()
