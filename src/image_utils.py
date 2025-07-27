"""
이미지 처리 유틸리티 함수들
"""
import os
import re
import numpy as np
import cv2
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtGui import QPixmap, QImage

def natural_keys(text):
    """자연스러운 숫자 정렬을 위한 키 함수"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', text)]

def read_image_with_fallback(path):
    """이미지 파일을 읽어서 QPixmap으로 반환"""
    pixmap = QPixmap(path)
    if not pixmap.isNull():
        return pixmap
    img_cv = cv2.imread(path)
    if img_cv is None:
        return None
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    h, w, ch = img_rgb.shape
    bytes_per_line = ch * w
    qimage = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimage)

def calculate_lab_l_value(image_path):
    """이미지의 중앙 ROI 영역에서 LAB L-값 평균 계산"""
    img = cv2.imread(image_path)
    if img is None:
        return None, None
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    roi_size = 250
    x1 = max(cx - roi_size, 0)
    y1 = max(cy - roi_size, 0)
    x2 = min(cx + roi_size, w)
    y2 = min(cy + roi_size, h)
    roi = img[y1:y2, x1:x2]
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2Lab)
    l_mean = float(np.mean(lab[:, :, 0]))
    return l_mean, (x1, y1, x2 - x1, y2 - y1)

def get_target_roi_info(image_path):
    img = cv2.imread(image_path)
    
    if img is None:
        return None, None
    
    H, W = img.shape[:2]
    N = max(1, int(W * 0.02))  # 샘플 구간 폭: 전체 가로의 left_ratio(예: 2%)
    
    # 1. 샘플 구간 설정
    left_bg = img[:, :N]            # 왼쪽 샘플(주로 배경)
    right_bg = img[:, -N:]          # 오른쪽 샘플(보조, 필요시 활용)
    center = img[:, N:-N] if N < W//2 else img  # 중앙(제품)
    
    # 2. 밝기 평균값 산출
    left_mean = np.mean(left_bg)
    right_mean = np.mean(right_bg)
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

def qpixmap_to_cvimg(pixmap):
    """QPixmap을 OpenCV 이미지 배열로 변환"""
    qimg = pixmap.toImage().convertToFormat(QImage.Format_RGB888)
    w, h = qimg.width(), qimg.height()
    ptr = qimg.bits()
    ptr.setsize(h * w * 3)
    arr = np.array(ptr, dtype=np.uint8).reshape(h, w, 3)
    return arr

def apply_lab_l_adjustment(pixmap, l_delta):
    """이미지에 LAB L-값 조정 적용"""
    if pixmap is None:
        return None
    
    img = qpixmap_to_cvimg(pixmap)
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    l, a, b = cv2.split(img_lab)
    l = np.clip(l + l_delta, 0, 255).astype(np.uint8)
    img_new = cv2.merge([l, a, b])
    img_rgb = cv2.cvtColor(img_new, cv2.COLOR_Lab2RGB)
    qimg = QImage(img_rgb.data, img_rgb.shape[1], img_rgb.shape[0], img_rgb.shape[1]*3, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)

def get_image_files_from_folder(folder_path):
    """폴더에서 이미지 파일 목록을 가져오기"""
    if not folder_path:
        return []
    
    try:
        files = [
            os.path.join(folder_path, f)
            for f in sorted(os.listdir(folder_path), key=natural_keys)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))
        ]
        return files
    except Exception as e:
        return []

# 6가지 컬러 대비 기법 함수들
def apply_hist_eq_color(img_rgb):
    """히스토그램 평활화 (Histogram Equalization)"""
    img_yuv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

def apply_clahe_color(img_rgb):
    """CLAHE (국소 히스토그램 평활화)"""
    img_yuv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

def apply_gamma_color(img_rgb, gamma=1.5):
    """감마 보정 (Gamma Correction)"""
    gamma_corr = np.power(img_rgb / 255.0, gamma)
    return np.uint8(gamma_corr * 255)

def apply_stretch_color(img_rgb):
    """명암 스트레칭 (Histogram Stretching)"""
    img_yuv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV)
    y = img_yuv[:,:,0]
    in_min = np.percentile(y, 2)
    in_max = np.percentile(y, 98)
    y_stretch = (y - in_min) * 255.0 / (in_max - in_min)
    y_stretch = np.clip(y_stretch, 0, 255).astype(np.uint8)
    img_yuv[:,:,0] = y_stretch
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

def apply_unsharp_color(img_rgb):
    """Unsharp Masking (날카롭게 하면서 대비도 향상)"""
    img_yuv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV)
    y = img_yuv[:,:,0]
    blur = cv2.GaussianBlur(y, (0,0), 3)
    y_unsharp = cv2.addWeighted(y, 1.5, blur, -0.5, 0)
    img_yuv[:,:,0] = np.clip(y_unsharp, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

def apply_contrast_enhancement(pixmap, method="original", roi_rect=None):
    """컬러 대비 기법 적용 (ROI 영역에만 적용)"""
    if pixmap is None or method == "original":
        return pixmap
    
    img = qpixmap_to_cvimg(pixmap)
    
    # ROI가 있으면 해당 영역에만 대비 기법 적용
    if roi_rect is not None:
        roi_x, roi_y, roi_w, roi_h = [int(v) for v in roi_rect]
        
        # ROI 영역 추출
        roi_img = img[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w].copy()
        
        # ROI 영역에만 대비 기법 적용
        if method == "histogram_eq":
            roi_enhanced = apply_hist_eq_color(roi_img)
        elif method == "clahe":
            roi_enhanced = apply_clahe_color(roi_img)
        elif method == "gamma":
            roi_enhanced = apply_gamma_color(roi_img)
        elif method == "stretch":
            roi_enhanced = apply_stretch_color(roi_img)
        elif method == "unsharp":
            roi_enhanced = apply_unsharp_color(roi_img)
        else:
            roi_enhanced = roi_img
        
        # 원본 이미지에 처리된 ROI 영역 다시 붙이기
        img_result = img.copy()
        img_result[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w] = roi_enhanced
        
    else:
        # ROI가 없으면 전체 이미지에 적용 (기존 방식)
        if method == "histogram_eq":
            img_result = apply_hist_eq_color(img)
        elif method == "clahe":
            img_result = apply_clahe_color(img)
        elif method == "gamma":
            img_result = apply_gamma_color(img)
        elif method == "stretch":
            img_result = apply_stretch_color(img)
        elif method == "unsharp":
            img_result = apply_unsharp_color(img)
        else:
            img_result = img
    
    qimg = QImage(img_result.data, img_result.shape[1], img_result.shape[0], 
                  img_result.shape[1]*3, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)
