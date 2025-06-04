"""
画像処理ユーティリティ

共通の画像処理機能を提供し、各解析モジュール間での重複を削減
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Any
import logging

class ImageUtils:
    """画像処理ユーティリティクラス"""
    
    @staticmethod
    def normalize_image(image: np.ndarray) -> np.ndarray:
        """
        画像を0-1の範囲に正規化
        
        Args:
            image: 入力画像
            
        Returns:
            正規化された画像
        """
        if image.dtype == np.uint8:
            return image.astype(np.float64) / 255.0
        elif image.max() > 1.0:
            return image / 255.0
        return image.astype(np.float64)
    
    @staticmethod
    def to_uint8(image: np.ndarray) -> np.ndarray:
        """
        画像をuint8形式に変換
        
        Args:
            image: 入力画像 (0-1 float または 0-255 uint8)
            
        Returns:
            uint8形式の画像
        """
        if image.dtype == np.uint8:
            return image
        return (image * 255).astype(np.uint8)
    
    @staticmethod
    def convert_color_space(image: np.ndarray, conversion: int) -> np.ndarray:
        """
        色空間変換（エラーハンドリング付き）
        
        Args:
            image: 入力画像
            conversion: OpenCVの色変換コード
            
        Returns:
            変換された画像
        """
        try:
            uint8_image = ImageUtils.to_uint8(image)
            return cv2.cvtColor(uint8_image, conversion)
        except cv2.error as e:
            logging.warning(f"色空間変換エラー: {e}")
            return image
    
    @staticmethod
    def get_optimal_sample_rate(image_shape: Tuple[int, int], max_pixels: int = 10000) -> int:
        """
        最適なサンプリングレートを計算
        
        Args:
            image_shape: 画像のサイズ (height, width)
            max_pixels: 処理する最大ピクセル数
            
        Returns:
            サンプリングレート
        """
        total_pixels = image_shape[0] * image_shape[1]
        if total_pixels <= max_pixels:
            return 1
        return int(np.sqrt(total_pixels / max_pixels))
    
    @staticmethod
    def adaptive_resize(image: np.ndarray, max_size: int = 2000) -> Tuple[np.ndarray, float]:
        """
        適応的リサイズ（アスペクト比保持）
        
        Args:
            image: 入力画像
            max_size: 最大サイズ
            
        Returns:
            リサイズされた画像とスケール比
        """
        h, w = image.shape[:2]
        max_dim = max(h, w)
        
        if max_dim <= max_size:
            return image, 1.0
        
        scale = max_size / max_dim
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized, scale
    
    @staticmethod
    def extract_channels(image: np.ndarray, color_space: str = 'RGB') -> Dict[str, np.ndarray]:
        """
        色空間チャンネルを抽出
        
        Args:
            image: 入力画像
            color_space: 色空間 ('RGB', 'HSV', 'LAB')
            
        Returns:
            チャンネル辞書
        """
        if color_space == 'RGB':
            if len(image.shape) == 3:
                return {'R': image[:, :, 0], 'G': image[:, :, 1], 'B': image[:, :, 2]}
            else:
                return {'Gray': image}
        
        elif color_space == 'HSV':
            hsv = ImageUtils.convert_color_space(image, cv2.COLOR_RGB2HSV)
            return {'H': hsv[:, :, 0], 'S': hsv[:, :, 1], 'V': hsv[:, :, 2]}
        
        elif color_space == 'LAB':
            lab = ImageUtils.convert_color_space(image, cv2.COLOR_RGB2LAB)
            return {'L': lab[:, :, 0], 'A': lab[:, :, 1], 'B': lab[:, :, 2]}
        
        else:
            raise ValueError(f"サポートされていない色空間: {color_space}")
    
    @staticmethod
    def calculate_image_stats(image: np.ndarray) -> Dict[str, float]:
        """
        画像の基本統計を計算
        
        Args:
            image: 入力画像
            
        Returns:
            統計辞書
        """
        flat_image = image.flatten() if len(image.shape) > 2 else image.flatten()
        
        return {
            'mean': float(np.mean(flat_image)),
            'std': float(np.std(flat_image)),
            'min': float(np.min(flat_image)),
            'max': float(np.max(flat_image)),
            'median': float(np.median(flat_image)),
            'percentile_25': float(np.percentile(flat_image, 25)),
            'percentile_75': float(np.percentile(flat_image, 75))
        }
    
    @staticmethod
    def create_image_mask(image: np.ndarray, threshold: float = 0.1) -> np.ndarray:
        """
        画像マスクを作成（暗すぎる領域を除外）
        
        Args:
            image: 入力画像
            threshold: 閾値
            
        Returns:
            マスク画像
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(ImageUtils.to_uint8(image), cv2.COLOR_RGB2GRAY) / 255.0
        else:
            gray = ImageUtils.normalize_image(image)
        
        return (gray > threshold).astype(np.uint8)
    
    @staticmethod
    def safe_divide(numerator: np.ndarray, denominator: np.ndarray, default: float = 0.0) -> np.ndarray:
        """
        安全な除算（ゼロ除算回避）
        
        Args:
            numerator: 分子
            denominator: 分母
            default: デフォルト値
            
        Returns:
            除算結果
        """
        mask = denominator != 0
        result = np.full_like(numerator, default, dtype=np.float64)
        result[mask] = numerator[mask] / denominator[mask]
        return result

class PerformanceMonitor:
    """パフォーマンス監視クラス"""
    
    def __init__(self):
        self.times = {}
        self.memory_usage = {}
        self.logger = logging.getLogger(__name__)
    
    def start_timer(self, operation: str):
        """タイマー開始"""
        import time
        self.times[operation] = time.time()
    
    def end_timer(self, operation: str):
        """タイマー終了"""
        import time
        if operation in self.times:
            elapsed = time.time() - self.times[operation]
            self.logger.info(f"{operation}: {elapsed:.2f}秒")
            return elapsed
        return 0
    
    def log_memory_usage(self, operation: str):
        """メモリ使用量をログ"""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.memory_usage[operation] = memory_mb
            self.logger.info(f"{operation} メモリ使用量: {memory_mb:.1f}MB")
            return memory_mb
        except ImportError:
            self.logger.warning("psutilが利用できません。メモリ監視をスキップ")
            return 0
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """パフォーマンスサマリーを取得"""
        return {
            'execution_times': self.times.copy(),
            'memory_usage': self.memory_usage.copy(),
            'total_time': sum(self.times.values()),
            'peak_memory': max(self.memory_usage.values()) if self.memory_usage else 0
        }