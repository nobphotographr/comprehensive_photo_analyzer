"""
Image Processor

画像処理基盤モジュール
画像の読み込み、前処理、基本的な変換処理を行う
"""

import cv2
import numpy as np
from PIL import Image, ExifTags
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, Union
import warnings

from utils.logger import get_logger


class ImageProcessor:
    """画像処理基盤クラス"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger("image_processor")
        
        # 処理設定
        self.max_size = config.get("processing", {}).get("max_image_size", [7000, 7000])
        self.analysis_size = config.get("processing", {}).get("analysis_size", [2000, 2000])
        self.resize_for_analysis = config.get("processing", {}).get("resize_for_analysis", True)
        self.preserve_aspect = config.get("processing", {}).get("preserve_aspect_ratio", True)
    
    def load_image_pair(self, original_path: str, processed_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        画像ペアを読み込み
        
        Args:
            original_path: 元画像パス
            processed_path: 処理済み画像パス
        
        Returns:
            (original_image, processed_image): 正規化された画像ペア
        """
        try:
            self.logger.info(f"画像ペア読み込み開始: {original_path} & {processed_path}")
            
            # 画像の読み込み
            original_img = self._load_single_image(original_path)
            processed_img = self._load_single_image(processed_path)
            
            # 互換性チェック
            if not self._validate_compatibility(original_img, processed_img):
                raise ValueError("画像ペアの互換性チェックに失敗しました")
            
            # 解析用に正規化
            original_norm, processed_norm = self._normalize_for_analysis(original_img, processed_img)
            
            self.logger.info(f"画像ペア読み込み完了: {original_norm.shape}")
            return original_norm, processed_norm
            
        except Exception as e:
            self.logger.error(f"画像ペア読み込みエラー: {e}")
            raise
    
    def _load_single_image(self, image_path: str) -> np.ndarray:
        """
        単一画像の読み込み
        
        Args:
            image_path: 画像ファイルパス
        
        Returns:
            読み込まれた画像（RGB形式）
        """
        try:
            path = Path(image_path)
            if not path.exists():
                raise FileNotFoundError(f"画像ファイルが見つかりません: {image_path}")
            
            # ファイル拡張子による読み込み方法の選択
            ext = path.suffix.lower()
            
            if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                # 一般的な画像形式はOpenCVで読み込み
                img = cv2.imread(str(path))
                if img is None:
                    raise ValueError(f"画像の読み込みに失敗しました: {image_path}")
                
                # BGR→RGB変換
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
            elif ext in ['.tiff', '.tif']:
                # TIFF形式はPILで読み込み（16bit対応）
                with Image.open(path) as pil_img:
                    img = np.array(pil_img)
                
                # グレースケールの場合はRGBに変換
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif img.shape[2] == 4:  # RGBA
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            
            else:
                raise ValueError(f"サポートされていない画像形式: {ext}")
            
            # サイズチェック
            height, width = img.shape[:2]
            if width > self.max_size[0] or height > self.max_size[1]:
                self.logger.warning(f"画像サイズが上限を超えています: {width}x{height}")
                img = self._resize_image(img, self.max_size)
            
            self.logger.debug(f"画像読み込み完了: {image_path} ({img.shape})")
            return img
            
        except Exception as e:
            self.logger.error(f"画像読み込みエラー ({image_path}): {e}")
            raise
    
    def _validate_compatibility(self, img1: np.ndarray, img2: np.ndarray) -> bool:
        """
        画像ペアの互換性チェック
        
        Args:
            img1: 画像1
            img2: 画像2
        
        Returns:
            互換性があるかどうか
        """
        try:
            # サイズチェック
            if img1.shape[:2] != img2.shape[:2]:
                self.logger.warning(f"画像サイズが異なります: {img1.shape[:2]} vs {img2.shape[:2]}")
                return False
            
            # チャンネル数チェック
            if len(img1.shape) != len(img2.shape):
                self.logger.warning("画像のチャンネル数が異なります")
                return False
            
            if len(img1.shape) == 3 and img1.shape[2] != img2.shape[2]:
                self.logger.warning("画像のカラーチャンネル数が異なります")
                return False
            
            # データ型チェック
            if img1.dtype != img2.dtype:
                self.logger.info(f"データ型が異なります: {img1.dtype} vs {img2.dtype} - 正規化で統一します")
            
            return True
            
        except Exception as e:
            self.logger.error(f"互換性チェックエラー: {e}")
            return False
    
    def _normalize_for_analysis(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        解析用の正規化処理
        
        Args:
            img1: 画像1
            img2: 画像2
        
        Returns:
            正規化された画像ペア
        """
        try:
            # 解析用にリサイズ
            if self.resize_for_analysis:
                img1 = self._resize_image(img1, self.analysis_size)
                img2 = self._resize_image(img2, self.analysis_size)
            
            # データ型を統一（float32, 0-1範囲）
            img1_norm = self._normalize_dtype(img1)
            img2_norm = self._normalize_dtype(img2)
            
            self.logger.debug(f"正規化完了: {img1_norm.shape}, dtype={img1_norm.dtype}")
            return img1_norm, img2_norm
            
        except Exception as e:
            self.logger.error(f"正規化エラー: {e}")
            raise
    
    def _resize_image(self, img: np.ndarray, target_size: list) -> np.ndarray:
        """
        画像リサイズ（アスペクト比保持オプション）
        
        Args:
            img: 入力画像
            target_size: 目標サイズ [width, height]
        
        Returns:
            リサイズされた画像
        """
        try:
            height, width = img.shape[:2]
            target_width, target_height = target_size
            
            if self.preserve_aspect:
                # アスペクト比を保持してリサイズ
                scale = min(target_width / width, target_height / height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                
                resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
                
                # 中央配置でパディング
                if new_width != target_width or new_height != target_height:
                    padded = np.zeros((target_height, target_width, img.shape[2]), dtype=img.dtype)
                    y_offset = (target_height - new_height) // 2
                    x_offset = (target_width - new_width) // 2
                    padded[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
                    resized = padded
            else:
                # 強制リサイズ
                resized = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
            
            return resized
            
        except Exception as e:
            self.logger.error(f"リサイズエラー: {e}")
            raise
    
    def _normalize_dtype(self, img: np.ndarray) -> np.ndarray:
        """
        データ型正規化（0-1のfloat32に統一）
        
        Args:
            img: 入力画像
        
        Returns:
            正規化された画像
        """
        try:
            if img.dtype == np.uint8:
                return img.astype(np.float32) / 255.0
            elif img.dtype == np.uint16:
                return img.astype(np.float32) / 65535.0
            elif img.dtype == np.float32:
                # すでにfloat32の場合は範囲をチェック
                if img.max() > 1.0:
                    return img / 255.0
                return img
            elif img.dtype == np.float64:
                img = img.astype(np.float32)
                if img.max() > 1.0:
                    return img / 255.0
                return img
            else:
                self.logger.warning(f"未知のデータ型: {img.dtype} - uint8として処理します")
                return img.astype(np.float32) / 255.0
                
        except Exception as e:
            self.logger.error(f"データ型正規化エラー: {e}")
            raise
    
    def extract_metadata(self, image_path: str) -> Dict[str, Any]:
        """
        画像メタデータの抽出
        
        Args:
            image_path: 画像ファイルパス
        
        Returns:
            抽出されたメタデータ
        """
        try:
            metadata = {
                "file_path": image_path,
                "file_size": Path(image_path).stat().st_size,
                "exif": {}
            }
            
            # EXIF情報の抽出
            try:
                with Image.open(image_path) as img:
                    metadata["image_size"] = img.size
                    metadata["image_mode"] = img.mode
                    
                    if hasattr(img, '_getexif') and img._getexif() is not None:
                        exif = img._getexif()
                        for tag_id, value in exif.items():
                            tag = ExifTags.TAGS.get(tag_id, tag_id)
                            metadata["exif"][tag] = value
            
            except Exception as e:
                self.logger.debug(f"EXIF抽出でエラー（通常の動作）: {e}")
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"メタデータ抽出エラー: {e}")
            return {"file_path": image_path, "error": str(e)}
    
    def convert_color_space(self, img: np.ndarray, target_space: str) -> np.ndarray:
        """
        色空間変換
        
        Args:
            img: 入力画像（RGB想定）
            target_space: 目標色空間 ("HSV", "LAB", "XYZ", etc.)
        
        Returns:
            変換された画像
        """
        try:
            if target_space.upper() == "RGB":
                return img
            elif target_space.upper() == "HSV":
                return cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
            elif target_space.upper() == "LAB":
                return cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
            elif target_space.upper() == "XYZ":
                return cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2XYZ).astype(np.float32) / 255.0
            elif target_space.upper() == "GRAY":
                return cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            else:
                raise ValueError(f"サポートされていない色空間: {target_space}")
                
        except Exception as e:
            self.logger.error(f"色空間変換エラー ({target_space}): {e}")
            raise