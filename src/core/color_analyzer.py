"""
Color Analyzer

色彩解析エンジン
RGB/HSV/LAB色空間での統計分析、ヒストグラム比較、色差計算を行う
Phase 1-3の基本色彩解析機能を提供
"""

import numpy as np
import cv2
from scipy import stats
from sklearn.cluster import KMeans
import logging
from typing import Dict, Any, List, Tuple, Optional

from utils.logger import get_logger, AnalysisLogger


class ColorAnalyzer:
    """色彩解析クラス"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger("color_analyzer")
        self.analysis_logger = AnalysisLogger()
        
        # 解析設定
        self.color_spaces = config.get("analysis", {}).get("color_spaces", ["RGB", "HSV", "LAB"])
        self.precision = config.get("analysis", {}).get("precision", "standard")
        
        # ヒストグラム設定
        self.hist_bins = self._get_histogram_bins()
        
    def analyze(self, original_img: np.ndarray, processed_img: np.ndarray) -> Dict[str, Any]:
        """
        色彩解析のメイン実行関数
        
        Args:
            original_img: 元画像
            processed_img: 処理済み画像
        
        Returns:
            色彩解析結果
        """
        try:
            # 画像情報
            image_info = {
                "shape": original_img.shape,
                "dtype": str(original_img.dtype),
                "color_spaces": self.color_spaces
            }
            
            self.analysis_logger.log_analysis_start("color_analysis", image_info)
            
            results = {
                "basic_statistics": {},
                "histograms": {},
                "color_shifts": {},
                "dominant_colors": {},
                "color_distribution": {},
                "quality_metrics": {}
            }
            
            # 各色空間での解析
            for color_space in self.color_spaces:
                self.logger.debug(f"色空間解析開始: {color_space}")
                
                # 色空間変換
                orig_converted = self._convert_color_space(original_img, color_space)
                proc_converted = self._convert_color_space(processed_img, color_space)
                
                # 基本統計量
                results["basic_statistics"][color_space] = self._calculate_basic_statistics(
                    orig_converted, proc_converted, color_space
                )
                
                # ヒストグラム解析
                results["histograms"][color_space] = self._analyze_histograms(
                    orig_converted, proc_converted, color_space
                )
                
                # 色変化解析
                results["color_shifts"][color_space] = self._analyze_color_shifts(
                    orig_converted, proc_converted, color_space
                )
            
            # RGB空間での詳細解析
            if "RGB" in self.color_spaces:
                results["dominant_colors"] = self._analyze_dominant_colors(original_img, processed_img)
                results["color_distribution"] = self._analyze_color_distribution(original_img, processed_img)
            
            # 品質指標
            results["quality_metrics"] = self._calculate_quality_metrics(original_img, processed_img)
            
            # サマリー生成
            results["summary"] = self._generate_summary(results)
            
            self.analysis_logger.log_analysis_end("color_analysis", results["summary"])
            return results
            
        except Exception as e:
            self.analysis_logger.log_analysis_error("color_analysis", e)
            raise
    
    def _get_histogram_bins(self) -> int:
        """精度設定に基づくヒストグラムビン数を取得"""
        precision_bins = {
            "standard": 64,
            "high": 128,
            "ultra": 256
        }
        return precision_bins.get(self.precision, 64)
    
    def _convert_color_space(self, img: np.ndarray, color_space: str) -> np.ndarray:
        """色空間変換"""
        try:
            if color_space.upper() == "RGB":
                return img
            elif color_space.upper() == "HSV":
                return cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
            elif color_space.upper() == "LAB":
                # LAB色空間は特別な正規化が必要
                lab = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
                # L: 0-100, a,b: -128 to 127 -> 正規化
                lab[:,:,0] = lab[:,:,0] / 100.0  # L channel
                lab[:,:,1] = (lab[:,:,1] + 128) / 255.0  # a channel
                lab[:,:,2] = (lab[:,:,2] + 128) / 255.0  # b channel
                return lab
            else:
                raise ValueError(f"サポートされていない色空間: {color_space}")
                
        except Exception as e:
            self.logger.error(f"色空間変換エラー ({color_space}): {e}")
            raise
    
    def _calculate_basic_statistics(self, orig_img: np.ndarray, proc_img: np.ndarray, color_space: str) -> Dict[str, Any]:
        """基本統計量の計算"""
        try:
            stats_result = {}
            
            channels = self._get_channel_names(color_space)
            
            for i, channel in enumerate(channels):
                orig_channel = orig_img[:,:,i] if len(orig_img.shape) == 3 else orig_img
                proc_channel = proc_img[:,:,i] if len(proc_img.shape) == 3 else proc_img
                
                # 基本統計量
                orig_stats = {
                    "mean": float(np.mean(orig_channel)),
                    "std": float(np.std(orig_channel)),
                    "min": float(np.min(orig_channel)),
                    "max": float(np.max(orig_channel)),
                    "median": float(np.median(orig_channel)),
                    "percentile_25": float(np.percentile(orig_channel, 25)),
                    "percentile_75": float(np.percentile(orig_channel, 75))
                }
                
                proc_stats = {
                    "mean": float(np.mean(proc_channel)),
                    "std": float(np.std(proc_channel)),
                    "min": float(np.min(proc_channel)),
                    "max": float(np.max(proc_channel)),
                    "median": float(np.median(proc_channel)),
                    "percentile_25": float(np.percentile(proc_channel, 25)),
                    "percentile_75": float(np.percentile(proc_channel, 75))
                }
                
                # 変化量
                changes = {
                    "mean_change": proc_stats["mean"] - orig_stats["mean"],
                    "std_change": proc_stats["std"] - orig_stats["std"],
                    "range_change": (proc_stats["max"] - proc_stats["min"]) - (orig_stats["max"] - orig_stats["min"]),
                    "median_change": proc_stats["median"] - orig_stats["median"]
                }
                
                stats_result[channel] = {
                    "original": orig_stats,
                    "processed": proc_stats,
                    "changes": changes
                }
            
            return stats_result
            
        except Exception as e:
            self.logger.error(f"基本統計量計算エラー: {e}")
            raise
    
    def _analyze_histograms(self, orig_img: np.ndarray, proc_img: np.ndarray, color_space: str) -> Dict[str, Any]:
        """ヒストグラム解析"""
        try:
            hist_result = {}
            channels = self._get_channel_names(color_space)
            
            for i, channel in enumerate(channels):
                orig_channel = orig_img[:,:,i] if len(orig_img.shape) == 3 else orig_img
                proc_channel = proc_img[:,:,i] if len(proc_img.shape) == 3 else proc_img
                
                # ヒストグラム計算
                orig_hist, bins = np.histogram(orig_channel.flatten(), bins=self.hist_bins, range=(0, 1))
                proc_hist, _ = np.histogram(proc_channel.flatten(), bins=self.hist_bins, range=(0, 1))
                
                # 正規化
                orig_hist_norm = orig_hist / np.sum(orig_hist)
                proc_hist_norm = proc_hist / np.sum(proc_hist)
                
                # ヒストグラム比較指標
                comparison = {
                    "correlation": float(np.corrcoef(orig_hist_norm, proc_hist_norm)[0, 1]),
                    "chi_squared": float(cv2.compareHist(
                        orig_hist.astype(np.float32), 
                        proc_hist.astype(np.float32), 
                        cv2.HISTCMP_CHISQR
                    )),
                    "intersection": float(cv2.compareHist(
                        orig_hist.astype(np.float32), 
                        proc_hist.astype(np.float32), 
                        cv2.HISTCMP_INTERSECT
                    )),
                    "bhattacharyya": float(cv2.compareHist(
                        orig_hist.astype(np.float32), 
                        proc_hist.astype(np.float32), 
                        cv2.HISTCMP_BHATTACHARYYA
                    ))
                }
                
                hist_result[channel] = {
                    "original_histogram": orig_hist.tolist(),
                    "processed_histogram": proc_hist.tolist(),
                    "bins": bins.tolist(),
                    "comparison": comparison
                }
            
            return hist_result
            
        except Exception as e:
            self.logger.error(f"ヒストグラム解析エラー: {e}")
            raise
    
    def _analyze_color_shifts(self, orig_img: np.ndarray, proc_img: np.ndarray, color_space: str) -> Dict[str, Any]:
        """色変化解析"""
        try:
            # 色差計算
            color_diff = proc_img - orig_img
            
            shift_result = {
                "global_shift": {},
                "spatial_variation": {},
                "shift_distribution": {}
            }
            
            channels = self._get_channel_names(color_space)
            
            for i, channel in enumerate(channels):
                channel_diff = color_diff[:,:,i] if len(color_diff.shape) == 3 else color_diff
                
                # グローバルシフト
                shift_result["global_shift"][channel] = {
                    "mean_shift": float(np.mean(channel_diff)),
                    "median_shift": float(np.median(channel_diff)),
                    "std_shift": float(np.std(channel_diff))
                }
                
                # 空間的変動
                shift_result["spatial_variation"][channel] = {
                    "max_positive": float(np.max(channel_diff)),
                    "max_negative": float(np.min(channel_diff)),
                    "range": float(np.max(channel_diff) - np.min(channel_diff)),
                    "spatial_std": float(np.std(channel_diff))
                }
                
                # シフト分布
                shift_hist, shift_bins = np.histogram(channel_diff.flatten(), bins=self.hist_bins//2, range=(-1, 1))
                shift_result["shift_distribution"][channel] = {
                    "histogram": shift_hist.tolist(),
                    "bins": shift_bins.tolist()
                }
            
            # 色空間特有の解析
            if color_space.upper() == "RGB":
                shift_result["rgb_specific"] = self._analyze_rgb_shifts(orig_img, proc_img)
            elif color_space.upper() == "HSV":
                shift_result["hsv_specific"] = self._analyze_hsv_shifts(orig_img, proc_img)
            elif color_space.upper() == "LAB":
                shift_result["lab_specific"] = self._analyze_lab_shifts(orig_img, proc_img)
            
            return shift_result
            
        except Exception as e:
            self.logger.error(f"色変化解析エラー: {e}")
            raise
    
    def _analyze_dominant_colors(self, orig_img: np.ndarray, proc_img: np.ndarray, n_colors: int = 8) -> Dict[str, Any]:
        """主要色解析（k-means clustering）"""
        try:
            result = {}
            
            for name, img in [("original", orig_img), ("processed", proc_img)]:
                # 画像をベクトル化
                pixels = img.reshape(-1, 3)
                
                # k-meansクラスタリング
                kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
                kmeans.fit(pixels)
                
                # 主要色と割合
                colors = kmeans.cluster_centers_
                labels = kmeans.labels_
                
                # 各色の割合計算
                unique_labels, counts = np.unique(labels, return_counts=True)
                percentages = counts / len(labels) * 100
                
                result[name] = {
                    "colors": colors.tolist(),
                    "percentages": percentages.tolist(),
                    "inertia": float(kmeans.inertia_)
                }
            
            # 主要色の変化解析
            result["color_changes"] = self._compare_dominant_colors(
                result["original"]["colors"], 
                result["processed"]["colors"]
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"主要色解析エラー: {e}")
            raise
    
    def _analyze_color_distribution(self, orig_img: np.ndarray, proc_img: np.ndarray) -> Dict[str, Any]:
        """色分布解析"""
        try:
            result = {}
            
            # 色相（Hue）分布の解析
            orig_hsv = cv2.cvtColor((orig_img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
            proc_hsv = cv2.cvtColor((proc_img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
            
            # 色相ヒストグラム
            orig_hue_hist = np.histogram(orig_hsv[:,:,0], bins=36, range=(0, 180))[0]
            proc_hue_hist = np.histogram(proc_hsv[:,:,0], bins=36, range=(0, 180))[0]
            
            result["hue_distribution"] = {
                "original": orig_hue_hist.tolist(),
                "processed": proc_hue_hist.tolist(),
                "correlation": float(np.corrcoef(orig_hue_hist, proc_hue_hist)[0, 1])
            }
            
            # 彩度分布
            orig_sat_hist = np.histogram(orig_hsv[:,:,1], bins=32, range=(0, 255))[0]
            proc_sat_hist = np.histogram(proc_hsv[:,:,1], bins=32, range=(0, 255))[0]
            
            result["saturation_distribution"] = {
                "original": orig_sat_hist.tolist(),
                "processed": proc_sat_hist.tolist(),
                "correlation": float(np.corrcoef(orig_sat_hist, proc_sat_hist)[0, 1])
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"色分布解析エラー: {e}")
            raise
    
    def _calculate_quality_metrics(self, orig_img: np.ndarray, proc_img: np.ndarray) -> Dict[str, Any]:
        """品質指標の計算"""
        try:
            # MSE（平均二乗誤差）
            mse = float(np.mean((orig_img - proc_img) ** 2))
            
            # PSNR（ピーク信号対雑音比）
            if mse == 0:
                psnr = float('inf')
            else:
                psnr = float(20 * np.log10(1.0 / np.sqrt(mse)))
            
            # SSIM（構造的類似性）は後のフェーズで実装
            
            return {
                "mse": mse,
                "psnr": psnr,
                "color_difference_magnitude": float(np.mean(np.linalg.norm(proc_img - orig_img, axis=2)))
            }
            
        except Exception as e:
            self.logger.error(f"品質指標計算エラー: {e}")
            raise
    
    def _get_channel_names(self, color_space: str) -> List[str]:
        """色空間のチャンネル名を取得"""
        channel_names = {
            "RGB": ["Red", "Green", "Blue"],
            "HSV": ["Hue", "Saturation", "Value"],
            "LAB": ["Lightness", "A", "B"]
        }
        return channel_names.get(color_space.upper(), ["Channel_0", "Channel_1", "Channel_2"])
    
    def _analyze_rgb_shifts(self, orig_img: np.ndarray, proc_img: np.ndarray) -> Dict[str, Any]:
        """RGB特有の解析"""
        # 色温度変化の推定
        orig_mean = np.mean(orig_img, axis=(0, 1))
        proc_mean = np.mean(proc_img, axis=(0, 1))
        
        # 簡易的な色温度変化指標
        temp_shift = (proc_mean[2] - proc_mean[0]) - (orig_mean[2] - orig_mean[0])
        
        return {
            "temperature_shift_indicator": float(temp_shift),
            "overall_brightness_change": float(np.mean(proc_mean - orig_mean))
        }
    
    def _analyze_hsv_shifts(self, orig_hsv: np.ndarray, proc_hsv: np.ndarray) -> Dict[str, Any]:
        """HSV特有の解析"""
        # 色相シフトの循環性を考慮した解析
        hue_diff = proc_hsv[:,:,0] - orig_hsv[:,:,0]
        
        # 循環性を考慮（180度を超える差は反対方向として計算）
        hue_diff = np.where(hue_diff > 0.5, hue_diff - 1, hue_diff)
        hue_diff = np.where(hue_diff < -0.5, hue_diff + 1, hue_diff)
        
        return {
            "circular_hue_shift": float(np.mean(hue_diff)),
            "saturation_boost": float(np.mean(proc_hsv[:,:,1] - orig_hsv[:,:,1]))
        }
    
    def _analyze_lab_shifts(self, orig_lab: np.ndarray, proc_lab: np.ndarray) -> Dict[str, Any]:
        """LAB特有の解析"""
        # Delta E計算（簡易版）
        delta_e = np.sqrt(
            np.sum((proc_lab - orig_lab) ** 2, axis=2)
        )
        
        return {
            "mean_delta_e": float(np.mean(delta_e)),
            "max_delta_e": float(np.max(delta_e)),
            "perceptual_difference": float(np.std(delta_e))
        }
    
    def _compare_dominant_colors(self, orig_colors: List, proc_colors: List) -> Dict[str, Any]:
        """主要色の変化を比較"""
        # 最も近い色のペアを見つけて比較
        changes = []
        for orig_color in orig_colors:
            distances = [np.linalg.norm(np.array(orig_color) - np.array(proc_color)) 
                        for proc_color in proc_colors]
            min_dist_idx = np.argmin(distances)
            changes.append({
                "original": orig_color,
                "matched_processed": proc_colors[min_dist_idx],
                "distance": float(distances[min_dist_idx])
            })
        
        return {
            "color_pair_changes": changes,
            "average_color_shift": float(np.mean([c["distance"] for c in changes]))
        }
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """解析結果のサマリー生成"""
        try:
            summary = {
                "color_spaces_analyzed": list(results["basic_statistics"].keys()),
                "overall_assessment": {}
            }
            
            # RGB解析がある場合の総合評価
            if "RGB" in results["basic_statistics"]:
                rgb_stats = results["basic_statistics"]["RGB"]
                
                # 明度変化
                brightness_change = np.mean([
                    rgb_stats["Red"]["changes"]["mean_change"],
                    rgb_stats["Green"]["changes"]["mean_change"], 
                    rgb_stats["Blue"]["changes"]["mean_change"]
                ])
                
                # コントラスト変化
                contrast_change = np.mean([
                    rgb_stats["Red"]["changes"]["std_change"],
                    rgb_stats["Green"]["changes"]["std_change"],
                    rgb_stats["Blue"]["changes"]["std_change"]
                ])
                
                summary["overall_assessment"] = {
                    "brightness_change": float(brightness_change),
                    "contrast_change": float(contrast_change),
                    "dominant_color_shift": results.get("dominant_colors", {}).get("color_changes", {}).get("average_color_shift", 0),
                    "quality_score": results.get("quality_metrics", {}).get("psnr", 0)
                }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"サマリー生成エラー: {e}")
            return {"error": str(e)}