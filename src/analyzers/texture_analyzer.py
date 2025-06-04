"""
テクスチャ・質感解析モジュール (Phase 4-6)

このモジュールは以下のテクスチャ解析機能を提供:
- エッジ検出とシャープネス測定
- ノイズレベル分析
- 表面質感の定量化
- Haralick特徴による質感解析
"""

import cv2
import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional
from scipy import ndimage
from scipy.stats import entropy
from skimage import feature, measure, filters
from skimage.feature import graycomatrix, graycoprops
import mahotas

class TextureAnalyzer:
    """テクスチャ・質感解析クラス"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初期化
        
        Args:
            config: 設定辞書
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # テクスチャ解析設定
        texture_config = config.get('analysis', {}).get('texture', {})
        self.edge_threshold = texture_config.get('edge_threshold', 50)
        self.noise_window_size = texture_config.get('noise_window_size', 5)
        self.glcm_distances = texture_config.get('glcm_distances', [1, 2, 3])
        self.glcm_angles = texture_config.get('glcm_angles', [0, 45, 90, 135])
        
    def analyze_texture(self, original: np.ndarray, processed: np.ndarray) -> Dict[str, Any]:
        """
        包括的なテクスチャ解析を実行
        
        Args:
            original: 元画像 (H, W, 3)
            processed: 処理済み画像 (H, W, 3)
            
        Returns:
            解析結果辞書
        """
        try:
            self.logger.info("テクスチャ解析開始")
            
            results = {
                'edge_analysis': self._analyze_edges(original, processed),
                'sharpness_analysis': self._analyze_sharpness(original, processed),
                'noise_analysis': self._analyze_noise(original, processed),
                'surface_texture': self._analyze_surface_texture(original, processed),
                'haralick_features': self._analyze_haralick_features(original, processed)
            }
            
            # 総合評価
            results['overall_assessment'] = self._calculate_overall_texture_score(results)
            
            self.logger.info("テクスチャ解析完了")
            return results
            
        except Exception as e:
            self.logger.error(f"テクスチャ解析エラー: {e}")
            return {}
    
    def _analyze_edges(self, original: np.ndarray, processed: np.ndarray) -> Dict[str, Any]:
        """
        エッジ検出解析
        
        Args:
            original: 元画像
            processed: 処理済み画像
            
        Returns:
            エッジ解析結果
        """
        try:
            self.logger.info("エッジ検出解析開始")
            
            # グレースケール変換（uint8に変換）
            gray_orig = cv2.cvtColor((original * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            gray_proc = cv2.cvtColor((processed * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            
            # 複数のエッジ検出手法
            results = {}
            
            # 1. Canny エッジ検出
            canny_orig = cv2.Canny(gray_orig, self.edge_threshold, self.edge_threshold * 2)
            canny_proc = cv2.Canny(gray_proc, self.edge_threshold, self.edge_threshold * 2)
            
            results['canny'] = {
                'original_edge_density': np.sum(canny_orig > 0) / canny_orig.size,
                'processed_edge_density': np.sum(canny_proc > 0) / canny_proc.size,
                'edge_preservation_ratio': self._calculate_edge_preservation(canny_orig, canny_proc)
            }
            
            # 2. Sobel エッジ検出
            sobel_orig_x = cv2.Sobel(gray_orig, cv2.CV_64F, 1, 0, ksize=3)
            sobel_orig_y = cv2.Sobel(gray_orig, cv2.CV_64F, 0, 1, ksize=3)
            sobel_orig_mag = np.sqrt(sobel_orig_x**2 + sobel_orig_y**2)
            
            sobel_proc_x = cv2.Sobel(gray_proc, cv2.CV_64F, 1, 0, ksize=3)
            sobel_proc_y = cv2.Sobel(gray_proc, cv2.CV_64F, 0, 1, ksize=3)
            sobel_proc_mag = np.sqrt(sobel_proc_x**2 + sobel_proc_y**2)
            
            results['sobel'] = {
                'original_magnitude_mean': np.mean(sobel_orig_mag),
                'processed_magnitude_mean': np.mean(sobel_proc_mag),
                'magnitude_change_ratio': np.mean(sobel_proc_mag) / (np.mean(sobel_orig_mag) + 1e-8)
            }
            
            # 3. Laplacian エッジ検出
            laplacian_orig = cv2.Laplacian(gray_orig, cv2.CV_64F)
            laplacian_proc = cv2.Laplacian(gray_proc, cv2.CV_64F)
            
            results['laplacian'] = {
                'original_variance': np.var(laplacian_orig),
                'processed_variance': np.var(laplacian_proc),
                'variance_change_ratio': np.var(laplacian_proc) / (np.var(laplacian_orig) + 1e-8)
            }
            
            self.logger.info("エッジ検出解析完了")
            return results
            
        except Exception as e:
            self.logger.error(f"エッジ検出解析エラー: {e}")
            return {}
    
    def _analyze_sharpness(self, original: np.ndarray, processed: np.ndarray) -> Dict[str, Any]:
        """
        シャープネス解析
        
        Args:
            original: 元画像
            processed: 処理済み画像
            
        Returns:
            シャープネス解析結果
        """
        try:
            self.logger.info("シャープネス解析開始")
            
            gray_orig = cv2.cvtColor((original * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            gray_proc = cv2.cvtColor((processed * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            
            results = {}
            
            # 1. Tenengrad方法（Sobel勾配の分散）
            grad_orig = filters.sobel(gray_orig)
            grad_proc = filters.sobel(gray_proc)
            
            results['tenengrad'] = {
                'original_sharpness': np.var(grad_orig),
                'processed_sharpness': np.var(grad_proc),
                'sharpness_change_ratio': np.var(grad_proc) / (np.var(grad_orig) + 1e-8)
            }
            
            # 2. Laplacian分散方法
            laplacian_orig = cv2.Laplacian(gray_orig, cv2.CV_64F)
            laplacian_proc = cv2.Laplacian(gray_proc, cv2.CV_64F)
            
            results['laplacian_variance'] = {
                'original_sharpness': np.var(laplacian_orig),
                'processed_sharpness': np.var(laplacian_proc),
                'sharpness_change_ratio': np.var(laplacian_proc) / (np.var(laplacian_orig) + 1e-8)
            }
            
            # 3. 高周波成分解析
            f_orig = np.fft.fft2(gray_orig)
            f_proc = np.fft.fft2(gray_proc)
            
            # 高周波成分の抽出（中心から遠い成分）
            rows, cols = gray_orig.shape
            crow, ccol = rows // 2, cols // 2
            
            # 高周波マスク作成
            mask = np.ones((rows, cols), np.uint8)
            r = 30  # 低周波領域の半径
            y, x = np.ogrid[:rows, :cols]
            mask_area = (x - ccol)**2 + (y - crow)**2 <= r*r
            mask[mask_area] = 0
            
            high_freq_orig = np.sum(np.abs(f_orig * mask))
            high_freq_proc = np.sum(np.abs(f_proc * mask))
            
            results['frequency_analysis'] = {
                'original_high_freq_energy': high_freq_orig,
                'processed_high_freq_energy': high_freq_proc,
                'high_freq_preservation_ratio': high_freq_proc / (high_freq_orig + 1e-8)
            }
            
            self.logger.info("シャープネス解析完了")
            return results
            
        except Exception as e:
            self.logger.error(f"シャープネス解析エラー: {e}")
            return {}
    
    def _analyze_noise(self, original: np.ndarray, processed: np.ndarray) -> Dict[str, Any]:
        """
        ノイズレベル分析
        
        Args:
            original: 元画像
            processed: 処理済み画像
            
        Returns:
            ノイズ解析結果
        """
        try:
            self.logger.info("ノイズ分析開始")
            
            gray_orig = cv2.cvtColor((original * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64)
            gray_proc = cv2.cvtColor((processed * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64)
            
            results = {}
            
            # 1. 局所標準偏差によるノイズ推定
            window_size = self.noise_window_size
            kernel = np.ones((window_size, window_size)) / (window_size * window_size)
            
            mean_orig = cv2.filter2D(gray_orig, -1, kernel)
            mean_proc = cv2.filter2D(gray_proc, -1, kernel)
            
            sqr_orig = cv2.filter2D(gray_orig**2, -1, kernel)
            sqr_proc = cv2.filter2D(gray_proc**2, -1, kernel)
            
            noise_orig = np.sqrt(np.maximum(sqr_orig - mean_orig**2, 0))
            noise_proc = np.sqrt(np.maximum(sqr_proc - mean_proc**2, 0))
            
            results['local_std_noise'] = {
                'original_noise_level': np.mean(noise_orig),
                'processed_noise_level': np.mean(noise_proc),
                'noise_reduction_ratio': (np.mean(noise_orig) - np.mean(noise_proc)) / (np.mean(noise_orig) + 1e-8)
            }
            
            # 2. Wavelet denoising推定
            try:
                import pywt
                
                # ウェーブレット変換
                coeffs_orig = pywt.dwt2(gray_orig, 'db4')
                coeffs_proc = pywt.dwt2(gray_proc, 'db4')
                
                # 高周波係数からノイズレベル推定
                _, (lh_orig, hl_orig, hh_orig) = coeffs_orig
                _, (lh_proc, hl_proc, hh_proc) = coeffs_proc
                
                noise_orig_wavelet = np.median(np.abs(hh_orig)) / 0.6745
                noise_proc_wavelet = np.median(np.abs(hh_proc)) / 0.6745
                
                results['wavelet_noise'] = {
                    'original_noise_level': noise_orig_wavelet,
                    'processed_noise_level': noise_proc_wavelet,
                    'noise_reduction_ratio': (noise_orig_wavelet - noise_proc_wavelet) / (noise_orig_wavelet + 1e-8)
                }
                
            except ImportError:
                self.logger.warning("PyWavelets が利用できません。ウェーブレットノイズ分析をスキップ")
            
            # 3. 高周波ノイズ分析
            # ガウシアンフィルタを適用して低周波成分を取得
            low_freq_orig = cv2.GaussianBlur(gray_orig, (5, 5), 1.0)
            low_freq_proc = cv2.GaussianBlur(gray_proc, (5, 5), 1.0)
            
            # 高周波成分 = 元画像 - 低周波成分
            high_freq_orig = gray_orig - low_freq_orig
            high_freq_proc = gray_proc - low_freq_proc
            
            results['high_freq_noise'] = {
                'original_noise_power': np.var(high_freq_orig),
                'processed_noise_power': np.var(high_freq_proc),
                'noise_reduction_ratio': (np.var(high_freq_orig) - np.var(high_freq_proc)) / (np.var(high_freq_orig) + 1e-8)
            }
            
            self.logger.info("ノイズ分析完了")
            return results
            
        except Exception as e:
            self.logger.error(f"ノイズ分析エラー: {e}")
            return {}
    
    def _analyze_surface_texture(self, original: np.ndarray, processed: np.ndarray) -> Dict[str, Any]:
        """
        表面質感分析
        
        Args:
            original: 元画像
            processed: 処理済み画像
            
        Returns:
            表面質感解析結果
        """
        try:
            self.logger.info("表面質感分析開始")
            
            gray_orig = cv2.cvtColor((original * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            gray_proc = cv2.cvtColor((processed * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            
            results = {}
            
            # 1. Local Binary Pattern (LBP) - 最適化版
            # 画像サイズに応じてLBPパラメータを調整
            height, width = gray_orig.shape
            if height * width > 500000:  # 大きな画像の場合
                radius, n_points = 4, 16  # 高速設定
            else:
                radius, n_points = 8, 24  # 標準設定
            
            lbp_orig = feature.local_binary_pattern(gray_orig, n_points, radius, method='uniform')
            lbp_proc = feature.local_binary_pattern(gray_proc, n_points, radius, method='uniform')
            
            # LBPヒストグラム
            hist_orig, _ = np.histogram(lbp_orig.ravel(), bins=26, range=(0, 26))
            hist_proc, _ = np.histogram(lbp_proc.ravel(), bins=26, range=(0, 26))
            
            # 正規化
            hist_orig = hist_orig.astype(float) / (hist_orig.sum() + 1e-8)
            hist_proc = hist_proc.astype(float) / (hist_proc.sum() + 1e-8)
            
            # LBPパターンの類似度
            lbp_similarity = 1.0 - 0.5 * np.sum(np.abs(hist_orig - hist_proc))
            
            results['lbp_analysis'] = {
                'original_uniformity': hist_orig[0],  # 均一パターンの割合
                'processed_uniformity': hist_proc[0],
                'pattern_similarity': lbp_similarity,
                'texture_entropy_original': entropy(hist_orig + 1e-8),
                'texture_entropy_processed': entropy(hist_proc + 1e-8)
            }
            
            # 2. Gabor フィルタ応答
            gabor_responses_orig = []
            gabor_responses_proc = []
            
            # 複数の方向と周波数でGaborフィルタ適用
            for theta in [0, 45, 90, 135]:
                for frequency in [0.1, 0.3, 0.5]:
                    real_orig, _ = filters.gabor(gray_orig, frequency=frequency, 
                                               theta=np.deg2rad(theta))
                    real_proc, _ = filters.gabor(gray_proc, frequency=frequency, 
                                               theta=np.deg2rad(theta))
                    
                    gabor_responses_orig.append(np.var(real_orig))
                    gabor_responses_proc.append(np.var(real_proc))
            
            results['gabor_analysis'] = {
                'original_texture_energy': np.mean(gabor_responses_orig),
                'processed_texture_energy': np.mean(gabor_responses_proc),
                'texture_preservation_ratio': np.mean(gabor_responses_proc) / (np.mean(gabor_responses_orig) + 1e-8),
                'directional_responses_original': gabor_responses_orig,
                'directional_responses_processed': gabor_responses_proc
            }
            
            # 3. 質感の粗さ解析
            # 標準偏差フィルタによる局所的な粗さ
            roughness_orig = ndimage.generic_filter(gray_orig.astype(float), np.std, size=5)
            roughness_proc = ndimage.generic_filter(gray_proc.astype(float), np.std, size=5)
            
            results['roughness_analysis'] = {
                'original_roughness': np.mean(roughness_orig),
                'processed_roughness': np.mean(roughness_proc),
                'roughness_change_ratio': np.mean(roughness_proc) / (np.mean(roughness_orig) + 1e-8),
                'roughness_uniformity_original': 1.0 / (np.std(roughness_orig) + 1e-8),
                'roughness_uniformity_processed': 1.0 / (np.std(roughness_proc) + 1e-8)
            }
            
            self.logger.info("表面質感分析完了")
            return results
            
        except Exception as e:
            self.logger.error(f"表面質感分析エラー: {e}")
            return {}
    
    def _analyze_haralick_features(self, original: np.ndarray, processed: np.ndarray) -> Dict[str, Any]:
        """
        Haralick特徴による質感解析
        
        Args:
            original: 元画像
            processed: 処理済み画像
            
        Returns:
            Haralick特徴解析結果
        """
        try:
            self.logger.info("Haralick特徴分析開始")
            
            gray_orig = cv2.cvtColor((original * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            gray_proc = cv2.cvtColor((processed * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            
            # グレーレベルを256から64に量子化（計算効率のため）
            gray_orig = (gray_orig // 4).astype(np.uint8)
            gray_proc = (gray_proc // 4).astype(np.uint8)
            
            results = {}
            
            # Mahotas を使用したHaralick特徴
            try:
                haralick_orig = mahotas.features.haralick(gray_orig)
                haralick_proc = mahotas.features.haralick(gray_proc)
                
                # 各方向の平均を取る
                haralick_mean_orig = np.mean(haralick_orig, axis=0)
                haralick_mean_proc = np.mean(haralick_proc, axis=0)
                
                feature_names = [
                    'angular_second_moment',
                    'contrast', 
                    'correlation',
                    'sum_of_squares_variance',
                    'inverse_difference_moment',
                    'sum_average',
                    'sum_variance',
                    'sum_entropy',
                    'entropy',
                    'difference_variance',
                    'difference_entropy',
                    'info_correlation_1',
                    'info_correlation_2'
                ]
                
                for i, name in enumerate(feature_names):
                    results[name] = {
                        'original': haralick_mean_orig[i],
                        'processed': haralick_mean_proc[i],
                        'change_ratio': haralick_mean_proc[i] / (haralick_mean_orig[i] + 1e-8)
                    }
                
            except Exception as e:
                self.logger.warning(f"Mahotas Haralick特徴計算エラー: {e}")
                
                # scikit-imageでの代替実装
                distances = self.glcm_distances
                angles = [np.deg2rad(a) for a in self.glcm_angles]
                
                glcm_orig = graycomatrix(gray_orig, distances, angles, 64, symmetric=True, normed=True)
                glcm_proc = graycomatrix(gray_proc, distances, angles, 64, symmetric=True, normed=True)
                
                properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
                
                for prop in properties:
                    orig_vals = graycoprops(glcm_orig, prop)
                    proc_vals = graycoprops(glcm_proc, prop)
                    
                    results[prop] = {
                        'original': np.mean(orig_vals),
                        'processed': np.mean(proc_vals),
                        'change_ratio': np.mean(proc_vals) / (np.mean(orig_vals) + 1e-8)
                    }
            
            self.logger.info("Haralick特徴分析完了")
            return results
            
        except Exception as e:
            self.logger.error(f"Haralick特徴分析エラー: {e}")
            return {}
    
    def _calculate_edge_preservation(self, edges_orig: np.ndarray, edges_proc: np.ndarray) -> float:
        """
        エッジ保存率計算
        
        Args:
            edges_orig: 元画像のエッジ
            edges_proc: 処理済み画像のエッジ
            
        Returns:
            エッジ保存率 (0-1)
        """
        intersection = np.logical_and(edges_orig, edges_proc)
        union = np.logical_or(edges_orig, edges_proc)
        
        if np.sum(union) == 0:
            return 1.0
        
        return np.sum(intersection) / np.sum(union)
    
    def _calculate_overall_texture_score(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        総合テクスチャ評価スコア計算
        
        Args:
            results: 各種解析結果
            
        Returns:
            総合評価結果
        """
        try:
            scores = {}
            
            # エッジ保存スコア
            if 'edge_analysis' in results:
                edge_data = results['edge_analysis']
                if 'canny' in edge_data:
                    scores['edge_preservation'] = edge_data['canny'].get('edge_preservation_ratio', 0)
            
            # シャープネス変化スコア
            if 'sharpness_analysis' in results:
                sharp_data = results['sharpness_analysis']
                if 'tenengrad' in sharp_data:
                    scores['sharpness_change'] = sharp_data['tenengrad'].get('sharpness_change_ratio', 1)
            
            # ノイズ削減スコア
            if 'noise_analysis' in results:
                noise_data = results['noise_analysis']
                if 'local_std_noise' in noise_data:
                    scores['noise_reduction'] = noise_data['local_std_noise'].get('noise_reduction_ratio', 0)
            
            # 質感保存スコア
            if 'surface_texture' in results:
                texture_data = results['surface_texture']
                if 'lbp_analysis' in texture_data:
                    scores['texture_preservation'] = texture_data['lbp_analysis'].get('pattern_similarity', 0)
            
            # 総合スコア（重み付き平均）
            weights = {
                'edge_preservation': 0.3,
                'sharpness_change': 0.25,
                'noise_reduction': 0.25,
                'texture_preservation': 0.2
            }
            
            overall_score = 0
            total_weight = 0
            
            for key, weight in weights.items():
                if key in scores:
                    overall_score += scores[key] * weight
                    total_weight += weight
            
            if total_weight > 0:
                overall_score /= total_weight
            
            return {
                'individual_scores': scores,
                'overall_score': overall_score,
                'quality_assessment': self._assess_texture_quality(overall_score)
            }
            
        except Exception as e:
            self.logger.error(f"総合評価計算エラー: {e}")
            return {'overall_score': 0, 'quality_assessment': 'unknown'}
    
    def _assess_texture_quality(self, score: float) -> str:
        """
        テクスチャ品質評価
        
        Args:
            score: 総合スコア
            
        Returns:
            品質評価文字列
        """
        if score >= 0.9:
            return 'excellent'
        elif score >= 0.8:
            return 'very_good'
        elif score >= 0.7:
            return 'good'
        elif score >= 0.6:
            return 'fair'
        elif score >= 0.5:
            return 'poor'
        else:
            return 'very_poor'