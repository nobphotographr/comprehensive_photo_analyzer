"""
印象・感情解析モジュール (Phase 7-9)

このモジュールは以下の印象・感情解析機能を提供:
- 色彩心理学に基づく感情分析
- 明度・コントラスト印象分析
- 美的評価指標による品質評価
- ムード・雰囲気解析
- LUT効果による感情変化定量化
"""

import cv2
import numpy as np
import logging
from typing import Dict, Any, Tuple, List
from scipy import stats
from scipy.spatial.distance import euclidean
import math

class ImpressionAnalyzer:
    """印象・感情解析クラス"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初期化
        
        Args:
            config: 設定辞書
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 印象解析設定
        impression_config = config.get('analysis', {}).get('impression', {})
        self.warmth_threshold = impression_config.get('warmth_threshold', 0.5)
        self.saturation_weight = impression_config.get('saturation_weight', 0.3)
        self.brightness_weight = impression_config.get('brightness_weight', 0.4)
        self.contrast_weight = impression_config.get('contrast_weight', 0.3)
        
        # 色彩感情マッピング
        self._initialize_color_emotion_mapping()
        
        # 美的評価設定
        self._initialize_aesthetic_parameters()
    
    def analyze_impression(self, original: np.ndarray, processed: np.ndarray) -> Dict[str, Any]:
        """
        包括的な印象・感情解析を実行
        
        Args:
            original: 元画像 (H, W, 3)
            processed: 処理済み画像 (H, W, 3)
            
        Returns:
            解析結果辞書
        """
        try:
            self.logger.info("印象・感情解析開始")
            
            results = {
                'color_psychology': self._analyze_color_psychology(original, processed),
                'brightness_contrast_impression': self._analyze_brightness_contrast_impression(original, processed),
                'aesthetic_evaluation': self._analyze_aesthetic_quality(original, processed),
                'mood_atmosphere': self._analyze_mood_atmosphere(original, processed),
                'emotional_change': self._analyze_emotional_change(original, processed)
            }
            
            # 総合印象評価
            results['overall_impression'] = self._calculate_overall_impression(results)
            
            self.logger.info("印象・感情解析完了")
            return results
            
        except Exception as e:
            self.logger.error(f"印象・感情解析エラー: {e}")
            return {}
    
    def _initialize_color_emotion_mapping(self):
        """色彩感情マッピングの初期化"""
        # HSV色相に基づく基本感情マッピング
        self.color_emotions = {
            # 色相範囲（度）: [感情値, 感情名, 説明]
            (0, 30): [0.7, "warm_passionate", "情熱的・エネルギッシュ"],      # 赤
            (30, 60): [0.8, "warm_cheerful", "明るい・楽観的"],              # オレンジ・黄
            (60, 120): [0.6, "natural_peaceful", "自然・平和"],              # 黄緑・緑
            (120, 180): [0.4, "cool_calm", "冷静・落ち着き"],                # 青緑・シアン
            (180, 240): [0.2, "cool_stable", "安定・信頼"],                  # 青
            (240, 300): [0.3, "cool_mysterious", "神秘的・高貴"],            # 青紫・紫
            (300, 360): [0.5, "warm_romantic", "ロマンチック・優雅"]         # 紫・マゼンタ
        }
        
        # 彩度による感情強度調整
        self.saturation_emotion_curve = {
            0.0: 0.0,    # 無彩色 - 中性的
            0.2: 0.3,    # 低彩度 - 穏やか
            0.5: 0.7,    # 中彩度 - 一般的
            0.8: 0.9,    # 高彩度 - 強い感情
            1.0: 1.0     # 最高彩度 - 極めて強い感情
        }
        
        # 明度による感情極性調整
        self.brightness_emotion_curve = {
            0.0: -0.8,   # 暗い - ネガティブ
            0.2: -0.4,   # 低明度 - やや暗い
            0.4: 0.0,    # 中間明度 - 中性
            0.6: 0.4,    # 高明度 - やや明るい
            0.8: 0.7,    # 明るい - ポジティブ
            1.0: 0.9     # 最高明度 - 非常にポジティブ
        }
    
    def _initialize_aesthetic_parameters(self):
        """美的評価パラメータの初期化"""
        # 黄金比
        self.golden_ratio = 1.618
        
        # 三分割法のグリッド位置
        self.rule_of_thirds_positions = [
            (1/3, 1/3), (2/3, 1/3), (1/3, 2/3), (2/3, 2/3)
        ]
        
        # 色調和のタイプ
        self.color_harmony_types = {
            'complementary': 180,    # 補色
            'triadic': 120,          # 三色配色
            'analogous': 30,         # 類似色
            'split_complementary': [150, 210],  # 分裂補色
            'tetradic': [90, 180, 270]          # 四色配色
        }
    
    def _analyze_color_psychology(self, original: np.ndarray, processed: np.ndarray) -> Dict[str, Any]:
        """
        色彩心理学に基づく感情分析
        
        Args:
            original: 元画像
            processed: 処理済み画像
            
        Returns:
            色彩心理分析結果
        """
        try:
            self.logger.info("色彩心理学分析開始")
            
            results = {}
            
            for name, image in [("original", original), ("processed", processed)]:
                # HSV変換
                hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
                h, s, v = hsv[:, :, 0], hsv[:, :, 1] / 255.0, hsv[:, :, 2] / 255.0
                
                # ベクトル化された感情スコア計算（最適化版）
                # 画像をサンプリングして計算量を削減
                sample_rate = max(1, (h.shape[0] * h.shape[1]) // 10000)  # 最大10,000ピクセルでサンプリング
                h_sampled = h[::sample_rate, ::sample_rate]
                s_sampled = s[::sample_rate, ::sample_rate]
                v_sampled = v[::sample_rate, ::sample_rate]
                
                # 色相を度に変換
                hue_degrees = h_sampled.astype(np.float32) * 2
                
                # ベクトル化された感情計算
                emotion_scores = []
                emotion_types = []
                
                # サンプリングされたピクセルのみ処理
                flat_hue = hue_degrees.flatten()
                flat_sat = s_sampled.flatten()
                flat_val = v_sampled.flatten()
                
                for i in range(len(flat_hue)):
                    # 色相による基本感情
                    base_emotion = self._get_hue_emotion(flat_hue[i])
                    
                    # 彩度・明度調整（ベクトル化）
                    saturation_multiplier = self._interpolate_curve(flat_sat[i], self.saturation_emotion_curve)
                    brightness_modifier = self._interpolate_curve(flat_val[i], self.brightness_emotion_curve)
                    
                    # 最終感情スコア
                    final_emotion = base_emotion['value'] * saturation_multiplier + brightness_modifier
                    final_emotion = np.clip(final_emotion, -1.0, 1.0)
                    
                    emotion_scores.append(final_emotion)
                    emotion_types.append(base_emotion['type'])
                
                # 統計計算
                emotion_scores = np.array(emotion_scores)
                
                # 温暖感分析
                warmth_score = self._calculate_warmth_score(h, s, v)
                
                # 感情分布
                emotion_distribution = self._calculate_emotion_distribution(emotion_types)
                
                # 感情強度
                emotion_intensity = np.std(emotion_scores)
                
                # 感情一貫性
                emotion_consistency = 1.0 / (1.0 + emotion_intensity)
                
                results[name] = {
                    'overall_emotion_score': np.mean(emotion_scores),
                    'emotion_intensity': emotion_intensity,
                    'emotion_consistency': emotion_consistency,
                    'warmth_score': warmth_score,
                    'emotion_distribution': emotion_distribution,
                    'dominant_emotion': max(emotion_distribution.items(), key=lambda x: x[1])[0],
                    'emotion_range': [float(np.min(emotion_scores)), float(np.max(emotion_scores))]
                }
            
            # 変化分析
            emotion_change = results["processed"]["overall_emotion_score"] - results["original"]["overall_emotion_score"]
            warmth_change = results["processed"]["warmth_score"] - results["original"]["warmth_score"]
            intensity_change = results["processed"]["emotion_intensity"] - results["original"]["emotion_intensity"]
            
            results['changes'] = {
                'emotion_change': emotion_change,
                'warmth_change': warmth_change,
                'intensity_change': intensity_change,
                'change_assessment': self._assess_emotional_change(emotion_change, warmth_change, intensity_change)
            }
            
            self.logger.info("色彩心理学分析完了")
            return results
            
        except Exception as e:
            self.logger.error(f"色彩心理学分析エラー: {e}")
            return {}
    
    def _analyze_brightness_contrast_impression(self, original: np.ndarray, processed: np.ndarray) -> Dict[str, Any]:
        """
        明度・コントラスト印象分析
        
        Args:
            original: 元画像
            processed: 処理済み画像
            
        Returns:
            明度・コントラスト印象分析結果
        """
        try:
            self.logger.info("明度・コントラスト印象分析開始")
            
            results = {}
            
            for name, image in [("original", original), ("processed", processed)]:
                # グレースケール変換
                gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
                
                # 明度分析
                brightness_mean = np.mean(gray)
                brightness_std = np.std(gray)
                brightness_impression = self._calculate_brightness_impression(brightness_mean)
                
                # コントラスト分析
                contrast_rms = np.sqrt(np.mean((gray - brightness_mean) ** 2))
                contrast_impression = self._calculate_contrast_impression(contrast_rms)
                
                # 明度分布分析
                brightness_histogram, _ = np.histogram(gray, bins=10, range=(0, 1))
                brightness_distribution = brightness_histogram / np.sum(brightness_histogram)
                
                # 動的レンジ
                dynamic_range = np.max(gray) - np.min(gray)
                
                # 局所コントラスト
                local_contrast = self._calculate_local_contrast(gray)
                
                results[name] = {
                    'brightness_mean': brightness_mean,
                    'brightness_std': brightness_std,
                    'brightness_impression': brightness_impression,
                    'contrast_rms': contrast_rms,
                    'contrast_impression': contrast_impression,
                    'dynamic_range': dynamic_range,
                    'local_contrast_mean': np.mean(local_contrast),
                    'brightness_distribution': brightness_distribution.tolist(),
                    'overall_mood_score': self._calculate_mood_from_brightness_contrast(
                        brightness_impression, contrast_impression
                    )
                }
            
            # 変化分析
            brightness_change = results["processed"]["brightness_mean"] - results["original"]["brightness_mean"]
            contrast_change = results["processed"]["contrast_rms"] - results["original"]["contrast_rms"]
            mood_change = results["processed"]["overall_mood_score"] - results["original"]["overall_mood_score"]
            
            results['changes'] = {
                'brightness_change': brightness_change,
                'contrast_change': contrast_change,
                'mood_change': mood_change,
                'impression_shift': self._assess_impression_shift(brightness_change, contrast_change, mood_change)
            }
            
            self.logger.info("明度・コントラスト印象分析完了")
            return results
            
        except Exception as e:
            self.logger.error(f"明度・コントラスト印象分析エラー: {e}")
            return {}
    
    def _analyze_aesthetic_quality(self, original: np.ndarray, processed: np.ndarray) -> Dict[str, Any]:
        """
        美的評価指標による品質評価
        
        Args:
            original: 元画像
            processed: 処理済み画像
            
        Returns:
            美的評価結果
        """
        try:
            self.logger.info("美的評価分析開始")
            
            results = {}
            
            for name, image in [("original", original), ("processed", processed)]:
                h, w = image.shape[:2]
                
                # 構図評価
                composition_score = self._evaluate_composition(image)
                
                # 色彩調和評価
                color_harmony_score = self._evaluate_color_harmony(image)
                
                # バランス評価
                balance_score = self._evaluate_visual_balance(image)
                
                # 黄金比評価
                golden_ratio_score = self._evaluate_golden_ratio_compliance(image)
                
                # 複雑度評価
                complexity_score = self._evaluate_visual_complexity(image)
                
                # 統一感評価
                unity_score = self._evaluate_visual_unity(image)
                
                # 総合美的スコア
                aesthetic_weights = {
                    'composition': 0.25,
                    'color_harmony': 0.2,
                    'balance': 0.2,
                    'golden_ratio': 0.1,
                    'complexity': 0.15,
                    'unity': 0.1
                }
                
                overall_aesthetic_score = (
                    composition_score * aesthetic_weights['composition'] +
                    color_harmony_score * aesthetic_weights['color_harmony'] +
                    balance_score * aesthetic_weights['balance'] +
                    golden_ratio_score * aesthetic_weights['golden_ratio'] +
                    complexity_score * aesthetic_weights['complexity'] +
                    unity_score * aesthetic_weights['unity']
                )
                
                results[name] = {
                    'composition_score': composition_score,
                    'color_harmony_score': color_harmony_score,
                    'balance_score': balance_score,
                    'golden_ratio_score': golden_ratio_score,
                    'complexity_score': complexity_score,
                    'unity_score': unity_score,
                    'overall_aesthetic_score': overall_aesthetic_score,
                    'aesthetic_quality': self._assess_aesthetic_quality(overall_aesthetic_score)
                }
            
            # 美的品質変化分析
            aesthetic_change = results["processed"]["overall_aesthetic_score"] - results["original"]["overall_aesthetic_score"]
            
            results['changes'] = {
                'aesthetic_change': aesthetic_change,
                'composition_change': results["processed"]["composition_score"] - results["original"]["composition_score"],
                'harmony_change': results["processed"]["color_harmony_score"] - results["original"]["color_harmony_score"],
                'balance_change': results["processed"]["balance_score"] - results["original"]["balance_score"],
                'aesthetic_improvement_assessment': self._assess_aesthetic_improvement(aesthetic_change)
            }
            
            self.logger.info("美的評価分析完了")
            return results
            
        except Exception as e:
            self.logger.error(f"美的評価分析エラー: {e}")
            return {}
    
    def _analyze_mood_atmosphere(self, original: np.ndarray, processed: np.ndarray) -> Dict[str, Any]:
        """
        ムード・雰囲気解析
        
        Args:
            original: 元画像
            processed: 処理済み画像
            
        Returns:
            ムード・雰囲気解析結果
        """
        try:
            self.logger.info("ムード・雰囲気分析開始")
            
            results = {}
            
            for name, image in [("original", original), ("processed", processed)]:
                # 各種ムード指標の計算
                mood_indicators = {}
                
                # 1. 暖かさ/冷たさ
                mood_indicators['warmth'] = self._calculate_image_warmth(image)
                
                # 2. 明るさ/暗さ
                mood_indicators['brightness_mood'] = self._calculate_brightness_mood(image)
                
                # 3. 活発さ/静寂さ
                mood_indicators['energy'] = self._calculate_energy_level(image)
                
                # 4. 柔らかさ/硬さ
                mood_indicators['softness'] = self._calculate_softness(image)
                
                # 5. 神秘性/明確性
                mood_indicators['mystery'] = self._calculate_mystery_level(image)
                
                # 6. 自然性/人工性
                mood_indicators['naturalness'] = self._calculate_naturalness(image)
                
                # ムードプロファイル作成
                mood_profile = self._create_mood_profile(mood_indicators)
                
                # 支配的ムード
                dominant_mood = self._determine_dominant_mood(mood_indicators)
                
                # 雰囲気スコア
                atmosphere_score = self._calculate_atmosphere_score(mood_indicators)
                
                results[name] = {
                    'mood_indicators': mood_indicators,
                    'mood_profile': mood_profile,
                    'dominant_mood': dominant_mood,
                    'atmosphere_score': atmosphere_score,
                    'mood_description': self._generate_mood_description(mood_profile, dominant_mood)
                }
            
            # ムード変化分析
            mood_shift = self._analyze_mood_shift(results["original"], results["processed"])
            
            results['changes'] = mood_shift
            
            self.logger.info("ムード・雰囲気分析完了")
            return results
            
        except Exception as e:
            self.logger.error(f"ムード・雰囲気分析エラー: {e}")
            return {}
    
    def _analyze_emotional_change(self, original: np.ndarray, processed: np.ndarray) -> Dict[str, Any]:
        """
        感情変化分析（LUT効果による）
        
        Args:
            original: 元画像
            processed: 処理済み画像
            
        Returns:
            感情変化分析結果
        """
        try:
            self.logger.info("感情変化分析開始")
            
            # LUT処理による主要な変化を検出
            changes = {}
            
            # 色相変化
            hue_shift = self._calculate_hue_shift(original, processed)
            
            # 彩度変化
            saturation_change = self._calculate_saturation_change(original, processed)
            
            # 明度変化
            brightness_change = self._calculate_brightness_change(original, processed)
            
            # 感情ベクトル変化
            emotion_vector_change = self._calculate_emotion_vector_change(original, processed)
            
            # LUT効果分類
            lut_effect_type = self._classify_lut_effect(hue_shift, saturation_change, brightness_change)
            
            # 感情変化の強度
            emotional_impact_strength = self._calculate_emotional_impact_strength(
                hue_shift, saturation_change, brightness_change, emotion_vector_change
            )
            
            # 変化の方向性（ポジティブ/ネガティブ）
            change_direction = self._determine_change_direction(emotion_vector_change)
            
            changes = {
                'hue_shift': hue_shift,
                'saturation_change': saturation_change,
                'brightness_change': brightness_change,
                'emotion_vector_change': emotion_vector_change,
                'lut_effect_type': lut_effect_type,
                'emotional_impact_strength': emotional_impact_strength,
                'change_direction': change_direction,
                'lut_effectiveness': self._evaluate_lut_effectiveness(emotional_impact_strength, change_direction)
            }
            
            self.logger.info("感情変化分析完了")
            return changes
            
        except Exception as e:
            self.logger.error(f"感情変化分析エラー: {e}")
            return {}
    
    def _calculate_overall_impression(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        総合印象評価スコア計算
        
        Args:
            results: 各種解析結果
            
        Returns:
            総合印象評価結果
        """
        try:
            scores = {}
            
            # 各解析結果からスコア抽出
            if 'color_psychology' in results:
                psychology_data = results['color_psychology']
                if 'changes' in psychology_data:
                    scores['emotional_impact'] = abs(psychology_data['changes'].get('emotion_change', 0))
            
            if 'brightness_contrast_impression' in results:
                brightness_data = results['brightness_contrast_impression']
                if 'changes' in brightness_data:
                    scores['mood_impact'] = abs(brightness_data['changes'].get('mood_change', 0))
            
            if 'aesthetic_evaluation' in results:
                aesthetic_data = results['aesthetic_evaluation']
                if 'changes' in aesthetic_data:
                    scores['aesthetic_improvement'] = max(0, aesthetic_data['changes'].get('aesthetic_change', 0))
            
            if 'emotional_change' in results:
                emotional_data = results['emotional_change']
                scores['lut_effectiveness'] = emotional_data.get('lut_effectiveness', 0)
            
            # 重み付き総合スコア
            weights = {
                'emotional_impact': 0.3,
                'mood_impact': 0.25,
                'aesthetic_improvement': 0.25,
                'lut_effectiveness': 0.2
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
                'overall_impression_score': overall_score,
                'impression_assessment': self._assess_overall_impression(overall_score),
                'improvement_recommendations': self._generate_improvement_recommendations(results)
            }
            
        except Exception as e:
            self.logger.error(f"総合印象評価計算エラー: {e}")
            return {'overall_impression_score': 0, 'impression_assessment': 'unknown'}
    
    # ヘルパーメソッドの実装（一部抜粋）
    
    def _get_hue_emotion(self, hue: float) -> Dict[str, Any]:
        """色相から基本感情を取得"""
        for (min_hue, max_hue), (value, emotion_type, description) in self.color_emotions.items():
            if min_hue <= hue < max_hue:
                return {'value': value, 'type': emotion_type, 'description': description}
        return {'value': 0.5, 'type': 'neutral', 'description': '中性的'}
    
    def _interpolate_curve(self, value: float, curve: Dict[float, float]) -> float:
        """カーブ補間"""
        keys = sorted(curve.keys())
        if value <= keys[0]:
            return curve[keys[0]]
        if value >= keys[-1]:
            return curve[keys[-1]]
        
        for i in range(len(keys) - 1):
            if keys[i] <= value <= keys[i + 1]:
                t = (value - keys[i]) / (keys[i + 1] - keys[i])
                return curve[keys[i]] * (1 - t) + curve[keys[i + 1]] * t
        return 0.0
    
    def _calculate_warmth_score(self, h: np.ndarray, s: np.ndarray, v: np.ndarray) -> float:
        """暖かさスコア計算"""
        # 暖色（赤・オレンジ・黄）の重み付き平均
        warm_mask = ((h * 2 >= 0) & (h * 2 <= 60)) | ((h * 2 >= 300) & (h * 2 <= 360))
        cool_mask = (h * 2 >= 180) & (h * 2 <= 240)
        
        warm_weight = np.sum(s[warm_mask] * v[warm_mask]) if np.any(warm_mask) else 0
        cool_weight = np.sum(s[cool_mask] * v[cool_mask]) if np.any(cool_mask) else 0
        total_weight = warm_weight + cool_weight
        
        if total_weight == 0:
            return 0.5
        
        return warm_weight / total_weight
    
    def _calculate_emotion_distribution(self, emotion_types: List[str]) -> Dict[str, float]:
        """感情分布計算"""
        total = len(emotion_types)
        if total == 0:
            return {}
        
        distribution = {}
        for emotion_type in set(emotion_types):
            distribution[emotion_type] = emotion_types.count(emotion_type) / total
        
        return distribution
    
    def _assess_emotional_change(self, emotion_change: float, warmth_change: float, intensity_change: float) -> str:
        """感情変化の評価"""
        if abs(emotion_change) < 0.1 and abs(warmth_change) < 0.1:
            return "minimal_change"
        elif emotion_change > 0.2:
            return "significantly_more_positive"
        elif emotion_change < -0.2:
            return "significantly_more_negative"
        elif warmth_change > 0.2:
            return "warmer_mood"
        elif warmth_change < -0.2:
            return "cooler_mood"
        else:
            return "moderate_change"
    
    def _calculate_brightness_impression(self, brightness: float) -> Dict[str, Any]:
        """明度印象計算"""
        if brightness < 0.2:
            return {'level': 'very_dark', 'mood': 'somber', 'energy': 'low'}
        elif brightness < 0.4:
            return {'level': 'dark', 'mood': 'serious', 'energy': 'calm'}
        elif brightness < 0.6:
            return {'level': 'medium', 'mood': 'balanced', 'energy': 'moderate'}
        elif brightness < 0.8:
            return {'level': 'bright', 'mood': 'cheerful', 'energy': 'energetic'}
        else:
            return {'level': 'very_bright', 'mood': 'vibrant', 'energy': 'high'}
    
    def _calculate_contrast_impression(self, contrast: float) -> Dict[str, Any]:
        """コントラスト印象計算"""
        if contrast < 0.1:
            return {'level': 'very_low', 'mood': 'soft', 'drama': 'minimal'}
        elif contrast < 0.2:
            return {'level': 'low', 'mood': 'gentle', 'drama': 'subtle'}
        elif contrast < 0.3:
            return {'level': 'medium', 'mood': 'balanced', 'drama': 'moderate'}
        elif contrast < 0.4:
            return {'level': 'high', 'mood': 'dynamic', 'drama': 'strong'}
        else:
            return {'level': 'very_high', 'mood': 'intense', 'drama': 'dramatic'}
    
    def _calculate_local_contrast(self, gray: np.ndarray) -> np.ndarray:
        """局所コントラスト計算"""
        # 3x3カーネルでの局所標準偏差
        kernel = np.ones((3, 3)) / 9
        mean_local = cv2.filter2D(gray, -1, kernel)
        mean_sq_local = cv2.filter2D(gray**2, -1, kernel)
        local_contrast = np.sqrt(np.maximum(mean_sq_local - mean_local**2, 0))
        return local_contrast
    
    def _calculate_mood_from_brightness_contrast(self, brightness_impression: Dict, contrast_impression: Dict) -> float:
        """明度・コントラストからムードスコア計算"""
        brightness_weights = {'very_dark': 0.1, 'dark': 0.3, 'medium': 0.5, 'bright': 0.7, 'very_bright': 0.9}
        contrast_weights = {'very_low': 0.2, 'low': 0.4, 'medium': 0.6, 'high': 0.8, 'very_high': 1.0}
        
        brightness_score = brightness_weights.get(brightness_impression['level'], 0.5)
        contrast_score = contrast_weights.get(contrast_impression['level'], 0.5)
        
        # 重み付き平均
        return brightness_score * 0.6 + contrast_score * 0.4
    
    def _assess_impression_shift(self, brightness_change: float, contrast_change: float, mood_change: float) -> str:
        """印象変化の評価"""
        if abs(mood_change) < 0.1:
            return "subtle_shift"
        elif mood_change > 0.2:
            return "significantly_brighter_mood"
        elif mood_change < -0.2:
            return "significantly_darker_mood"
        elif brightness_change > 0.1 and contrast_change > 0.05:
            return "enhanced_visibility"
        elif brightness_change < -0.1 and contrast_change < -0.05:
            return "muted_appearance"
        else:
            return "moderate_mood_shift"
    
    # その他のメソッドは簡略化...
    def _evaluate_composition(self, image: np.ndarray) -> float:
        """構図評価（簡略版）"""
        return 0.7  # 実装省略
    
    def _evaluate_color_harmony(self, image: np.ndarray) -> float:
        """色彩調和評価（簡略版）"""
        return 0.6  # 実装省略
    
    def _evaluate_visual_balance(self, image: np.ndarray) -> float:
        """視覚バランス評価（簡略版）"""
        return 0.8  # 実装省略
    
    def _evaluate_golden_ratio_compliance(self, image: np.ndarray) -> float:
        """黄金比評価（簡略版）"""
        return 0.5  # 実装省略
    
    def _evaluate_visual_complexity(self, image: np.ndarray) -> float:
        """視覚複雑度評価（簡略版）"""
        return 0.6  # 実装省略
    
    def _evaluate_visual_unity(self, image: np.ndarray) -> float:
        """視覚統一感評価（簡略版）"""
        return 0.7  # 実装省略
    
    def _assess_aesthetic_quality(self, score: float) -> str:
        """美的品質評価"""
        if score >= 0.8: return "excellent"
        elif score >= 0.7: return "very_good"
        elif score >= 0.6: return "good"
        elif score >= 0.5: return "fair"
        else: return "poor"
    
    def _assess_aesthetic_improvement(self, change: float) -> str:
        """美的改善評価"""
        if change >= 0.1: return "significant_improvement"
        elif change >= 0.05: return "moderate_improvement"
        elif change >= -0.05: return "minimal_change"
        else: return "degradation"
    
    # ムード関連メソッド（簡略版）
    def _calculate_image_warmth(self, image: np.ndarray) -> float:
        return 0.6  # 実装省略
    
    def _calculate_brightness_mood(self, image: np.ndarray) -> float:
        return 0.7  # 実装省略
    
    def _calculate_energy_level(self, image: np.ndarray) -> float:
        return 0.5  # 実装省略
    
    def _calculate_softness(self, image: np.ndarray) -> float:
        return 0.6  # 実装省略
    
    def _calculate_mystery_level(self, image: np.ndarray) -> float:
        return 0.4  # 実装省略
    
    def _calculate_naturalness(self, image: np.ndarray) -> float:
        return 0.8  # 実装省略
    
    def _create_mood_profile(self, indicators: Dict) -> Dict:
        return indicators  # 簡略版
    
    def _determine_dominant_mood(self, indicators: Dict) -> str:
        return "balanced"  # 簡略版
    
    def _calculate_atmosphere_score(self, indicators: Dict) -> float:
        return 0.7  # 簡略版
    
    def _generate_mood_description(self, profile: Dict, dominant: str) -> str:
        return f"画像の雰囲気は{dominant}で特徴付けられます"  # 簡略版
    
    def _analyze_mood_shift(self, original: Dict, processed: Dict) -> Dict:
        return {"mood_shift": "moderate"}  # 簡略版
    
    # 感情変化関連メソッド（簡略版）
    def _calculate_hue_shift(self, original: np.ndarray, processed: np.ndarray) -> float:
        return 0.1  # 実装省略
    
    def _calculate_saturation_change(self, original: np.ndarray, processed: np.ndarray) -> float:
        return 0.05  # 実装省略
    
    def _calculate_brightness_change(self, original: np.ndarray, processed: np.ndarray) -> float:
        return 0.08  # 実装省略
    
    def _calculate_emotion_vector_change(self, original: np.ndarray, processed: np.ndarray) -> float:
        return 0.12  # 実装省略
    
    def _classify_lut_effect(self, hue: float, sat: float, bright: float) -> str:
        return "warm_enhancement"  # 簡略版
    
    def _calculate_emotional_impact_strength(self, hue: float, sat: float, bright: float, emotion: float) -> float:
        return np.sqrt(hue**2 + sat**2 + bright**2 + emotion**2)
    
    def _determine_change_direction(self, emotion_change: float) -> str:
        if emotion_change > 0.1: return "positive"
        elif emotion_change < -0.1: return "negative" 
        else: return "neutral"
    
    def _evaluate_lut_effectiveness(self, strength: float, direction: str) -> float:
        base_score = min(strength, 1.0)
        if direction == "positive": return base_score * 1.2
        elif direction == "negative": return base_score * 0.8
        else: return base_score
    
    def _assess_overall_impression(self, score: float) -> str:
        if score >= 0.8: return "excellent_impression"
        elif score >= 0.6: return "good_impression"
        elif score >= 0.4: return "moderate_impression"
        else: return "poor_impression"
    
    def _generate_improvement_recommendations(self, results: Dict) -> List[str]:
        recommendations = []
        # 簡略版の推奨事項
        recommendations.append("色彩バランスの調整を検討してください")
        recommendations.append("明度・コントラストの最適化を行ってください")
        return recommendations