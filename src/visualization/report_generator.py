"""
Report Generator

レポート生成モジュール
HTML形式での包括的解析レポートの生成を行う
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import base64
from io import BytesIO

import matplotlib
matplotlib.use('Agg')  # GUI無しバックエンドを使用
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image

from utils.logger import get_logger


class ReportGenerator:
    """レポート生成クラス"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger("report_generator")
        
        # 出力設定
        self.output_format = config.get("output", {}).get("format", "html")
        self.include_raw_data = config.get("output", {}).get("include_raw_data", True)
        self.include_visualizations = config.get("output", {}).get("include_visualizations", True)
        
        # テンプレート設定
        self.template_dir = Path(__file__).parent / "templates" / "html"
        
        # 色設定
        sns.set_style("whitegrid")
        plt.style.use('seaborn-v0_8')
    
    def generate_report(self, analysis_results: Dict[str, Any], original_path: str, processed_path: str) -> str:
        """
        総合解析レポートの生成
        
        Args:
            analysis_results: 解析結果
            original_path: 元画像パス
            processed_path: 処理済み画像パス
        
        Returns:
            生成されたレポートファイルのパス
        """
        try:
            self.logger.info("レポート生成開始")
            
            # 出力ディレクトリの準備
            output_dir = Path(self.config.get("output", {}).get("directory", "data/output"))
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # ファイル名の生成
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            orig_name = Path(original_path).stem
            output_name = f"analysis_report_{orig_name}_{timestamp}"
            
            # HTML レポート生成
            if self.output_format in ["html", "all"]:
                html_path = self._generate_html_report(analysis_results, original_path, processed_path, output_dir, output_name)
            
            # JSON データ出力
            if self.output_format in ["json", "all"] and self.include_raw_data:
                json_path = self._generate_json_report(analysis_results, output_dir, output_name)
            
            # CSV データ出力
            if self.output_format in ["csv", "all"]:
                csv_path = self._generate_csv_report(analysis_results, output_dir, output_name)
            
            self.logger.info(f"レポート生成完了: {output_name}")
            return str(html_path) if self.output_format in ["html", "all"] else str(output_dir / f"{output_name}.json")
            
        except Exception as e:
            self.logger.error(f"レポート生成エラー: {e}")
            raise
    
    def generate_batch_report(self, analysis_results: Dict[str, Any], original_path: str, processed_path: str, 
                            output_dir: str, output_prefix: str) -> str:
        """
        バッチ処理用レポート生成
        
        Args:
            analysis_results: 解析結果
            original_path: 元画像パス  
            processed_path: 処理済み画像パス
            output_dir: 出力ディレクトリ
            output_prefix: 出力ファイル名プレフィックス
        
        Returns:
            生成されたレポートファイルのパス
        """
        try:
            output_path = Path(output_dir)
            
            # HTMLレポート生成
            html_path = self._generate_html_report(
                analysis_results, original_path, processed_path, output_path, output_prefix
            )
            
            # JSON データ出力
            if self.include_raw_data:
                self._generate_json_report(analysis_results, output_path, output_prefix)
            
            return str(html_path)
            
        except Exception as e:
            self.logger.error(f"バッチレポート生成エラー: {e}")
            raise
    
    def _generate_html_report(self, results: Dict[str, Any], orig_path: str, proc_path: str, 
                            output_dir: Path, output_name: str) -> Path:
        """HTML レポートの生成"""
        try:
            # 可視化画像の生成
            visualizations = self._generate_visualizations(results, output_dir, output_name)
            
            # 画像のエンコード
            image_data = self._encode_images(orig_path, proc_path)
            
            # HTML コンテンツの生成
            html_content = self._build_html_content(results, visualizations, image_data, orig_path, proc_path)
            
            # ファイル保存
            html_path = output_dir / f"{output_name}.html"
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.debug(f"HTML レポート保存: {html_path}")
            return html_path
            
        except Exception as e:
            self.logger.error(f"HTML レポート生成エラー: {e}")
            raise
    
    def _generate_visualizations(self, results: Dict[str, Any], output_dir: Path, output_name: str) -> Dict[str, str]:
        """可視化画像の生成"""
        try:
            visualizations = {}
            
            if not self.include_visualizations:
                return visualizations
            
            # 色彩解析の可視化
            if "color_analysis" in results:
                color_results = results["color_analysis"]
                
                # ヒストグラム比較
                if "histograms" in color_results:
                    hist_path = self._create_histogram_comparison(color_results["histograms"], output_dir, f"{output_name}_histograms")
                    visualizations["histograms"] = hist_path
                
                # 統計比較チャート
                if "basic_statistics" in color_results:
                    stats_path = self._create_statistics_chart(color_results["basic_statistics"], output_dir, f"{output_name}_statistics")
                    visualizations["statistics"] = stats_path
                
                # 色変化可視化
                if "color_shifts" in color_results:
                    shifts_path = self._create_color_shifts_chart(color_results["color_shifts"], output_dir, f"{output_name}_shifts")
                    visualizations["color_shifts"] = shifts_path
                
                # 主要色比較
                if "dominant_colors" in color_results:
                    colors_path = self._create_dominant_colors_chart(color_results["dominant_colors"], output_dir, f"{output_name}_colors")
                    visualizations["dominant_colors"] = colors_path
                
                # 高度解析の可視化
                if "advanced_analysis" in color_results:
                    advanced_viz = self._create_advanced_analysis_visualizations(
                        color_results["advanced_analysis"], output_dir, output_name
                    )
                    visualizations.update(advanced_viz)
            
            # テクスチャ解析の可視化
            if "texture_analysis" in results:
                texture_viz = self._create_texture_analysis_visualizations(
                    results["texture_analysis"], output_dir, output_name
                )
                visualizations.update(texture_viz)
            
            # 印象・感情解析の可視化
            if "impression_analysis" in results:
                impression_viz = self._create_impression_analysis_visualizations(
                    results["impression_analysis"], output_dir, output_name
                )
                visualizations.update(impression_viz)
            
            return visualizations
            
        except Exception as e:
            self.logger.error(f"可視化生成エラー: {e}")
            return {}
    
    def _create_histogram_comparison(self, hist_data: Dict[str, Any], output_dir: Path, filename: str) -> str:
        """ヒストグラム比較チャートの生成"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Color Histogram Comparison', fontsize=16)
            
            plot_positions = [(0, 0), (0, 1), (1, 0)]
            color_spaces = list(hist_data.keys())[:3]  # 最大3つの色空間
            
            for idx, color_space in enumerate(color_spaces):
                if idx >= 3:
                    break
                    
                row, col = plot_positions[idx]
                ax = axes[row, col]
                
                channels = list(hist_data[color_space].keys())
                for channel in channels:
                    if channel.endswith('_histogram') or 'histogram' not in channel:
                        continue
                        
                    channel_data = hist_data[color_space][channel]
                    if 'original_histogram' in channel_data and 'processed_histogram' in channel_data:
                        orig_hist = np.array(channel_data['original_histogram'])
                        proc_hist = np.array(channel_data['processed_histogram'])
                        bins = np.array(channel_data['bins'])
                        
                        bin_centers = (bins[:-1] + bins[1:]) / 2
                        
                        ax.plot(bin_centers, orig_hist, label=f'{channel} (Original)', alpha=0.7)
                        ax.plot(bin_centers, proc_hist, label=f'{channel} (Processed)', alpha=0.7)
                
                ax.set_title(f'{color_space} Histogram')
                ax.set_xlabel('Value')
                ax.set_ylabel('Frequency')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # 空いているサブプロットを非表示
            if len(color_spaces) < 4:
                axes[1, 1].set_visible(False)
            
            plt.tight_layout()
            
            # 画像として保存
            image_path = output_dir / f"{filename}.png"
            plt.savefig(image_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return self._encode_image_to_base64(image_path)
            
        except Exception as e:
            self.logger.error(f"ヒストグラムチャート生成エラー: {e}")
            return ""
    
    def _create_statistics_chart(self, stats_data: Dict[str, Any], output_dir: Path, filename: str) -> str:
        """統計チャートの生成"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Color Statistics Comparison', fontsize=16)
            
            # データ整理
            metrics = ['mean', 'std', 'median']
            color_spaces = list(stats_data.keys())
            
            for idx, metric in enumerate(metrics[:4]):
                row, col = idx // 2, idx % 2
                ax = axes[row, col]
                
                original_values = []
                processed_values = []
                labels = []
                
                for cs in color_spaces:
                    for channel in stats_data[cs]:
                        if metric in stats_data[cs][channel]['original']:
                            original_values.append(stats_data[cs][channel]['original'][metric])
                            processed_values.append(stats_data[cs][channel]['processed'][metric])
                            labels.append(f"{cs}_{channel}")
                
                x = np.arange(len(labels))
                width = 0.35
                
                ax.bar(x - width/2, original_values, width, label='Original', alpha=0.8)
                ax.bar(x + width/2, processed_values, width, label='Processed', alpha=0.8)
                
                ax.set_xlabel('Channels')
                ax.set_ylabel(metric.capitalize())
                ax.set_title(f'{metric.capitalize()} Comparison')
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=45, ha='right')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # 空いているサブプロットを非表示
            if len(metrics) < 4:
                axes[1, 1].set_visible(False)
            
            plt.tight_layout()
            
            # 画像として保存
            image_path = output_dir / f"{filename}.png"
            plt.savefig(image_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return self._encode_image_to_base64(image_path)
            
        except Exception as e:
            self.logger.error(f"統計チャート生成エラー: {e}")
            return ""
    
    def _create_color_shifts_chart(self, shifts_data: Dict[str, Any], output_dir: Path, filename: str) -> str:
        """色変化チャートの生成"""
        try:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle('Color Shifts Analysis', fontsize=16)
            
            color_spaces = list(shifts_data.keys())[:3]
            
            for idx, cs in enumerate(color_spaces):
                ax = axes[idx] if len(color_spaces) > 1 else axes
                
                if 'global_shift' in shifts_data[cs]:
                    global_shifts = shifts_data[cs]['global_shift']
                    channels = list(global_shifts.keys())
                    mean_shifts = [global_shifts[ch]['mean_shift'] for ch in channels]
                    
                    colors = ['red', 'green', 'blue'] if cs == 'RGB' else ['orange', 'purple', 'cyan']
                    
                    bars = ax.bar(channels, mean_shifts, color=colors[:len(channels)], alpha=0.7)
                    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                    ax.set_title(f'{cs} Global Shifts')
                    ax.set_ylabel('Mean Shift')
                    ax.grid(True, alpha=0.3)
                    
                    # 値をバーの上に表示
                    for bar, value in zip(bars, mean_shifts):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01 if height >= 0 else height - 0.01,
                               f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top')
            
            plt.tight_layout()
            
            # 画像として保存
            image_path = output_dir / f"{filename}.png"
            plt.savefig(image_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return self._encode_image_to_base64(image_path)
            
        except Exception as e:
            self.logger.error(f"色変化チャート生成エラー: {e}")
            return ""
    
    def _create_dominant_colors_chart(self, colors_data: Dict[str, Any], output_dir: Path, filename: str) -> str:
        """主要色チャートの生成"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle('Dominant Colors Comparison', fontsize=16)
            
            # 元画像の主要色
            if 'original' in colors_data:
                orig_colors = np.array(colors_data['original']['colors'])
                orig_percentages = colors_data['original']['percentages']
                
                # 色パレット表示
                for i, (color, pct) in enumerate(zip(orig_colors, orig_percentages)):
                    # 色の値を0-1の範囲に正規化
                    normalized_color = np.clip(color, 0, 1)
                    ax1.barh(i, pct, color=normalized_color, alpha=0.8)
                    ax1.text(pct + 1, i, f'{pct:.1f}%', va='center')
                
                ax1.set_title('Original Image')
                ax1.set_xlabel('Percentage')
                ax1.set_ylabel('Color Index')
                ax1.set_xlim(0, max(orig_percentages) * 1.2)
            
            # 処理済み画像の主要色
            if 'processed' in colors_data:
                proc_colors = np.array(colors_data['processed']['colors'])
                proc_percentages = colors_data['processed']['percentages']
                
                # 色パレット表示
                for i, (color, pct) in enumerate(zip(proc_colors, proc_percentages)):
                    # 色の値を0-1の範囲に正規化
                    normalized_color = np.clip(color, 0, 1)
                    ax2.barh(i, pct, color=normalized_color, alpha=0.8)
                    ax2.text(pct + 1, i, f'{pct:.1f}%', va='center')
                
                ax2.set_title('Processed Image')
                ax2.set_xlabel('Percentage')
                ax2.set_ylabel('Color Index')
                ax2.set_xlim(0, max(proc_percentages) * 1.2)
            
            plt.tight_layout()
            
            # 画像として保存
            image_path = output_dir / f"{filename}.png"
            plt.savefig(image_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return self._encode_image_to_base64(image_path)
            
        except Exception as e:
            self.logger.error(f"主要色チャート生成エラー: {e}")
            return ""
    
    def _create_advanced_analysis_visualizations(self, advanced_data: Dict[str, Any], 
                                               output_dir: Path, output_name: str) -> Dict[str, str]:
        """高度解析結果の可視化"""
        try:
            visualizations = {}
            
            # Delta E2000ヒートマップ
            if "delta_e2000" in advanced_data and "delta_e_map" in advanced_data["delta_e2000"]:
                delta_e_path = self._create_delta_e_heatmap(
                    advanced_data["delta_e2000"], output_dir, f"{output_name}_delta_e2000"
                )
                visualizations["delta_e2000"] = delta_e_path
            
            # 色温度分析チャート
            if "color_temperature" in advanced_data:
                temp_path = self._create_color_temperature_chart(
                    advanced_data["color_temperature"], output_dir, f"{output_name}_temperature"
                )
                visualizations["color_temperature"] = temp_path
            
            # 色域分析チャート
            if "color_gamut" in advanced_data:
                gamut_path = self._create_color_gamut_chart(
                    advanced_data["color_gamut"], output_dir, f"{output_name}_gamut"
                )
                visualizations["color_gamut"] = gamut_path
            
            # 色彩調和分析（ultra精度の場合）
            if "color_harmony" in advanced_data:
                harmony_path = self._create_color_harmony_chart(
                    advanced_data["color_harmony"], output_dir, f"{output_name}_harmony"
                )
                visualizations["color_harmony"] = harmony_path
            
            return visualizations
            
        except Exception as e:
            self.logger.error(f"高度解析可視化エラー: {e}")
            return {}
    
    def _create_delta_e_heatmap(self, delta_e_data: Dict[str, Any], output_dir: Path, filename: str) -> str:
        """Delta E2000ヒートマップの生成"""
        try:
            if "error" in delta_e_data or "delta_e_map" not in delta_e_data:
                return ""
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle('Delta E2000 Analysis', fontsize=16)
            
            delta_e_map = np.array(delta_e_data["delta_e_map"])
            stats = delta_e_data.get("delta_e_statistics", {})
            
            # Delta Eヒートマップ
            im1 = ax1.imshow(delta_e_map, cmap='hot', interpolation='bilinear')
            ax1.set_title('Delta E2000 Heatmap')
            ax1.set_xlabel('Width')
            ax1.set_ylabel('Height')
            plt.colorbar(im1, ax=ax1, label='Delta E2000')
            
            # Delta E分布ヒストグラム
            ax2.hist(delta_e_map.flatten(), bins=50, alpha=0.7, color='red', edgecolor='black')
            ax2.axvline(stats.get("mean", 0), color='blue', linestyle='--', 
                       label=f'Mean: {stats.get("mean", 0):.2f}')
            ax2.axvline(stats.get("median", 0), color='green', linestyle='--', 
                       label=f'Median: {stats.get("median", 0):.2f}')
            ax2.set_title('Delta E2000 Distribution')
            ax2.set_xlabel('Delta E2000 Value')
            ax2.set_ylabel('Frequency')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 画像として保存
            image_path = output_dir / f"{filename}.png"
            plt.savefig(image_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return self._encode_image_to_base64(image_path)
            
        except Exception as e:
            self.logger.error(f"Delta Eヒートマップ生成エラー: {e}")
            return ""
    
    def _create_color_temperature_chart(self, temp_data: Dict[str, Any], output_dir: Path, filename: str) -> str:
        """色温度分析チャートの生成"""
        try:
            if "error" in temp_data:
                return ""
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Color Temperature Analysis', fontsize=16)
            
            # グローバル色温度変化
            global_temp = temp_data.get("global_temperature", {})
            if global_temp:
                temps = [global_temp.get("original_kelvin", 5500), 
                        global_temp.get("processed_kelvin", 5500)]
                labels = ['Original', 'Processed']
                colors = ['orange', 'blue']
                
                bars = ax1.bar(labels, temps, color=colors, alpha=0.7)
                ax1.set_title('Global Color Temperature')
                ax1.set_ylabel('Temperature (K)')
                ax1.grid(True, alpha=0.3)
                
                # 値をバーの上に表示
                for bar, temp in zip(bars, temps):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height + 50,
                            f'{temp:.0f}K', ha='center', va='bottom')
            
            # 空間的色温度分析
            spatial_temp = temp_data.get("spatial_temperature", {})
            if spatial_temp:
                regions = list(spatial_temp.keys())
                changes = [spatial_temp[region].get("change_kelvin", 0) for region in regions]
                
                colors_map = ['red' if c > 0 else 'blue' if c < 0 else 'gray' for c in changes]
                bars = ax2.bar(range(len(regions)), changes, color=colors_map, alpha=0.7)
                ax2.set_title('Spatial Temperature Changes')
                ax2.set_ylabel('Temperature Change (K)')
                ax2.set_xticks(range(len(regions)))
                ax2.set_xticklabels(regions, rotation=45, ha='right')
                ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax2.grid(True, alpha=0.3)
            
            # ホワイトバランス分析
            wb_data = temp_data.get("white_balance_analysis", {})
            if wb_data:
                rgb_shifts = wb_data.get("rgb_shift", [0, 0, 0])
                rgb_labels = ['Red', 'Green', 'Blue']
                rgb_colors = ['red', 'green', 'blue']
                
                bars = ax3.bar(rgb_labels, rgb_shifts, color=rgb_colors, alpha=0.7)
                ax3.set_title('White Balance Shift')
                ax3.set_ylabel('RGB Shift')
                ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax3.grid(True, alpha=0.3)
            
            # ティント分析
            tint_data = temp_data.get("tint_analysis", {})
            if tint_data:
                tint_value = tint_data.get("tint_shift_value", 0)
                tint_direction = tint_data.get("tint_direction", "変化なし")
                
                color = 'magenta' if tint_value > 0 else 'green' if tint_value < 0 else 'gray'
                ax4.bar(['Tint Shift'], [tint_value], color=color, alpha=0.7)
                ax4.set_title(f'Tint Analysis: {tint_direction}')
                ax4.set_ylabel('Tint Shift Value')
                ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 画像として保存
            image_path = output_dir / f"{filename}.png"
            plt.savefig(image_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return self._encode_image_to_base64(image_path)
            
        except Exception as e:
            self.logger.error(f"色温度チャート生成エラー: {e}")
            return ""
    
    def _create_color_gamut_chart(self, gamut_data: Dict[str, Any], output_dir: Path, filename: str) -> str:
        """色域分析チャートの生成"""
        try:
            if "error" in gamut_data:
                return ""
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle('Color Gamut Analysis', fontsize=16)
            
            # 色域カバレッジ比較
            coverage_data = gamut_data.get("gamut_coverage", {})
            if coverage_data:
                color_spaces = list(coverage_data.keys())
                original_coverage = [coverage_data[cs].get("original", 0) for cs in color_spaces]
                processed_coverage = [coverage_data[cs].get("processed", 0) for cs in color_spaces]
                
                x = np.arange(len(color_spaces))
                width = 0.35
                
                bars1 = ax1.bar(x - width/2, original_coverage, width, label='Original', alpha=0.8)
                bars2 = ax1.bar(x + width/2, processed_coverage, width, label='Processed', alpha=0.8)
                
                ax1.set_title('Color Gamut Coverage')
                ax1.set_ylabel('Coverage Ratio')
                ax1.set_xlabel('Color Space')
                ax1.set_xticks(x)
                ax1.set_xticklabels(color_spaces)
                ax1.legend()
                ax1.grid(True, alpha=0.3)
            
            # アウトオブガマット分析
            oog_data = gamut_data.get("out_of_gamut_analysis", {})
            if oog_data:
                color_spaces = list(oog_data.keys())
                oog_changes = [oog_data[cs].get("change", 0) for cs in color_spaces]
                
                colors_map = ['red' if c > 0 else 'green' if c < 0 else 'gray' for c in oog_changes]
                bars = ax2.bar(color_spaces, oog_changes, color=colors_map, alpha=0.7)
                ax2.set_title('Out-of-Gamut Change')
                ax2.set_ylabel('Change (%)')
                ax2.set_xlabel('Color Space')
                ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 画像として保存
            image_path = output_dir / f"{filename}.png"
            plt.savefig(image_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return self._encode_image_to_base64(image_path)
            
        except Exception as e:
            self.logger.error(f"色域チャート生成エラー: {e}")
            return ""
    
    def _create_color_harmony_chart(self, harmony_data: Dict[str, Any], output_dir: Path, filename: str) -> str:
        """色彩調和分析チャートの生成"""
        try:
            if "error" in harmony_data:
                return ""
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Color Harmony Analysis', fontsize=16)
            
            # 補色関係
            comp_data = harmony_data.get("complementary_analysis", {})
            if comp_data:
                values = [comp_data.get("original_complementary_pairs", 0),
                         comp_data.get("processed_complementary_pairs", 0)]
                labels = ['Original', 'Processed']
                
                bars = ax1.bar(labels, values, color=['orange', 'purple'], alpha=0.7)
                ax1.set_title('Complementary Color Pairs')
                ax1.set_ylabel('Number of Pairs')
                ax1.grid(True, alpha=0.3)
            
            # 類似色関係
            analog_data = harmony_data.get("analogous_analysis", {})
            if analog_data:
                values = [analog_data.get("original_analogous_pairs", 0),
                         analog_data.get("processed_analogous_pairs", 0)]
                labels = ['Original', 'Processed']
                
                bars = ax2.bar(labels, values, color=['green', 'blue'], alpha=0.7)
                ax2.set_title('Analogous Color Pairs')
                ax2.set_ylabel('Number of Pairs')
                ax2.grid(True, alpha=0.3)
            
            # 三角配色関係
            triadic_data = harmony_data.get("triadic_analysis", {})
            if triadic_data:
                values = [triadic_data.get("original_triadic_sets", 0),
                         triadic_data.get("processed_triadic_sets", 0)]
                labels = ['Original', 'Processed']
                
                bars = ax3.bar(labels, values, color=['red', 'cyan'], alpha=0.7)
                ax3.set_title('Triadic Color Sets')
                ax3.set_ylabel('Number of Sets')
                ax3.grid(True, alpha=0.3)
            
            # 調和スコア
            harmony_score = harmony_data.get("harmony_score", {})
            if harmony_score:
                scores = [harmony_score.get("original_harmony_score", 0),
                         harmony_score.get("processed_harmony_score", 0)]
                labels = ['Original', 'Processed']
                
                bars = ax4.bar(labels, scores, color=['gold', 'silver'], alpha=0.7)
                ax4.set_title('Harmony Score')
                ax4.set_ylabel('Score (0-1)')
                ax4.set_ylim(0, 1)
                ax4.grid(True, alpha=0.3)
                
                # 値をバーの上に表示
                for bar, score in zip(bars, scores):
                    height = bar.get_height()
                    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                            f'{score:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # 画像として保存
            image_path = output_dir / f"{filename}.png"
            plt.savefig(image_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return self._encode_image_to_base64(image_path)
            
        except Exception as e:
            self.logger.error(f"色彩調和チャート生成エラー: {e}")
            return ""
    
    def _encode_images(self, orig_path: str, proc_path: str) -> Dict[str, str]:
        """画像のBase64エンコード"""
        try:
            image_data = {}
            
            # 元画像
            with Image.open(orig_path) as img:
                # リサイズ（レポート用）
                img.thumbnail((800, 600), Image.Resampling.LANCZOS)
                buffer = BytesIO()
                img.save(buffer, format='JPEG', quality=85)
                image_data['original'] = base64.b64encode(buffer.getvalue()).decode()
            
            # 処理済み画像
            with Image.open(proc_path) as img:
                # リサイズ（レポート用）
                img.thumbnail((800, 600), Image.Resampling.LANCZOS)
                buffer = BytesIO()
                img.save(buffer, format='JPEG', quality=85)
                image_data['processed'] = base64.b64encode(buffer.getvalue()).decode()
            
            return image_data
            
        except Exception as e:
            self.logger.error(f"画像エンコードエラー: {e}")
            return {}
    
    def _encode_image_to_base64(self, image_path: Path) -> str:
        """画像ファイルをBase64エンコード"""
        try:
            with open(image_path, 'rb') as f:
                return base64.b64encode(f.read()).decode()
        except Exception as e:
            self.logger.error(f"画像ファイルエンコードエラー: {e}")
            return ""
    
    def _build_html_content(self, results: Dict[str, Any], visualizations: Dict[str, str], 
                          image_data: Dict[str, str], orig_path: str, proc_path: str) -> str:
        """HTML コンテンツの構築"""
        try:
            # 基本情報
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            orig_name = Path(orig_path).name
            proc_name = Path(proc_path).name
            
            # サマリー情報の抽出
            summary = results.get("color_analysis", {}).get("summary", {})
            
            html_content = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>写真解析レポート - {orig_name}</title>
    <style>
        body {{
            font-family: 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            border-bottom: 3px solid #007acc;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            color: #333;
            margin: 0;
            font-size: 2.5em;
        }}
        .header .subtitle {{
            color: #666;
            font-size: 1.1em;
            margin-top: 10px;
        }}
        .image-comparison {{
            display: flex;
            gap: 20px;
            margin: 30px 0;
            justify-content: center;
        }}
        .image-container {{
            text-align: center;
            flex: 1;
            max-width: 400px;
        }}
        .image-container img {{
            width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        .image-container h3 {{
            margin-top: 15px;
            color: #333;
            font-size: 1.2em;
        }}
        .section {{
            margin: 40px 0;
            padding: 25px;
            background-color: #f9f9f9;
            border-radius: 8px;
            border-left: 4px solid #007acc;
        }}
        .section h2 {{
            color: #333;
            margin-top: 0;
            font-size: 1.8em;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .metric-card h4 {{
            margin: 0 0 10px 0;
            color: #555;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .metric-card .value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #007acc;
            margin: 5px 0;
        }}
        .chart-container {{
            text-align: center;
            margin: 30px 0;
        }}
        .chart-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .data-table th, .data-table td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        .data-table th {{
            background-color: #007acc;
            color: white;
            font-weight: bold;
        }}
        .data-table tr:hover {{
            background-color: #f5f5f5;
        }}
        .footer {{
            text-align: center;
            margin-top: 50px;
            padding-top: 30px;
            border-top: 2px solid #eee;
            color: #666;
            font-size: 0.9em;
        }}
        .positive {{ color: #28a745; }}
        .negative {{ color: #dc3545; }}
        .neutral {{ color: #6c757d; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>写真解析レポート</h1>
            <div class="subtitle">
                生成日時: {timestamp}<br>
                解析対象: {orig_name} ↔ {proc_name}
            </div>
        </div>
        
        <div class="image-comparison">
            <div class="image-container">
                <img src="data:image/jpeg;base64,{image_data.get('original', '')}" alt="元画像">
                <h3>元画像</h3>
                <p>{orig_name}</p>
            </div>
            <div class="image-container">
                <img src="data:image/jpeg;base64,{image_data.get('processed', '')}" alt="処理済み画像">
                <h3>処理済み画像</h3>
                <p>{proc_name}</p>
            </div>
        </div>
        
        <div class="section">
            <h2>📊 解析サマリー</h2>
            <div class="summary-grid">
                {self._build_summary_cards(summary)}
            </div>
        </div>
        
        {self._build_color_analysis_section(results.get("color_analysis", {}), visualizations)}
        
        {self._build_advanced_analysis_section(results.get("color_analysis", {}).get("advanced_analysis", {}), visualizations)}
        
        <div class="footer">
            <p>Comprehensive Photo Analysis Tool v{self._get_version()}</p>
            <p>このレポートは自動生成されました</p>
        </div>
    </div>
</body>
</html>
"""
            
            return html_content
            
        except Exception as e:
            self.logger.error(f"HTML コンテンツ構築エラー: {e}")
            raise
    
    def _build_summary_cards(self, summary: Dict[str, Any]) -> str:
        """サマリーカードのHTML生成"""
        try:
            cards_html = ""
            
            assessment = summary.get("overall_assessment", {})
            
            if "brightness_change" in assessment:
                value = assessment["brightness_change"]
                class_name = "positive" if value > 0 else "negative" if value < 0 else "neutral"
                cards_html += f"""
                <div class="metric-card">
                    <h4>明度変化</h4>
                    <div class="value {class_name}">{value:+.3f}</div>
                </div>
                """
            
            if "contrast_change" in assessment:
                value = assessment["contrast_change"]
                class_name = "positive" if value > 0 else "negative" if value < 0 else "neutral"
                cards_html += f"""
                <div class="metric-card">
                    <h4>コントラスト変化</h4>
                    <div class="value {class_name}">{value:+.3f}</div>
                </div>
                """
            
            if "quality_score" in assessment:
                value = assessment["quality_score"]
                cards_html += f"""
                <div class="metric-card">
                    <h4>品質スコア (PSNR)</h4>
                    <div class="value">{value:.1f} dB</div>
                </div>
                """
            
            if "dominant_color_shift" in assessment:
                value = assessment["dominant_color_shift"]
                cards_html += f"""
                <div class="metric-card">
                    <h4>主要色変化</h4>
                    <div class="value">{value:.3f}</div>
                </div>
                """
            
            return cards_html
            
        except Exception as e:
            self.logger.error(f"サマリーカード生成エラー: {e}")
            return ""
    
    def _build_color_analysis_section(self, color_analysis: Dict[str, Any], visualizations: Dict[str, str]) -> str:
        """色彩解析セクションのHTML生成"""
        try:
            if not color_analysis:
                return ""
            
            section_html = """
        <div class="section">
            <h2>🎨 色彩解析</h2>
            """
            
            # ヒストグラム比較
            if "histograms" in visualizations:
                section_html += f"""
            <div class="chart-container">
                <h3>色相ヒストグラム比較</h3>
                <img src="data:image/png;base64,{visualizations['histograms']}" alt="ヒストグラム比較">
            </div>
                """
            
            # 統計比較
            if "statistics" in visualizations:
                section_html += f"""
            <div class="chart-container">
                <h3>統計値比較</h3>
                <img src="data:image/png;base64,{visualizations['statistics']}" alt="統計比較">
            </div>
                """
            
            # 色変化
            if "color_shifts" in visualizations:
                section_html += f"""
            <div class="chart-container">
                <h3>色変化解析</h3>
                <img src="data:image/png;base64,{visualizations['color_shifts']}" alt="色変化">
            </div>
                """
            
            # 主要色比較
            if "dominant_colors" in visualizations:
                section_html += f"""
            <div class="chart-container">
                <h3>主要色比較</h3>
                <img src="data:image/png;base64,{visualizations['dominant_colors']}" alt="主要色比較">
            </div>
                """
            
            section_html += """
        </div>
            """
            
            return section_html
            
        except Exception as e:
            self.logger.error(f"色彩解析セクション生成エラー: {e}")
            return ""
    
    def _build_advanced_analysis_section(self, advanced_analysis: Dict[str, Any], visualizations: Dict[str, str]) -> str:
        """高度解析セクションのHTML生成"""
        try:
            if not advanced_analysis or "error" in advanced_analysis:
                return ""
            
            section_html = """
        <div class="section">
            <h2>🔬 高度色彩解析</h2>
            """
            
            # Delta E2000セクション
            if "delta_e2000" in advanced_analysis and "delta_e2000" in visualizations:
                delta_e_data = advanced_analysis["delta_e2000"]
                perceptual = delta_e_data.get("perceptual_assessment", {})
                
                section_html += f"""
            <div class="subsection">
                <h3>Delta E2000 精密色差分析</h3>
                <div class="metric-grid">
                    <div class="metric-card">
                        <h4>平均色差</h4>
                        <div class="value">{delta_e_data.get('delta_e_statistics', {}).get('mean', 0):.3f}</div>
                    </div>
                    <div class="metric-card">
                        <h4>知覚カテゴリ</h4>
                        <div class="value">{perceptual.get('perception_category', 'N/A')}</div>
                    </div>
                    <div class="metric-card">
                        <h4>3 JND以上</h4>
                        <div class="value">{perceptual.get('jnd_distribution', {}).get('above_3_jnd_percent', 0):.1f}%</div>
                    </div>
                </div>
                <div class="chart-container">
                    <img src="data:image/png;base64,{visualizations['delta_e2000']}" alt="Delta E2000分析">
                </div>
            </div>
                """
            
            # 色温度セクション
            if "color_temperature" in advanced_analysis and "color_temperature" in visualizations:
                temp_data = advanced_analysis["color_temperature"]
                global_temp = temp_data.get("global_temperature", {})
                wb_data = temp_data.get("white_balance_analysis", {})
                
                section_html += f"""
            <div class="subsection">
                <h3>色温度・ホワイトバランス分析</h3>
                <div class="metric-grid">
                    <div class="metric-card">
                        <h4>色温度変化</h4>
                        <div class="value">{global_temp.get('change_kelvin', 0):+.0f}K</div>
                    </div>
                    <div class="metric-card">
                        <h4>処理後色温度</h4>
                        <div class="value">{global_temp.get('processed_kelvin', 5500):.0f}K</div>
                    </div>
                    <div class="metric-card">
                        <h4>WB主要シフト</h4>
                        <div class="value">{wb_data.get('dominant_shift', 'N/A')}</div>
                    </div>
                </div>
                <div class="chart-container">
                    <img src="data:image/png;base64,{visualizations['color_temperature']}" alt="色温度分析">
                </div>
            </div>
                """
            
            # 色域セクション
            if "color_gamut" in advanced_analysis and "color_gamut" in visualizations:
                gamut_data = advanced_analysis["color_gamut"]
                visual_gamut = gamut_data.get("visual_gamut_analysis", {})
                
                section_html += f"""
            <div class="subsection">
                <h3>色域分析</h3>
                <div class="metric-grid">
                    <div class="metric-card">
                        <h4>彩度変化</h4>
                        <div class="value">{visual_gamut.get('average_saturation_change', 0):+.1f}</div>
                    </div>
                    <div class="metric-card">
                        <h4>視覚的インパクト</h4>
                        <div class="value">{visual_gamut.get('visual_impact', 'N/A')}</div>
                    </div>
                    <div class="metric-card">
                        <h4>彩度強化</h4>
                        <div class="value">{'はい' if visual_gamut.get('saturation_enhancement', False) else 'いいえ'}</div>
                    </div>
                </div>
                <div class="chart-container">
                    <img src="data:image/png;base64,{visualizations['color_gamut']}" alt="色域分析">
                </div>
            </div>
                """
            
            # 色彩調和セクション（ultra精度の場合）
            if "color_harmony" in advanced_analysis and "color_harmony" in visualizations:
                harmony_data = advanced_analysis["color_harmony"]
                harmony_score = harmony_data.get("harmony_score", {})
                
                section_html += f"""
            <div class="subsection">
                <h3>色彩調和分析</h3>
                <div class="metric-grid">
                    <div class="metric-card">
                        <h4>調和スコア</h4>
                        <div class="value">{harmony_score.get('processed_harmony_score', 0):.3f}</div>
                    </div>
                    <div class="metric-card">
                        <h4>調和カテゴリ</h4>
                        <div class="value">{harmony_score.get('harmony_category', 'N/A')}</div>
                    </div>
                    <div class="metric-card">
                        <h4>調和改善</h4>
                        <div class="value">{harmony_score.get('harmony_improvement', 0):+.3f}</div>
                    </div>
                </div>
                <div class="chart-container">
                    <img src="data:image/png;base64,{visualizations['color_harmony']}" alt="色彩調和分析">
                </div>
            </div>
                """
            
            section_html += """
        </div>
            """
            
            return section_html
            
        except Exception as e:
            self.logger.error(f"高度解析セクション生成エラー: {e}")
            return ""
    
    def _generate_json_report(self, results: Dict[str, Any], output_dir: Path, output_name: str) -> Path:
        """JSON レポートの生成"""
        try:
            json_path = output_dir / f"{output_name}.json"
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            self.logger.debug(f"JSON レポート保存: {json_path}")
            return json_path
            
        except Exception as e:
            self.logger.error(f"JSON レポート生成エラー: {e}")
            raise
    
    def _generate_csv_report(self, results: Dict[str, Any], output_dir: Path, output_name: str) -> Path:
        """CSV レポートの生成"""
        try:
            # 簡易CSV形式でサマリーデータを出力
            csv_path = output_dir / f"{output_name}.csv"
            
            # ここではサマリー情報のみCSV化
            summary = results.get("color_analysis", {}).get("summary", {})
            assessment = summary.get("overall_assessment", {})
            
            with open(csv_path, 'w', encoding='utf-8') as f:
                f.write("metric,value\\n")
                for key, value in assessment.items():
                    f.write(f"{key},{value}\\n")
            
            self.logger.debug(f"CSV レポート保存: {csv_path}")
            return csv_path
            
        except Exception as e:
            self.logger.error(f"CSV レポート生成エラー: {e}")
            raise
    
    def _create_texture_analysis_visualizations(self, texture_data: Dict[str, Any], output_dir: Path, output_name: str) -> Dict[str, str]:
        """テクスチャ解析の可視化"""
        try:
            visualizations = {}
            
            # エッジ解析チャート
            if "edge_analysis" in texture_data:
                edge_viz = self._create_edge_analysis_chart(
                    texture_data["edge_analysis"], output_dir, f"{output_name}_edges"
                )
                if edge_viz:
                    visualizations["edge_analysis"] = edge_viz
            
            # シャープネス解析チャート
            if "sharpness_analysis" in texture_data:
                sharpness_viz = self._create_sharpness_analysis_chart(
                    texture_data["sharpness_analysis"], output_dir, f"{output_name}_sharpness"
                )
                if sharpness_viz:
                    visualizations["sharpness_analysis"] = sharpness_viz
            
            # ノイズ解析チャート
            if "noise_analysis" in texture_data:
                noise_viz = self._create_noise_analysis_chart(
                    texture_data["noise_analysis"], output_dir, f"{output_name}_noise"
                )
                if noise_viz:
                    visualizations["noise_analysis"] = noise_viz
            
            # 表面質感解析チャート
            if "surface_texture" in texture_data:
                surface_viz = self._create_surface_texture_chart(
                    texture_data["surface_texture"], output_dir, f"{output_name}_surface"
                )
                if surface_viz:
                    visualizations["surface_texture"] = surface_viz
            
            # Haralick特徴チャート
            if "haralick_features" in texture_data:
                haralick_viz = self._create_haralick_features_chart(
                    texture_data["haralick_features"], output_dir, f"{output_name}_haralick"
                )
                if haralick_viz:
                    visualizations["haralick_features"] = haralick_viz
            
            # 総合評価チャート
            if "overall_assessment" in texture_data:
                overall_viz = self._create_texture_overall_chart(
                    texture_data["overall_assessment"], output_dir, f"{output_name}_texture_overall"
                )
                if overall_viz:
                    visualizations["texture_overall"] = overall_viz
            
            return visualizations
            
        except Exception as e:
            self.logger.error(f"テクスチャ可視化生成エラー: {e}")
            return {}
    
    def _create_edge_analysis_chart(self, edge_data: Dict[str, Any], output_dir: Path, filename: str) -> str:
        """エッジ解析チャートの生成"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Edge Analysis', fontsize=16)
            
            # Canny エッジ解析
            canny_data = edge_data.get("canny", {})
            if canny_data:
                categories = ['Original', 'Processed']
                densities = [
                    canny_data.get("original_edge_density", 0),
                    canny_data.get("processed_edge_density", 0)
                ]
                colors = ['blue', 'orange']
                
                bars = axes[0, 0].bar(categories, densities, color=colors, alpha=0.7)
                axes[0, 0].set_title('Canny Edge Density')
                axes[0, 0].set_ylabel('Edge Density')
                axes[0, 0].grid(True, alpha=0.3)
                
                # 値をバーの上に表示
                for bar, density in zip(bars, densities):
                    height = bar.get_height()
                    axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                                  f'{density:.4f}', ha='center', va='bottom')
            
            # Sobel 強度解析
            sobel_data = edge_data.get("sobel", {})
            if sobel_data:
                categories = ['Original', 'Processed']
                magnitudes = [
                    sobel_data.get("original_magnitude_mean", 0),
                    sobel_data.get("processed_magnitude_mean", 0)
                ]
                
                axes[0, 1].bar(categories, magnitudes, color=['blue', 'orange'], alpha=0.7)
                axes[0, 1].set_title('Sobel Edge Magnitude')
                axes[0, 1].set_ylabel('Mean Magnitude')
                axes[0, 1].grid(True, alpha=0.3)
            
            # Laplacian 分散解析
            laplacian_data = edge_data.get("laplacian", {})
            if laplacian_data:
                categories = ['Original', 'Processed']
                variances = [
                    laplacian_data.get("original_variance", 0),
                    laplacian_data.get("processed_variance", 0)
                ]
                
                axes[1, 0].bar(categories, variances, color=['blue', 'orange'], alpha=0.7)
                axes[1, 0].set_title('Laplacian Variance')
                axes[1, 0].set_ylabel('Variance')
                axes[1, 0].grid(True, alpha=0.3)
            
            # 保存率・変化率サマリー
            preservation_ratio = canny_data.get("edge_preservation_ratio", 0)
            sobel_change = sobel_data.get("magnitude_change_ratio", 1)
            laplacian_change = laplacian_data.get("variance_change_ratio", 1)
            
            metrics = ['Edge Preservation', 'Sobel Change', 'Laplacian Change']
            values = [preservation_ratio, sobel_change, laplacian_change]
            colors_map = ['green' if v >= 0.8 else 'orange' if v >= 0.6 else 'red' for v in values]
            
            bars = axes[1, 1].bar(metrics, values, color=colors_map, alpha=0.7)
            axes[1, 1].set_title('Edge Analysis Summary')
            axes[1, 1].set_ylabel('Ratio')
            axes[1, 1].set_xticklabels(metrics, rotation=45, ha='right')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            
            # 画像として保存
            image_path = output_dir / f"{filename}.png"
            plt.savefig(image_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return self._encode_image_to_base64(image_path)
            
        except Exception as e:
            self.logger.error(f"エッジ解析チャート生成エラー: {e}")
            return ""
    
    def _create_sharpness_analysis_chart(self, sharpness_data: Dict[str, Any], output_dir: Path, filename: str) -> str:
        """シャープネス解析チャートの生成"""
        try:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle('Sharpness Analysis', fontsize=16)
            
            # Tenengrad シャープネス
            tenengrad_data = sharpness_data.get("tenengrad", {})
            if tenengrad_data:
                categories = ['Original', 'Processed']
                sharpness_values = [
                    tenengrad_data.get("original_sharpness", 0),
                    tenengrad_data.get("processed_sharpness", 0)
                ]
                
                axes[0].bar(categories, sharpness_values, color=['blue', 'orange'], alpha=0.7)
                axes[0].set_title('Tenengrad Sharpness')
                axes[0].set_ylabel('Sharpness Value')
                axes[0].grid(True, alpha=0.3)
            
            # Laplacian シャープネス
            laplacian_data = sharpness_data.get("laplacian_variance", {})
            if laplacian_data:
                categories = ['Original', 'Processed']
                sharpness_values = [
                    laplacian_data.get("original_sharpness", 0),
                    laplacian_data.get("processed_sharpness", 0)
                ]
                
                axes[1].bar(categories, sharpness_values, color=['blue', 'orange'], alpha=0.7)
                axes[1].set_title('Laplacian Variance Sharpness')
                axes[1].set_ylabel('Sharpness Value')
                axes[1].grid(True, alpha=0.3)
            
            # 高周波成分
            freq_data = sharpness_data.get("frequency_analysis", {})
            if freq_data:
                categories = ['Original', 'Processed']
                freq_energy = [
                    freq_data.get("original_high_freq_energy", 0),
                    freq_data.get("processed_high_freq_energy", 0)
                ]
                
                axes[2].bar(categories, freq_energy, color=['blue', 'orange'], alpha=0.7)
                axes[2].set_title('High Frequency Energy')
                axes[2].set_ylabel('Energy')
                axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 画像として保存
            image_path = output_dir / f"{filename}.png"
            plt.savefig(image_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return self._encode_image_to_base64(image_path)
            
        except Exception as e:
            self.logger.error(f"シャープネス解析チャート生成エラー: {e}")
            return ""
    
    def _create_noise_analysis_chart(self, noise_data: Dict[str, Any], output_dir: Path, filename: str) -> str:
        """ノイズ解析チャートの生成"""
        try:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle('Noise Analysis', fontsize=16)
            
            # 局所標準偏差ノイズ
            local_std_data = noise_data.get("local_std_noise", {})
            if local_std_data:
                categories = ['Original', 'Processed']
                noise_levels = [
                    local_std_data.get("original_noise_level", 0),
                    local_std_data.get("processed_noise_level", 0)
                ]
                
                bars = axes[0].bar(categories, noise_levels, color=['red', 'green'], alpha=0.7)
                axes[0].set_title('Local Standard Deviation Noise')
                axes[0].set_ylabel('Noise Level')
                axes[0].grid(True, alpha=0.3)
                
                # ノイズ削減率を表示
                reduction_ratio = local_std_data.get("noise_reduction_ratio", 0)
                axes[0].text(0.5, max(noise_levels) * 0.8, 
                           f'Reduction: {reduction_ratio:.2%}',
                           ha='center', transform=axes[0].transData)
            
            # Wavelet ノイズ
            wavelet_data = noise_data.get("wavelet_noise", {})
            if wavelet_data:
                categories = ['Original', 'Processed']
                noise_levels = [
                    wavelet_data.get("original_noise_level", 0),
                    wavelet_data.get("processed_noise_level", 0)
                ]
                
                axes[1].bar(categories, noise_levels, color=['red', 'green'], alpha=0.7)
                axes[1].set_title('Wavelet Noise Estimation')
                axes[1].set_ylabel('Noise Level')
                axes[1].grid(True, alpha=0.3)
            
            # 高周波ノイズ
            high_freq_data = noise_data.get("high_freq_noise", {})
            if high_freq_data:
                categories = ['Original', 'Processed']
                noise_power = [
                    high_freq_data.get("original_noise_power", 0),
                    high_freq_data.get("processed_noise_power", 0)
                ]
                
                axes[2].bar(categories, noise_power, color=['red', 'green'], alpha=0.7)
                axes[2].set_title('High Frequency Noise Power')
                axes[2].set_ylabel('Power')
                axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 画像として保存
            image_path = output_dir / f"{filename}.png"
            plt.savefig(image_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return self._encode_image_to_base64(image_path)
            
        except Exception as e:
            self.logger.error(f"ノイズ解析チャート生成エラー: {e}")
            return ""
    
    def _create_surface_texture_chart(self, surface_data: Dict[str, Any], output_dir: Path, filename: str) -> str:
        """表面質感解析チャートの生成"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Surface Texture Analysis', fontsize=16)
            
            # LBP 解析
            lbp_data = surface_data.get("lbp_analysis", {})
            if lbp_data:
                metrics = ['Uniformity', 'Pattern Similarity', 'Texture Entropy']
                original_values = [
                    lbp_data.get("original_uniformity", 0),
                    lbp_data.get("pattern_similarity", 0),
                    lbp_data.get("texture_entropy_original", 0)
                ]
                processed_values = [
                    lbp_data.get("processed_uniformity", 0),
                    lbp_data.get("pattern_similarity", 0),
                    lbp_data.get("texture_entropy_processed", 0)
                ]
                
                x = np.arange(len(metrics))
                width = 0.35
                
                axes[0, 0].bar(x - width/2, original_values, width, label='Original', alpha=0.7)
                axes[0, 0].bar(x + width/2, processed_values, width, label='Processed', alpha=0.7)
                axes[0, 0].set_title('LBP Analysis')
                axes[0, 0].set_ylabel('Value')
                axes[0, 0].set_xticks(x)
                axes[0, 0].set_xticklabels(metrics, rotation=45, ha='right')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
            
            # Gabor 解析
            gabor_data = surface_data.get("gabor_analysis", {})
            if gabor_data:
                categories = ['Original', 'Processed']
                energy_values = [
                    gabor_data.get("original_texture_energy", 0),
                    gabor_data.get("processed_texture_energy", 0)
                ]
                
                axes[0, 1].bar(categories, energy_values, color=['blue', 'orange'], alpha=0.7)
                axes[0, 1].set_title('Gabor Texture Energy')
                axes[0, 1].set_ylabel('Energy')
                axes[0, 1].grid(True, alpha=0.3)
                
                # 保存率を表示
                preservation_ratio = gabor_data.get("texture_preservation_ratio", 0)
                axes[0, 1].text(0.5, max(energy_values) * 0.8,
                              f'Preservation: {preservation_ratio:.2%}',
                              ha='center', transform=axes[0, 1].transData)
            
            # 粗さ解析
            roughness_data = surface_data.get("roughness_analysis", {})
            if roughness_data:
                categories = ['Original', 'Processed']
                roughness_values = [
                    roughness_data.get("original_roughness", 0),
                    roughness_data.get("processed_roughness", 0)
                ]
                
                axes[1, 0].bar(categories, roughness_values, color=['brown', 'green'], alpha=0.7)
                axes[1, 0].set_title('Surface Roughness')
                axes[1, 0].set_ylabel('Roughness')
                axes[1, 0].grid(True, alpha=0.3)
            
            # 方向別Gabor応答
            if gabor_data and "directional_responses_original" in gabor_data:
                directions = ['0°', '45°', '90°', '135°'] * 3  # 3つの周波数 × 4方向
                orig_responses = gabor_data.get("directional_responses_original", [])
                proc_responses = gabor_data.get("directional_responses_processed", [])
                
                if len(orig_responses) >= 12 and len(proc_responses) >= 12:
                    x = np.arange(12)
                    width = 0.35
                    
                    axes[1, 1].bar(x - width/2, orig_responses[:12], width, label='Original', alpha=0.7)
                    axes[1, 1].bar(x + width/2, proc_responses[:12], width, label='Processed', alpha=0.7)
                    axes[1, 1].set_title('Directional Gabor Responses')
                    axes[1, 1].set_ylabel('Response')
                    axes[1, 1].set_xlabel('Direction & Frequency')
                    axes[1, 1].legend()
                    axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 画像として保存
            image_path = output_dir / f"{filename}.png"
            plt.savefig(image_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return self._encode_image_to_base64(image_path)
            
        except Exception as e:
            self.logger.error(f"表面質感解析チャート生成エラー: {e}")
            return ""
    
    def _create_haralick_features_chart(self, haralick_data: Dict[str, Any], output_dir: Path, filename: str) -> str:
        """Haralick特徴チャートの生成"""
        try:
            # 主要なHaralick特徴のみプロット
            main_features = ['contrast', 'correlation', 'energy', 'homogeneity']
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Haralick Texture Features', fontsize=16)
            
            for i, feature in enumerate(main_features):
                if feature in haralick_data:
                    row, col = i // 2, i % 2
                    
                    feature_data = haralick_data[feature]
                    categories = ['Original', 'Processed']
                    values = [
                        feature_data.get("original", 0),
                        feature_data.get("processed", 0)
                    ]
                    
                    bars = axes[row, col].bar(categories, values, color=['blue', 'orange'], alpha=0.7)
                    axes[row, col].set_title(f'{feature.title()}')
                    axes[row, col].set_ylabel('Value')
                    axes[row, col].grid(True, alpha=0.3)
                    
                    # 変化率を表示
                    change_ratio = feature_data.get("change_ratio", 1)
                    axes[row, col].text(0.5, max(values) * 0.8,
                                      f'Change: {change_ratio:.2f}x',
                                      ha='center', transform=axes[row, col].transData)
            
            plt.tight_layout()
            
            # 画像として保存
            image_path = output_dir / f"{filename}.png"
            plt.savefig(image_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return self._encode_image_to_base64(image_path)
            
        except Exception as e:
            self.logger.error(f"Haralick特徴チャート生成エラー: {e}")
            return ""
    
    def _create_texture_overall_chart(self, overall_data: Dict[str, Any], output_dir: Path, filename: str) -> str:
        """テクスチャ総合評価チャートの生成"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle('Texture Analysis Overall Assessment', fontsize=16)
            
            # 個別スコア
            individual_scores = overall_data.get("individual_scores", {})
            if individual_scores:
                metrics = list(individual_scores.keys())
                scores = list(individual_scores.values())
                
                colors = ['green' if s >= 0.8 else 'orange' if s >= 0.6 else 'red' for s in scores]
                bars = ax1.bar(metrics, scores, color=colors, alpha=0.7)
                ax1.set_title('Individual Texture Metrics')
                ax1.set_ylabel('Score')
                ax1.set_xticklabels(metrics, rotation=45, ha='right')
                ax1.grid(True, alpha=0.3)
                ax1.set_ylim(0, 1.2)
                
                # スコアを表示
                for bar, score in zip(bars, scores):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                           f'{score:.3f}', ha='center', va='bottom')
            
            # 総合評価
            overall_score = overall_data.get("overall_score", 0)
            quality_assessment = overall_data.get("quality_assessment", "unknown")
            
            # 円グラフで総合スコア表示
            sizes = [overall_score, 1 - overall_score]
            colors_pie = ['green', 'lightgray']
            labels = ['Quality Score', 'Remaining']
            
            wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors_pie, 
                                              autopct='%1.1f%%', startangle=90)
            ax2.set_title(f'Overall Score: {overall_score:.3f}\nAssessment: {quality_assessment}')
            
            plt.tight_layout()
            
            # 画像として保存
            image_path = output_dir / f"{filename}.png"
            plt.savefig(image_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return self._encode_image_to_base64(image_path)
            
        except Exception as e:
            self.logger.error(f"テクスチャ総合評価チャート生成エラー: {e}")
            return ""
    
    def _create_impression_analysis_visualizations(self, impression_data: Dict[str, Any], output_dir: Path, output_name: str) -> Dict[str, str]:
        """印象・感情解析の可視化"""
        try:
            visualizations = {}
            
            # 色彩心理学チャート
            if "color_psychology" in impression_data:
                psychology_viz = self._create_color_psychology_chart(
                    impression_data["color_psychology"], output_dir, f"{output_name}_psychology"
                )
                if psychology_viz:
                    visualizations["color_psychology"] = psychology_viz
            
            # 明度・コントラスト印象チャート
            if "brightness_contrast_impression" in impression_data:
                brightness_viz = self._create_brightness_impression_chart(
                    impression_data["brightness_contrast_impression"], output_dir, f"{output_name}_brightness"
                )
                if brightness_viz:
                    visualizations["brightness_impression"] = brightness_viz
            
            # 美的評価チャート
            if "aesthetic_evaluation" in impression_data:
                aesthetic_viz = self._create_aesthetic_evaluation_chart(
                    impression_data["aesthetic_evaluation"], output_dir, f"{output_name}_aesthetic"
                )
                if aesthetic_viz:
                    visualizations["aesthetic_evaluation"] = aesthetic_viz
            
            # ムード・雰囲気チャート
            if "mood_atmosphere" in impression_data:
                mood_viz = self._create_mood_atmosphere_chart(
                    impression_data["mood_atmosphere"], output_dir, f"{output_name}_mood"
                )
                if mood_viz:
                    visualizations["mood_atmosphere"] = mood_viz
            
            # 感情変化チャート
            if "emotional_change" in impression_data:
                emotion_viz = self._create_emotional_change_chart(
                    impression_data["emotional_change"], output_dir, f"{output_name}_emotion"
                )
                if emotion_viz:
                    visualizations["emotional_change"] = emotion_viz
            
            # 総合印象チャート
            if "overall_impression" in impression_data:
                overall_viz = self._create_impression_overall_chart(
                    impression_data["overall_impression"], output_dir, f"{output_name}_impression_overall"
                )
                if overall_viz:
                    visualizations["impression_overall"] = overall_viz
            
            return visualizations
            
        except Exception as e:
            self.logger.error(f"印象可視化生成エラー: {e}")
            return {}
    
    def _create_color_psychology_chart(self, psychology_data: Dict[str, Any], output_dir: Path, filename: str) -> str:
        """色彩心理学チャートの生成"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Color Psychology Analysis', fontsize=16)
            
            # 感情スコア比較
            if 'original' in psychology_data and 'processed' in psychology_data:
                categories = ['Overall Emotion', 'Warmth', 'Emotion Intensity', 'Consistency']
                original_values = [
                    psychology_data['original'].get('overall_emotion_score', 0),
                    psychology_data['original'].get('warmth_score', 0),
                    psychology_data['original'].get('emotion_intensity', 0),
                    psychology_data['original'].get('emotion_consistency', 0)
                ]
                processed_values = [
                    psychology_data['processed'].get('overall_emotion_score', 0),
                    psychology_data['processed'].get('warmth_score', 0),
                    psychology_data['processed'].get('emotion_intensity', 0),
                    psychology_data['processed'].get('emotion_consistency', 0)
                ]
                
                x = np.arange(len(categories))
                width = 0.35
                
                axes[0, 0].bar(x - width/2, original_values, width, label='Original', alpha=0.8)
                axes[0, 0].bar(x + width/2, processed_values, width, label='Processed', alpha=0.8)
                axes[0, 0].set_title('Emotional Metrics Comparison')
                axes[0, 0].set_ylabel('Score')
                axes[0, 0].set_xticks(x)
                axes[0, 0].set_xticklabels(categories, rotation=45, ha='right')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
            
            # 感情分布（元画像）
            if 'original' in psychology_data and 'emotion_distribution' in psychology_data['original']:
                emotion_dist = psychology_data['original']['emotion_distribution']
                if emotion_dist:
                    emotions = list(emotion_dist.keys())
                    values = list(emotion_dist.values())
                    
                    colors = ['red', 'orange', 'green', 'blue', 'purple', 'pink', 'brown'][:len(emotions)]
                    axes[0, 1].pie(values, labels=emotions, colors=colors, autopct='%1.1f%%')
                    axes[0, 1].set_title('Original - Emotion Distribution')
            
            # 感情分布（処理済み画像）
            if 'processed' in psychology_data and 'emotion_distribution' in psychology_data['processed']:
                emotion_dist = psychology_data['processed']['emotion_distribution']
                if emotion_dist:
                    emotions = list(emotion_dist.keys())
                    values = list(emotion_dist.values())
                    
                    colors = ['red', 'orange', 'green', 'blue', 'purple', 'pink', 'brown'][:len(emotions)]
                    axes[1, 0].pie(values, labels=emotions, colors=colors, autopct='%1.1f%%')
                    axes[1, 0].set_title('Processed - Emotion Distribution')
            
            # 感情変化サマリー
            if 'changes' in psychology_data:
                changes = psychology_data['changes']
                change_metrics = ['Emotion Change', 'Warmth Change', 'Intensity Change']
                change_values = [
                    changes.get('emotion_change', 0),
                    changes.get('warmth_change', 0),
                    changes.get('intensity_change', 0)
                ]
                
                colors = ['green' if v > 0 else 'red' if v < 0 else 'gray' for v in change_values]
                bars = axes[1, 1].bar(change_metrics, change_values, color=colors, alpha=0.7)
                axes[1, 1].set_title('Emotional Changes')
                axes[1, 1].set_ylabel('Change Amount')
                axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
                axes[1, 1].grid(True, alpha=0.3)
                
                # 値をバーの上に表示
                for bar, value in zip(bars, change_values):
                    height = bar.get_height()
                    axes[1, 1].text(bar.get_x() + bar.get_width()/2., 
                                   height + (0.01 if height >= 0 else -0.03),
                                   f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top')
            
            plt.tight_layout()
            
            # 画像として保存
            image_path = output_dir / f"{filename}.png"
            plt.savefig(image_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return self._encode_image_to_base64(image_path)
            
        except Exception as e:
            self.logger.error(f"色彩心理学チャート生成エラー: {e}")
            return ""
    
    def _create_brightness_impression_chart(self, brightness_data: Dict[str, Any], output_dir: Path, filename: str) -> str:
        """明度・コントラスト印象チャートの生成"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Brightness & Contrast Impression Analysis', fontsize=16)
            
            # 明度・コントラスト値比較
            if 'original' in brightness_data and 'processed' in brightness_data:
                categories = ['Brightness Mean', 'Contrast RMS', 'Dynamic Range', 'Local Contrast']
                original_values = [
                    brightness_data['original'].get('brightness_mean', 0),
                    brightness_data['original'].get('contrast_rms', 0),
                    brightness_data['original'].get('dynamic_range', 0),
                    brightness_data['original'].get('local_contrast_mean', 0)
                ]
                processed_values = [
                    brightness_data['processed'].get('brightness_mean', 0),
                    brightness_data['processed'].get('contrast_rms', 0),
                    brightness_data['processed'].get('dynamic_range', 0),
                    brightness_data['processed'].get('local_contrast_mean', 0)
                ]
                
                x = np.arange(len(categories))
                width = 0.35
                
                axes[0, 0].bar(x - width/2, original_values, width, label='Original', alpha=0.8)
                axes[0, 0].bar(x + width/2, processed_values, width, label='Processed', alpha=0.8)
                axes[0, 0].set_title('Brightness & Contrast Metrics')
                axes[0, 0].set_ylabel('Value')
                axes[0, 0].set_xticks(x)
                axes[0, 0].set_xticklabels(categories, rotation=45, ha='right')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
            
            # 明度分布（元画像）
            if 'original' in brightness_data and 'brightness_distribution' in brightness_data['original']:
                dist = brightness_data['original']['brightness_distribution']
                bins = np.arange(len(dist))
                axes[0, 1].bar(bins, dist, alpha=0.7, color='blue')
                axes[0, 1].set_title('Original - Brightness Distribution')
                axes[0, 1].set_xlabel('Brightness Bins')
                axes[0, 1].set_ylabel('Frequency')
                axes[0, 1].grid(True, alpha=0.3)
            
            # 明度分布（処理済み画像）
            if 'processed' in brightness_data and 'brightness_distribution' in brightness_data['processed']:
                dist = brightness_data['processed']['brightness_distribution']
                bins = np.arange(len(dist))
                axes[1, 0].bar(bins, dist, alpha=0.7, color='orange')
                axes[1, 0].set_title('Processed - Brightness Distribution')
                axes[1, 0].set_xlabel('Brightness Bins')
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].grid(True, alpha=0.3)
            
            # ムードスコア比較
            if 'original' in brightness_data and 'processed' in brightness_data:
                categories = ['Original', 'Processed']
                mood_scores = [
                    brightness_data['original'].get('overall_mood_score', 0),
                    brightness_data['processed'].get('overall_mood_score', 0)
                ]
                
                colors = ['lightblue', 'orange']
                bars = axes[1, 1].bar(categories, mood_scores, color=colors, alpha=0.7)
                axes[1, 1].set_title('Overall Mood Score')
                axes[1, 1].set_ylabel('Mood Score')
                axes[1, 1].grid(True, alpha=0.3)
                
                # 変化量を表示
                if 'changes' in brightness_data:
                    mood_change = brightness_data['changes'].get('mood_change', 0)
                    axes[1, 1].text(0.5, max(mood_scores) * 0.8,
                                   f'Change: {mood_change:+.3f}',
                                   ha='center', transform=axes[1, 1].transData)
            
            plt.tight_layout()
            
            # 画像として保存
            image_path = output_dir / f"{filename}.png"
            plt.savefig(image_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return self._encode_image_to_base64(image_path)
            
        except Exception as e:
            self.logger.error(f"明度印象チャート生成エラー: {e}")
            return ""
    
    def _create_aesthetic_evaluation_chart(self, aesthetic_data: Dict[str, Any], output_dir: Path, filename: str) -> str:
        """美的評価チャートの生成"""
        try:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle('Aesthetic Evaluation', fontsize=16)
            
            # 美的指標比較
            if 'original' in aesthetic_data and 'processed' in aesthetic_data:
                metrics = ['Composition', 'Color Harmony', 'Balance', 'Golden Ratio', 'Complexity', 'Unity']
                original_values = [
                    aesthetic_data['original'].get('composition_score', 0),
                    aesthetic_data['original'].get('color_harmony_score', 0),
                    aesthetic_data['original'].get('balance_score', 0),
                    aesthetic_data['original'].get('golden_ratio_score', 0),
                    aesthetic_data['original'].get('complexity_score', 0),
                    aesthetic_data['original'].get('unity_score', 0)
                ]
                processed_values = [
                    aesthetic_data['processed'].get('composition_score', 0),
                    aesthetic_data['processed'].get('color_harmony_score', 0),
                    aesthetic_data['processed'].get('balance_score', 0),
                    aesthetic_data['processed'].get('golden_ratio_score', 0),
                    aesthetic_data['processed'].get('complexity_score', 0),
                    aesthetic_data['processed'].get('unity_score', 0)
                ]
                
                x = np.arange(len(metrics))
                width = 0.35
                
                axes[0].bar(x - width/2, original_values, width, label='Original', alpha=0.8)
                axes[0].bar(x + width/2, processed_values, width, label='Processed', alpha=0.8)
                axes[0].set_title('Aesthetic Metrics Comparison')
                axes[0].set_ylabel('Score')
                axes[0].set_xticks(x)
                axes[0].set_xticklabels(metrics, rotation=45, ha='right')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
            
            # 総合美的スコア
            if 'original' in aesthetic_data and 'processed' in aesthetic_data:
                categories = ['Original', 'Processed']
                overall_scores = [
                    aesthetic_data['original'].get('overall_aesthetic_score', 0),
                    aesthetic_data['processed'].get('overall_aesthetic_score', 0)
                ]
                
                colors = ['lightcoral', 'lightgreen']
                bars = axes[1].bar(categories, overall_scores, color=colors, alpha=0.7)
                axes[1].set_title('Overall Aesthetic Score')
                axes[1].set_ylabel('Score')
                axes[1].set_ylim(0, 1)
                axes[1].grid(True, alpha=0.3)
                
                # スコアを表示
                for bar, score in zip(bars, overall_scores):
                    height = bar.get_height()
                    axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                               f'{score:.3f}', ha='center', va='bottom')
            
            # 美的変化
            if 'changes' in aesthetic_data:
                changes = aesthetic_data['changes']
                change_metrics = ['Aesthetic', 'Composition', 'Harmony', 'Balance']
                change_values = [
                    changes.get('aesthetic_change', 0),
                    changes.get('composition_change', 0),
                    changes.get('harmony_change', 0),
                    changes.get('balance_change', 0)
                ]
                
                colors = ['green' if v > 0 else 'red' if v < 0 else 'gray' for v in change_values]
                axes[2].bar(change_metrics, change_values, color=colors, alpha=0.7)
                axes[2].set_title('Aesthetic Changes')
                axes[2].set_ylabel('Change')
                axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
                axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 画像として保存
            image_path = output_dir / f"{filename}.png"
            plt.savefig(image_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return self._encode_image_to_base64(image_path)
            
        except Exception as e:
            self.logger.error(f"美的評価チャート生成エラー: {e}")
            return ""
    
    def _create_mood_atmosphere_chart(self, mood_data: Dict[str, Any], output_dir: Path, filename: str) -> str:
        """ムード・雰囲気チャートの生成"""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle('Mood & Atmosphere Analysis', fontsize=16)
            
            # ムード指標比較
            if 'original' in mood_data and 'processed' in mood_data:
                orig_indicators = mood_data['original'].get('mood_indicators', {})
                proc_indicators = mood_data['processed'].get('mood_indicators', {})
                
                if orig_indicators and proc_indicators:
                    indicators = list(orig_indicators.keys())
                    original_values = list(orig_indicators.values())
                    processed_values = list(proc_indicators.values())
                    
                    x = np.arange(len(indicators))
                    width = 0.35
                    
                    axes[0].bar(x - width/2, original_values, width, label='Original', alpha=0.8)
                    axes[0].bar(x + width/2, processed_values, width, label='Processed', alpha=0.8)
                    axes[0].set_title('Mood Indicators Comparison')
                    axes[0].set_ylabel('Score')
                    axes[0].set_xticks(x)
                    axes[0].set_xticklabels(indicators, rotation=45, ha='right')
                    axes[0].legend()
                    axes[0].grid(True, alpha=0.3)
            
            # 雰囲気スコア
            if 'original' in mood_data and 'processed' in mood_data:
                categories = ['Original', 'Processed']
                atmosphere_scores = [
                    mood_data['original'].get('atmosphere_score', 0),
                    mood_data['processed'].get('atmosphere_score', 0)
                ]
                
                colors = ['skyblue', 'lightcoral']
                bars = axes[1].bar(categories, atmosphere_scores, color=colors, alpha=0.7)
                axes[1].set_title('Atmosphere Score')
                axes[1].set_ylabel('Score')
                axes[1].grid(True, alpha=0.3)
                
                # ドミナントムードを表示
                orig_mood = mood_data['original'].get('dominant_mood', 'unknown')
                proc_mood = mood_data['processed'].get('dominant_mood', 'unknown')
                axes[1].text(0, atmosphere_scores[0] + 0.05, orig_mood, ha='center')
                axes[1].text(1, atmosphere_scores[1] + 0.05, proc_mood, ha='center')
            
            plt.tight_layout()
            
            # 画像として保存
            image_path = output_dir / f"{filename}.png"
            plt.savefig(image_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return self._encode_image_to_base64(image_path)
            
        except Exception as e:
            self.logger.error(f"ムード雰囲気チャート生成エラー: {e}")
            return ""
    
    def _create_emotional_change_chart(self, emotion_data: Dict[str, Any], output_dir: Path, filename: str) -> str:
        """感情変化チャートの生成"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle('Emotional Change Analysis (LUT Effects)', fontsize=16)
            
            # LUT変化指標
            change_metrics = ['Hue Shift', 'Saturation Change', 'Brightness Change', 'Emotion Vector']
            change_values = [
                emotion_data.get('hue_shift', 0),
                emotion_data.get('saturation_change', 0),
                emotion_data.get('brightness_change', 0),
                emotion_data.get('emotion_vector_change', 0)
            ]
            
            colors = ['red', 'green', 'blue', 'purple']
            bars = ax1.bar(change_metrics, change_values, color=colors, alpha=0.7)
            ax1.set_title('LUT Effect Components')
            ax1.set_ylabel('Change Amount')
            ax1.set_xticklabels(change_metrics, rotation=45, ha='right')
            ax1.grid(True, alpha=0.3)
            
            # 値を表示
            for bar, value in zip(bars, change_values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
            
            # LUT効果評価
            impact_strength = emotion_data.get('emotional_impact_strength', 0)
            effectiveness = emotion_data.get('lut_effectiveness', 0)
            change_direction = emotion_data.get('change_direction', 'neutral')
            
            # 円グラフでLUT効果の評価
            if effectiveness > 0:
                sizes = [effectiveness, 1 - effectiveness]
                labels = ['Effective', 'Room for Improvement']
                colors_pie = ['lightgreen', 'lightgray']
                
                ax2.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
                ax2.set_title(f'LUT Effectiveness\nDirection: {change_direction}\nStrength: {impact_strength:.3f}')
            
            plt.tight_layout()
            
            # 画像として保存
            image_path = output_dir / f"{filename}.png"
            plt.savefig(image_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return self._encode_image_to_base64(image_path)
            
        except Exception as e:
            self.logger.error(f"感情変化チャート生成エラー: {e}")
            return ""
    
    def _create_impression_overall_chart(self, overall_data: Dict[str, Any], output_dir: Path, filename: str) -> str:
        """印象総合評価チャートの生成"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle('Overall Impression Assessment', fontsize=16)
            
            # 個別印象スコア
            individual_scores = overall_data.get("individual_scores", {})
            if individual_scores:
                metrics = list(individual_scores.keys())
                scores = list(individual_scores.values())
                
                colors = ['green' if s >= 0.7 else 'orange' if s >= 0.5 else 'red' for s in scores]
                bars = ax1.bar(metrics, scores, color=colors, alpha=0.7)
                ax1.set_title('Individual Impression Metrics')
                ax1.set_ylabel('Score')
                ax1.set_xticklabels(metrics, rotation=45, ha='right')
                ax1.grid(True, alpha=0.3)
                ax1.set_ylim(0, 1.2)
                
                # スコアを表示
                for bar, score in zip(bars, scores):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                           f'{score:.3f}', ha='center', va='bottom')
            
            # 総合印象評価
            overall_score = overall_data.get("overall_impression_score", 0)
            impression_assessment = overall_data.get("impression_assessment", "unknown")
            
            # 円グラフで総合印象表示
            sizes = [overall_score, 1 - overall_score]
            colors_pie = ['gold', 'lightgray']
            labels = ['Impression Quality', 'Remaining']
            
            wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors_pie, 
                                              autopct='%1.1f%%', startangle=90)
            ax2.set_title(f'Overall Impression Score: {overall_score:.3f}\nAssessment: {impression_assessment}')
            
            plt.tight_layout()
            
            # 画像として保存
            image_path = output_dir / f"{filename}.png"
            plt.savefig(image_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return self._encode_image_to_base64(image_path)
            
        except Exception as e:
            self.logger.error(f"印象総合評価チャート生成エラー: {e}")
            return ""

    def _get_version(self) -> str:
        """バージョン情報を取得"""
        try:
            from .. import __version__
            return __version__
        except ImportError:
            return "0.1.0"