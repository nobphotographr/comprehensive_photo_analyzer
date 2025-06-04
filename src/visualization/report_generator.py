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
    
    def _get_version(self) -> str:
        """バージョン情報を取得"""
        try:
            from .. import __version__
            return __version__
        except ImportError:
            return "0.1.0"