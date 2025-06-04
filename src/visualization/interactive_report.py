"""
インタラクティブレポート生成モジュール

Plotly、Chart.js等を使用した動的で操作可能なHTMLレポートを生成
"""

import json
import base64
from pathlib import Path
from typing import Dict, Any, List
import logging
from datetime import datetime

class InteractiveReportGenerator:
    """インタラクティブレポート生成クラス"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初期化
        
        Args:
            config: 設定辞書
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # レポート設定
        report_config = config.get('interactive_report', {})
        self.enable_plotly = report_config.get('enable_plotly', True)
        self.enable_chartjs = report_config.get('enable_chartjs', True)
        self.enable_comparison_slider = report_config.get('enable_comparison_slider', True)
        self.enable_export_features = report_config.get('enable_export_features', True)
        self.theme = report_config.get('theme', 'modern')
    
    def generate_interactive_report(self, analysis_results: Dict[str, Any], 
                                  original_path: str, processed_path: str,
                                  output_dir: Path, output_name: str) -> str:
        """
        インタラクティブHTMLレポートの生成
        
        Args:
            analysis_results: 解析結果
            original_path: 元画像パス
            processed_path: 処理済み画像パス
            output_dir: 出力ディレクトリ
            output_name: 出力ファイル名
            
        Returns:
            生成されたレポートファイルのパス
        """
        try:
            self.logger.info("インタラクティブレポート生成開始")
            
            # 画像データのエンコード
            image_data = self._encode_images(original_path, processed_path)
            
            # インタラクティブチャートデータの準備
            chart_data = self._prepare_chart_data(analysis_results)
            
            # HTMLコンテンツの構築
            html_content = self._build_interactive_html(
                analysis_results, image_data, chart_data, 
                original_path, processed_path
            )
            
            # ファイル保存
            report_path = output_dir / f"{output_name}_interactive.html"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"インタラクティブレポート生成完了: {report_path}")
            return str(report_path)
            
        except Exception as e:
            self.logger.error(f"インタラクティブレポート生成エラー: {e}")
            raise
    
    def _encode_images(self, original_path: str, processed_path: str) -> Dict[str, str]:
        """画像をbase64エンコード"""
        image_data = {}
        
        try:
            with open(original_path, 'rb') as f:
                original_b64 = base64.b64encode(f.read()).decode('utf-8')
                image_data['original'] = original_b64
        except Exception as e:
            self.logger.warning(f"元画像エンコードエラー: {e}")
            image_data['original'] = ""
        
        try:
            with open(processed_path, 'rb') as f:
                processed_b64 = base64.b64encode(f.read()).decode('utf-8')
                image_data['processed'] = processed_b64
        except Exception as e:
            self.logger.warning(f"処理済み画像エンコードエラー: {e}")
            image_data['processed'] = ""
        
        return image_data
    
    def _prepare_chart_data(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """インタラクティブチャート用データの準備"""
        chart_data = {}
        
        # 色彩解析データ
        if 'color_analysis' in analysis_results:
            chart_data['color'] = self._prepare_color_chart_data(
                analysis_results['color_analysis']
            )
        
        # テクスチャ解析データ
        if 'texture_analysis' in analysis_results:
            chart_data['texture'] = self._prepare_texture_chart_data(
                analysis_results['texture_analysis']
            )
        
        # 印象解析データ
        if 'impression_analysis' in analysis_results:
            chart_data['impression'] = self._prepare_impression_chart_data(
                analysis_results['impression_analysis']
            )
        
        return chart_data
    
    def _prepare_color_chart_data(self, color_data: Dict[str, Any]) -> Dict[str, Any]:
        """色彩解析チャートデータの準備"""
        chart_data = {}
        
        # ヒストグラムデータ
        if 'histograms' in color_data:
            histograms = color_data['histograms']
            chart_data['histograms'] = {
                'original': {
                    'red': histograms.get('original', {}).get('red', []),
                    'green': histograms.get('original', {}).get('green', []),
                    'blue': histograms.get('original', {}).get('blue', [])
                },
                'processed': {
                    'red': histograms.get('processed', {}).get('red', []),
                    'green': histograms.get('processed', {}).get('green', []),
                    'blue': histograms.get('processed', {}).get('blue', [])
                }
            }
        
        # 統計データ
        if 'basic_statistics' in color_data:
            stats = color_data['basic_statistics']
            chart_data['statistics'] = {
                'original': {
                    'mean_rgb': stats.get('original', {}).get('mean_rgb', [0, 0, 0]),
                    'std_rgb': stats.get('original', {}).get('std_rgb', [0, 0, 0])
                },
                'processed': {
                    'mean_rgb': stats.get('processed', {}).get('mean_rgb', [0, 0, 0]),
                    'std_rgb': stats.get('processed', {}).get('std_rgb', [0, 0, 0])
                }
            }
        
        return chart_data
    
    def _prepare_texture_chart_data(self, texture_data: Dict[str, Any]) -> Dict[str, Any]:
        """テクスチャ解析チャートデータの準備"""
        chart_data = {}
        
        # エッジ解析データ
        if 'edge_analysis' in texture_data:
            edge_data = texture_data['edge_analysis']
            chart_data['edge_metrics'] = []
            
            if 'canny' in edge_data:
                chart_data['edge_metrics'].append({
                    'method': 'Canny',
                    'original': edge_data['canny'].get('original_edge_density', 0),
                    'processed': edge_data['canny'].get('processed_edge_density', 0)
                })
            
            if 'sobel' in edge_data:
                chart_data['edge_metrics'].append({
                    'method': 'Sobel',
                    'original': edge_data['sobel'].get('original_magnitude_mean', 0),
                    'processed': edge_data['sobel'].get('processed_magnitude_mean', 0)
                })
        
        # 総合評価データ
        if 'overall_assessment' in texture_data:
            overall = texture_data['overall_assessment']
            chart_data['overall_score'] = overall.get('overall_score', 0)
            chart_data['individual_scores'] = overall.get('individual_scores', {})
        
        return chart_data
    
    def _prepare_impression_chart_data(self, impression_data: Dict[str, Any]) -> Dict[str, Any]:
        """印象解析チャートデータの準備"""
        chart_data = {}
        
        # 色彩心理学データ
        if 'color_psychology' in impression_data:
            psychology = impression_data['color_psychology']
            chart_data['emotion_scores'] = {
                'original': psychology.get('original', {}).get('overall_emotion_score', 0),
                'processed': psychology.get('processed', {}).get('overall_emotion_score', 0)
            }
            
            # 感情分布
            if 'original' in psychology and 'emotion_distribution' in psychology['original']:
                chart_data['emotion_distribution'] = {
                    'original': psychology['original']['emotion_distribution'],
                    'processed': psychology.get('processed', {}).get('emotion_distribution', {})
                }
        
        # 美的評価データ
        if 'aesthetic_evaluation' in impression_data:
            aesthetic = impression_data['aesthetic_evaluation']
            chart_data['aesthetic_scores'] = {
                'original': aesthetic.get('original', {}).get('overall_aesthetic_score', 0),
                'processed': aesthetic.get('processed', {}).get('overall_aesthetic_score', 0)
            }
        
        return chart_data
    
    def _build_interactive_html(self, analysis_results: Dict[str, Any], 
                               image_data: Dict[str, str], chart_data: Dict[str, Any],
                               original_path: str, processed_path: str) -> str:
        """インタラクティブHTMLコンテンツの構築"""
        
        # ファイル名の取得
        orig_name = Path(original_path).name
        proc_name = Path(processed_path).name
        timestamp = datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')
        
        html_content = f'''<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interactive Photo Analysis Report</title>
    <script src="https://cdn.plot.ly/plotly-2.25.2.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        {self._get_css_styles()}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Interactive Photo Analysis Report</h1>
            <div class="subtitle">
                <p>生成日時: {timestamp}</p>
                <p>解析対象: <span class="filename">{orig_name}</span> ↔ <span class="filename">{proc_name}</span></p>
            </div>
        </div>
        
        <!-- 画像比較セクション -->
        <div class="section">
            <h2>🖼️ Image Comparison</h2>
            {self._generate_image_comparison_section(image_data)}
        </div>
        
        <!-- タブナビゲーション -->
        <div class="tab-container">
            <div class="tab-nav">
                <button class="tab-btn active" onclick="showTab('color')">色彩解析</button>
                <button class="tab-btn" onclick="showTab('texture')">テクスチャ解析</button>
                <button class="tab-btn" onclick="showTab('impression')">印象解析</button>
                <button class="tab-btn" onclick="showTab('summary')">総合評価</button>
            </div>
            
            <!-- 色彩解析タブ -->
            <div id="color-tab" class="tab-content active">
                <h2>🎨 Color Analysis</h2>
                {self._generate_color_analysis_section(chart_data.get('color', {}))}
            </div>
            
            <!-- テクスチャ解析タブ -->
            <div id="texture-tab" class="tab-content">
                <h2>🔍 Texture Analysis</h2>
                {self._generate_texture_analysis_section(chart_data.get('texture', {}))}
            </div>
            
            <!-- 印象解析タブ -->
            <div id="impression-tab" class="tab-content">
                <h2>💭 Impression Analysis</h2>
                {self._generate_impression_analysis_section(chart_data.get('impression', {}))}
            </div>
            
            <!-- 総合評価タブ -->
            <div id="summary-tab" class="tab-content">
                <h2>📈 Summary</h2>
                {self._generate_summary_section(analysis_results)}
            </div>
        </div>
    </div>
    
    <script>
        {self._get_javascript_functions()}
        
        // チャートデータ
        const chartData = {json.dumps(chart_data, ensure_ascii=False)};
        
        // チャートの初期化
        initializeCharts();
    </script>
</body>
</html>'''
        
        return html_content
    
    def _get_css_styles(self) -> str:
        """CSSスタイルの取得"""
        return '''
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-top: 20px;
            margin-bottom: 20px;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: linear-gradient(45deg, #4CAF50, #45a049);
            color: white;
            border-radius: 10px;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .subtitle {
            opacity: 0.9;
        }
        
        .filename {
            background: rgba(255,255,255,0.2);
            padding: 2px 8px;
            border-radius: 4px;
            font-family: monospace;
        }
        
        .section {
            margin: 30px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #4CAF50;
        }
        
        .image-comparison {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 20px 0;
        }
        
        .image-container {
            text-align: center;
            padding: 15px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .image-container img {
            max-width: 100%;
            height: auto;
            border-radius: 4px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        }
        
        .image-container h3 {
            margin: 10px 0 5px 0;
            color: #333;
        }
        
        .tab-container {
            margin-top: 30px;
        }
        
        .tab-nav {
            display: flex;
            background: #e9ecef;
            border-radius: 8px 8px 0 0;
            padding: 5px;
        }
        
        .tab-btn {
            flex: 1;
            padding: 12px 20px;
            border: none;
            background: transparent;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            border-radius: 4px;
            transition: all 0.3s ease;
        }
        
        .tab-btn:hover {
            background: rgba(76, 175, 80, 0.1);
        }
        
        .tab-btn.active {
            background: #4CAF50;
            color: white;
        }
        
        .tab-content {
            display: none;
            padding: 30px;
            background: white;
            border-radius: 0 0 8px 8px;
            border: 1px solid #e9ecef;
            border-top: none;
        }
        
        .tab-content.active {
            display: block;
        }
        
        .chart-container {
            margin: 20px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
        }
        
        .chart-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        
        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            text-align: center;
        }
        
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #4CAF50;
        }
        
        .metric-label {
            color: #666;
            margin-top: 5px;
        }
        
        .comparison-slider {
            margin: 20px 0;
        }
        
        .slider {
            width: 100%;
            height: 5px;
            border-radius: 5px;
            background: #ddd;
            outline: none;
            opacity: 0.7;
            transition: opacity 0.2s;
        }
        
        .slider:hover {
            opacity: 1;
        }
        
        .slider::-webkit-slider-thumb {
            appearance: none;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: #4CAF50;
            cursor: pointer;
        }
        '''
    
    def _generate_image_comparison_section(self, image_data: Dict[str, str]) -> str:
        """画像比較セクションの生成"""
        return f'''
        <div class="image-comparison">
            <div class="image-container">
                <img src="data:image/jpeg;base64,{image_data.get('original', '')}" alt="Original Image" id="original-img">
                <h3>Original Image</h3>
                <p>元画像</p>
            </div>
            <div class="image-container">
                <img src="data:image/jpeg;base64,{image_data.get('processed', '')}" alt="Processed Image" id="processed-img">
                <h3>Processed Image</h3>
                <p>処理済み画像</p>
            </div>
        </div>
        
        {self._generate_comparison_slider() if self.enable_comparison_slider else ""}
        '''
    
    def _generate_comparison_slider(self) -> str:
        """比較スライダーの生成"""
        return '''
        <div class="comparison-slider">
            <label for="comparison-range">比較スライダー:</label>
            <input type="range" id="comparison-range" class="slider" min="0" max="100" value="50">
            <p>左右にスライドして画像を比較できます</p>
        </div>
        '''
    
    def _generate_color_analysis_section(self, color_data: Dict[str, Any]) -> str:
        """色彩解析セクションの生成"""
        return f'''
        <div class="chart-grid">
            <div class="chart-container">
                <h3>RGB Histogram</h3>
                <div id="rgb-histogram" style="height: 400px;"></div>
            </div>
            <div class="chart-container">
                <h3>Color Statistics</h3>
                <div id="color-stats" style="height: 400px;"></div>
            </div>
        </div>
        
        <div class="chart-grid">
            <div class="metric-card">
                <div class="metric-value" id="color-change-value">-</div>
                <div class="metric-label">Overall Color Change</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="brightness-change-value">-</div>
                <div class="metric-label">Brightness Change</div>
            </div>
        </div>
        '''
    
    def _generate_texture_analysis_section(self, texture_data: Dict[str, Any]) -> str:
        """テクスチャ解析セクションの生成"""
        return '''
        <div class="chart-grid">
            <div class="chart-container">
                <h3>Edge Detection Metrics</h3>
                <div id="edge-metrics" style="height: 400px;"></div>
            </div>
            <div class="chart-container">
                <h3>Texture Quality Score</h3>
                <div id="texture-score" style="height: 400px;"></div>
            </div>
        </div>
        '''
    
    def _generate_impression_analysis_section(self, impression_data: Dict[str, Any]) -> str:
        """印象解析セクションの生成"""
        return '''
        <div class="chart-grid">
            <div class="chart-container">
                <h3>Emotion Analysis</h3>
                <div id="emotion-chart" style="height: 400px;"></div>
            </div>
            <div class="chart-container">
                <h3>Aesthetic Evaluation</h3>
                <div id="aesthetic-chart" style="height: 400px;"></div>
            </div>
        </div>
        '''
    
    def _generate_summary_section(self, analysis_results: Dict[str, Any]) -> str:
        """総合評価セクションの生成"""
        return '''
        <div class="chart-container">
            <h3>Overall Analysis Summary</h3>
            <div id="summary-chart" style="height: 500px;"></div>
        </div>
        
        <div class="chart-grid">
            <div class="metric-card">
                <div class="metric-value" id="overall-score-value">-</div>
                <div class="metric-label">Overall Quality Score</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="processing-time-value">-</div>
                <div class="metric-label">Processing Time</div>
            </div>
        </div>
        '''
    
    def _get_javascript_functions(self) -> str:
        """JavaScript関数の取得"""
        return '''
        function showTab(tabName) {
            // Hide all tab contents
            const tabContents = document.querySelectorAll('.tab-content');
            tabContents.forEach(content => content.classList.remove('active'));
            
            // Remove active class from all tab buttons
            const tabBtns = document.querySelectorAll('.tab-btn');
            tabBtns.forEach(btn => btn.classList.remove('active'));
            
            // Show selected tab content
            document.getElementById(tabName + '-tab').classList.add('active');
            
            // Add active class to clicked button
            event.target.classList.add('active');
        }
        
        function initializeCharts() {
            // Initialize color charts
            if (chartData.color) {
                initializeColorCharts();
            }
            
            // Initialize texture charts
            if (chartData.texture) {
                initializeTextureCharts();
            }
            
            // Initialize impression charts
            if (chartData.impression) {
                initializeImpressionCharts();
            }
        }
        
        function initializeColorCharts() {
            // RGB Histogram
            if (chartData.color.histograms) {
                const histData = chartData.color.histograms;
                const traces = [];
                
                ['red', 'green', 'blue'].forEach(color => {
                    if (histData.original[color]) {
                        traces.push({
                            x: Array.from({length: histData.original[color].length}, (_, i) => i),
                            y: histData.original[color],
                            name: `Original ${color.toUpperCase()}`,
                            type: 'bar',
                            marker: {color: color, opacity: 0.7}
                        });
                    }
                });
                
                const layout = {
                    title: 'RGB Histogram Comparison',
                    xaxis: {title: 'Intensity'},
                    yaxis: {title: 'Frequency'},
                    barmode: 'group'
                };
                
                Plotly.newPlot('rgb-histogram', traces, layout);
            }
        }
        
        function initializeTextureCharts() {
            // Edge metrics
            if (chartData.texture.edge_metrics) {
                const edgeData = chartData.texture.edge_metrics;
                const methods = edgeData.map(d => d.method);
                const originalValues = edgeData.map(d => d.original);
                const processedValues = edgeData.map(d => d.processed);
                
                const traces = [
                    {
                        x: methods,
                        y: originalValues,
                        name: 'Original',
                        type: 'bar'
                    },
                    {
                        x: methods,
                        y: processedValues,
                        name: 'Processed',
                        type: 'bar'
                    }
                ];
                
                const layout = {
                    title: 'Edge Detection Comparison',
                    xaxis: {title: 'Method'},
                    yaxis: {title: 'Value'},
                    barmode: 'group'
                };
                
                Plotly.newPlot('edge-metrics', traces, layout);
            }
        }
        
        function initializeImpressionCharts() {
            // Emotion analysis
            if (chartData.impression.emotion_scores) {
                const emotionData = chartData.impression.emotion_scores;
                
                const trace = {
                    values: [emotionData.original, emotionData.processed],
                    labels: ['Original', 'Processed'],
                    type: 'pie',
                    hole: 0.4
                };
                
                const layout = {
                    title: 'Emotion Score Comparison'
                };
                
                Plotly.newPlot('emotion-chart', [trace], layout);
            }
        }
        
        // Comparison slider functionality
        const comparisonSlider = document.getElementById('comparison-range');
        if (comparisonSlider) {
            comparisonSlider.addEventListener('input', function() {
                const value = this.value;
                const originalImg = document.getElementById('original-img');
                const processedImg = document.getElementById('processed-img');
                
                if (originalImg && processedImg) {
                    originalImg.style.opacity = (100 - value) / 100;
                    processedImg.style.opacity = value / 100;
                }
            });
        }
        '''