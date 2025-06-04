#!/usr/bin/env python3
"""
印象・感情解析のみのテストスクリプト
"""

import sys
import os
from pathlib import Path
import numpy as np
import cv2

# プロジェクトのsrcディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent / "src"))

from analyzers.impression_analyzer import ImpressionAnalyzer
from utils.config_manager import ConfigManager

def main():
    """印象・感情解析テスト"""
    
    # 設定読み込み
    config_manager = ConfigManager()
    config = config_manager.load_config("config/default_config.yaml")
    
    # テスト画像読み込み
    original = cv2.imread("data/sample/gradient_original.jpg")
    processed = cv2.imread("data/sample/gradient_processed.jpg")
    
    if original is None or processed is None:
        print("エラー: テスト画像が見つかりません")
        return
    
    # RGB変換とfloat型に正規化
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    
    print(f"画像サイズ: {original.shape}")
    print(f"画像データ型: {original.dtype}")
    print(f"画像値範囲: {original.min():.3f} - {original.max():.3f}")
    
    # 印象解析器初期化
    impression_analyzer = ImpressionAnalyzer(config)
    
    # 印象解析実行
    print("印象・感情解析開始...")
    results = impression_analyzer.analyze_impression(original, processed)
    
    if results:
        print("印象・感情解析完了!")
        print(f"解析結果の項目数: {len(results)}")
        
        # 色彩心理学結果
        if "color_psychology" in results:
            psychology = results["color_psychology"]
            if "changes" in psychology:
                emotion_change = psychology["changes"].get("emotion_change", 0)
                print(f"感情変化: {emotion_change:+.3f}")
        
        # 総合印象結果を表示
        if "overall_impression" in results:
            overall = results["overall_impression"]
            print(f"総合印象スコア: {overall.get('overall_impression_score', 'N/A')}")
            print(f"印象評価: {overall.get('impression_assessment', 'N/A')}")
    else:
        print("印象・感情解析が失敗しました")

if __name__ == "__main__":
    main()