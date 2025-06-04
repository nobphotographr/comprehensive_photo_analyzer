#!/usr/bin/env python3
"""
テクスチャ解析のみのテストスクリプト
"""

import sys
import os
from pathlib import Path
import numpy as np
import cv2

# プロジェクトのsrcディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent / "src"))

from analyzers.texture_analyzer import TextureAnalyzer
from utils.config_manager import ConfigManager

def main():
    """テクスチャ解析テスト"""
    
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
    
    # テクスチャ解析器初期化
    texture_analyzer = TextureAnalyzer(config)
    
    # テクスチャ解析実行
    print("テクスチャ解析開始...")
    results = texture_analyzer.analyze_texture(original, processed)
    
    if results:
        print("テクスチャ解析完了!")
        print(f"解析結果の項目数: {len(results)}")
        
        # 総合評価結果を表示
        if "overall_assessment" in results:
            overall = results["overall_assessment"]
            print(f"総合スコア: {overall.get('overall_score', 'N/A')}")
            print(f"品質評価: {overall.get('quality_assessment', 'N/A')}")
    else:
        print("テクスチャ解析が失敗しました")

if __name__ == "__main__":
    main()