#!/usr/bin/env python3
"""
サンプル画像生成スクリプト

テスト用の元画像と処理済み画像ペアを作成する
"""

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import os
from pathlib import Path


def create_sample_images():
    """テスト用サンプル画像の作成"""
    
    # 出力ディレクトリの作成
    sample_dir = Path("data/sample")
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    print("サンプル画像を作成中...")
    
    # 基本画像サイズ
    width, height = 800, 600
    
    # 1. 基本的なカラーグラデーション画像
    create_gradient_images(sample_dir, width, height)
    
    # 2. 風景風画像
    create_landscape_images(sample_dir, width, height)
    
    # 3. ポートレート風画像
    create_portrait_images(sample_dir, width, height)
    
    print(f"サンプル画像を {sample_dir} に作成しました")


def create_gradient_images(output_dir: Path, width: int, height: int):
    """グラデーション画像ペアの作成"""
    
    # 元画像: 水平グラデーション
    original = np.zeros((height, width, 3), dtype=np.uint8)
    for x in range(width):
        # RGB各チャンネルに異なるグラデーション
        r = int((x / width) * 255)
        g = int(((width - x) / width) * 255)
        b = int((abs(x - width/2) / (width/2)) * 255)
        original[:, x] = [r, g, b]
    
    # 処理済み画像: 色調変更版
    processed = original.copy()
    # 彩度とコントラストを調整
    processed = cv2.convertScaleAbs(processed, alpha=1.2, beta=20)  # コントラスト・明度調整
    hsv = cv2.cvtColor(processed, cv2.COLOR_RGB2HSV)
    hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], 1.3)  # 彩度アップ
    processed = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    # 保存
    cv2.imwrite(str(output_dir / "gradient_original.jpg"), cv2.cvtColor(original, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_dir / "gradient_processed.jpg"), cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))
    
    print("  ✓ グラデーション画像ペア作成完了")


def create_landscape_images(output_dir: Path, width: int, height: int):
    """風景風画像ペアの作成"""
    
    # 元画像: 空と地面のシンプルな風景
    original = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 空部分 (上半分)
    sky_color = [135, 206, 235]  # スカイブルー
    original[:height//2, :] = sky_color
    
    # 地面部分 (下半分)
    ground_color = [34, 139, 34]  # 森の緑
    original[height//2:, :] = ground_color
    
    # 太陽を追加
    center = (width//4, height//4)
    cv2.circle(original, center, 40, (255, 255, 0), -1)
    
    # 雲を追加
    for i in range(5):
        x = int(width * (0.2 + i * 0.15))
        y = int(height * (0.1 + (i % 2) * 0.1))
        cv2.ellipse(original, (x, y), (30, 15), 0, 0, 360, (255, 255, 255), -1)
    
    # 処理済み画像: 暖色調に変更
    processed = original.copy()
    
    # 色温度を暖かく
    processed[:, :, 0] = np.clip(processed[:, :, 0] * 1.1, 0, 255)  # 赤を強化
    processed[:, :, 2] = np.clip(processed[:, :, 2] * 0.9, 0, 255)  # 青を抑制
    
    # コントラストアップ
    processed = cv2.convertScaleAbs(processed, alpha=1.15, beta=10)
    
    # 保存
    cv2.imwrite(str(output_dir / "landscape_original.jpg"), cv2.cvtColor(original, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_dir / "landscape_processed.jpg"), cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))
    
    print("  ✓ 風景画像ペア作成完了")


def create_portrait_images(output_dir: Path, width: int, height: int):
    """ポートレート風画像ペアの作成"""
    
    # 元画像: 中央に人物シルエット風
    original = np.full((height, width, 3), [70, 70, 70], dtype=np.uint8)  # グレー背景
    
    # 簡単な人物形状
    center_x, center_y = width // 2, height // 2
    
    # 顔 (楕円)
    cv2.ellipse(original, (center_x, center_y - 50), (60, 80), 0, 0, 360, (220, 180, 140), -1)
    
    # 体 (長方形)
    cv2.rectangle(original, (center_x - 80, center_y + 30), (center_x + 80, center_y + 200), (100, 150, 200), -1)
    
    # 髪
    cv2.ellipse(original, (center_x, center_y - 80), (70, 50), 0, 0, 360, (101, 67, 33), -1)
    
    # 処理済み画像: ポートレート風処理
    processed = original.copy()
    
    # 肌色を温かく
    skin_mask = cv2.inRange(original, (200, 160, 120), (240, 200, 160))
    processed[skin_mask > 0] = [240, 200, 160]
    
    # 全体の色調整
    processed = cv2.convertScaleAbs(processed, alpha=1.1, beta=15)
    
    # ビネット効果
    rows, cols = height, width
    kernel_x = cv2.getGaussianKernel(cols, cols/3)
    kernel_y = cv2.getGaussianKernel(rows, rows/3)
    kernel = kernel_y * kernel_x.T
    mask = kernel / kernel.max()
    
    for i in range(3):
        processed[:, :, i] = processed[:, :, i] * mask
    
    processed = np.clip(processed, 0, 255).astype(np.uint8)
    
    # 保存
    cv2.imwrite(str(output_dir / "portrait_original.jpg"), cv2.cvtColor(original, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_dir / "portrait_processed.jpg"), cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))
    
    print("  ✓ ポートレート画像ペア作成完了")


def create_test_pairs_config():
    """テスト用のペア設定ファイルを作成"""
    
    pairs_config = {
        "pairs": [
            {
                "name": "gradient_test",
                "original": "gradient_original.jpg",
                "processed": "gradient_processed.jpg",
                "description": "カラーグラデーションテスト"
            },
            {
                "name": "landscape_test", 
                "original": "landscape_original.jpg",
                "processed": "landscape_processed.jpg",
                "description": "風景写真テスト"
            },
            {
                "name": "portrait_test",
                "original": "portrait_original.jpg", 
                "processed": "portrait_processed.jpg",
                "description": "ポートレートテスト"
            }
        ]
    }
    
    import json
    config_path = Path("data/sample/pairs.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(pairs_config, f, indent=2, ensure_ascii=False)
    
    print(f"  ✓ ペア設定ファイル作成: {config_path}")


if __name__ == "__main__":
    create_sample_images()
    create_test_pairs_config()
    print("\n🎨 すべてのサンプル画像の作成が完了しました！")
    print("\nテスト実行例:")
    print("python main.py --mode single --original data/sample/gradient_original.jpg --processed data/sample/gradient_processed.jpg --phase basic")