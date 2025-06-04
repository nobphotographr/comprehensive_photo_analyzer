#!/usr/bin/env python3
"""
Comprehensive Photo Analysis Tool - Main Entry Point

写真・画像総合解析ツールのメインエントリポイント
段階的開発（Phase 1-9）に対応した統合システム
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# プロジェクトのsrcディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.config_manager import ConfigManager
from utils.logger import setup_logger
from utils.batch_processor import BatchProcessor
from core.image_processor import ImageProcessor
from core.color_analyzer import ColorAnalyzer
from visualization.report_generator import ReportGenerator


def setup_argument_parser() -> argparse.ArgumentParser:
    """コマンドライン引数の設定"""
    parser = argparse.ArgumentParser(
        description="Comprehensive Photo Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 基本解析（Phase 1-3）
  python main.py --mode single --original original.jpg --processed processed.jpg --phase basic
  
  # 総合解析（全Phase）
  python main.py --mode comprehensive --original original.jpg --processed processed.jpg --phase all
  
  # バッチ処理
  python main.py --mode batch --input-dir data/input/ --output-dir data/output/
  
  # 設定ファイル指定
  python main.py --config config/custom_analysis.yaml
        """
    )
    
    # 動作モード
    parser.add_argument(
        "--mode", 
        choices=["single", "comprehensive", "batch"],
        default="single",
        help="動作モード: single(単一ペア), comprehensive(総合解析), batch(バッチ処理)"
    )
    
    # 画像ファイル指定（single/comprehensive用）
    parser.add_argument(
        "--original",
        type=str,
        help="元画像ファイルパス"
    )
    
    parser.add_argument(
        "--processed", 
        type=str,
        help="処理済み画像ファイルパス"
    )
    
    # バッチ処理用
    parser.add_argument(
        "--input-dir",
        type=str,
        help="入力ディレクトリ（バッチ処理用）"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        help="出力ディレクトリ"
    )
    
    # Phase指定
    parser.add_argument(
        "--phase",
        choices=["basic", "1-3", "intermediate", "4-6", "advanced", "7-9", "all"],
        default="basic",
        help="解析フェーズ: basic(1-3), intermediate(4-6), advanced(7-9), all(全て)"
    )
    
    # 設定ファイル
    parser.add_argument(
        "--config",
        type=str,
        help="設定ファイルパス（YAML形式）"
    )
    
    # 解析器指定
    parser.add_argument(
        "--analyzers",
        type=str,
        help="使用する解析器をカンマ区切りで指定（color,texture,impression等）"
    )
    
    # 出力形式
    parser.add_argument(
        "--output-format",
        choices=["html", "json", "csv", "all"],
        default="html",
        help="出力形式: html, json, csv, all"
    )
    
    # ログレベル
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="ログレベル"
    )
    
    # ベンチマークモード
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="ベンチマークモードで実行（性能測定）"
    )
    
    return parser


def validate_arguments(args: argparse.Namespace) -> bool:
    """引数の妥当性チェック"""
    
    if args.mode in ["single", "comprehensive"]:
        if not args.original or not args.processed:
            print("エラー: single/comprehensiveモードでは --original と --processed が必要です")
            return False
        
        if not Path(args.original).exists():
            print(f"エラー: 元画像ファイルが見つかりません: {args.original}")
            return False
        
        if not Path(args.processed).exists():
            print(f"エラー: 処理済み画像ファイルが見つかりません: {args.processed}")
            return False
    
    elif args.mode == "batch":
        if not args.input_dir:
            print("エラー: batchモードでは --input-dir が必要です")
            return False
        
        if not Path(args.input_dir).exists():
            print(f"エラー: 入力ディレクトリが見つかりません: {args.input_dir}")
            return False
    
    return True


def load_configuration(args: argparse.Namespace) -> Dict[str, Any]:
    """設定の読み込み"""
    config_manager = ConfigManager()
    
    if args.config:
        config = config_manager.load_config(args.config)
    else:
        # デフォルト設定を使用
        config = config_manager.get_default_config()
    
    # コマンドライン引数で設定を上書き
    if args.phase:
        config["analysis"]["phase"] = args.phase
    
    if args.analyzers:
        config["analysis"]["analyzers"] = [a.strip() for a in args.analyzers.split(",")]
    
    if args.output_format:
        config["output"]["format"] = args.output_format
    
    if args.output_dir:
        config["output"]["directory"] = args.output_dir
    
    return config


def run_single_analysis(args: argparse.Namespace, config: Dict[str, Any]) -> bool:
    """単一画像ペア解析の実行"""
    try:
        logger = logging.getLogger(__name__)
        logger.info(f"単一解析開始: {args.original} vs {args.processed}")
        
        # 画像処理器の初期化
        processor = ImageProcessor(config)
        original_img, processed_img = processor.load_image_pair(
            args.original, args.processed
        )
        
        # 解析実行
        results = {}
        
        # Phase 1-3: 基本色彩解析
        if config["analysis"]["phase"] in ["basic", "1-3", "all"]:
            color_analyzer = ColorAnalyzer(config)
            results["color_analysis"] = color_analyzer.analyze(original_img, processed_img)
        
        # レポート生成
        report_generator = ReportGenerator(config)
        report_path = report_generator.generate_report(
            results, 
            args.original, 
            args.processed
        )
        
        logger.info(f"解析完了: レポートを {report_path} に出力しました")
        return True
        
    except Exception as e:
        logger.error(f"解析中にエラーが発生しました: {e}")
        return False


def run_batch_analysis(args: argparse.Namespace, config: Dict[str, Any]) -> bool:
    """バッチ解析の実行"""
    try:
        logger = logging.getLogger(__name__)
        logger.info(f"バッチ解析開始: {args.input_dir}")
        
        batch_processor = BatchProcessor(config)
        results = batch_processor.process_directory(
            args.input_dir,
            args.output_dir or "data/output/batch_results"
        )
        
        logger.info(f"バッチ解析完了: {len(results)} 件のペアを処理しました")
        return True
        
    except Exception as e:
        logger.error(f"バッチ解析中にエラーが発生しました: {e}")
        return False


def main():
    """メイン関数"""
    
    # 引数解析
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # 引数検証
    if not validate_arguments(args):
        parser.print_help()
        sys.exit(1)
    
    # ログ設定
    logger = setup_logger(args.log_level)
    logger.info("Comprehensive Photo Analysis Tool 開始")
    
    try:
        # 設定読み込み
        config = load_configuration(args)
        
        # ベンチマークモード
        if args.benchmark:
            from utils.performance_monitor import PerformanceMonitor
            monitor = PerformanceMonitor()
            monitor.start_monitoring()
        
        # 実行モードに応じた処理
        success = False
        
        if args.mode in ["single", "comprehensive"]:
            success = run_single_analysis(args, config)
        
        elif args.mode == "batch":
            success = run_batch_analysis(args, config)
        
        # ベンチマーク結果出力
        if args.benchmark:
            monitor.stop_monitoring()
            monitor.print_results()
        
        # 終了処理
        if success:
            logger.info("処理が正常に完了しました")
            sys.exit(0)
        else:
            logger.error("処理中にエラーが発生しました")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("ユーザーによる処理中断")
        sys.exit(1)
    except Exception as e:
        logger.error(f"予期しないエラーが発生しました: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()