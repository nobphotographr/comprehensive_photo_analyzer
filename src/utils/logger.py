"""
Logger Module

ログ機能モジュール
統一されたログ出力とファイル保存機能を提供
"""

import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logger(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    console_output: bool = True,
    file_output: bool = True
) -> logging.Logger:
    """
    ログシステムの設定
    
    Args:
        log_level: ログレベル (DEBUG, INFO, WARNING, ERROR)
        log_file: ログファイルパス（Noneの場合は自動生成）
        console_output: コンソール出力の有無
        file_output: ファイル出力の有無
    
    Returns:
        設定済みのロガー
    """
    
    # ログレベルの設定
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # ルートロガーの設定
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # 既存のハンドラーをクリア
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # ログフォーマットの設定
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # コンソール出力ハンドラー
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # ファイル出力ハンドラー
    if file_output:
        if log_file is None:
            # ログディレクトリの作成
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            
            # ログファイル名の自動生成
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"photo_analyzer_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # プロジェクト用ロガーを作成
    logger = logging.getLogger("photo_analyzer")
    logger.info("ログシステムが初期化されました")
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    名前付きロガーを取得
    
    Args:
        name: ロガー名
    
    Returns:
        ロガーインスタンス
    """
    return logging.getLogger(f"photo_analyzer.{name}")


class PerformanceLogger:
    """パフォーマンスログ専用クラス"""
    
    def __init__(self, name: str = "performance"):
        self.logger = get_logger(name)
        self.start_times = {}
    
    def start_timer(self, operation: str):
        """タイマー開始"""
        import time
        self.start_times[operation] = time.time()
        self.logger.debug(f"開始: {operation}")
    
    def end_timer(self, operation: str):
        """タイマー終了"""
        import time
        if operation in self.start_times:
            elapsed = time.time() - self.start_times[operation]
            self.logger.info(f"完了: {operation} ({elapsed:.2f}秒)")
            del self.start_times[operation]
            return elapsed
        else:
            self.logger.warning(f"タイマーが開始されていません: {operation}")
            return None
    
    def log_memory_usage(self, operation: str = ""):
        """メモリ使用量をログ出力"""
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            if operation:
                self.logger.info(f"メモリ使用量 ({operation}): {memory_mb:.1f}MB")
            else:
                self.logger.info(f"メモリ使用量: {memory_mb:.1f}MB")
                
        except ImportError:
            self.logger.warning("psutilが利用できません - メモリ監視は無効")
        except Exception as e:
            self.logger.error(f"メモリ使用量取得エラー: {e}")


class AnalysisLogger:
    """解析専用ログクラス"""
    
    def __init__(self):
        self.logger = get_logger("analysis")
        self.perf_logger = PerformanceLogger("analysis_performance")
    
    def log_analysis_start(self, analyzer_name: str, image_info: dict):
        """解析開始ログ"""
        self.logger.info(f"解析開始: {analyzer_name}")
        self.logger.debug(f"画像情報: {image_info}")
        self.perf_logger.start_timer(analyzer_name)
    
    def log_analysis_end(self, analyzer_name: str, results_summary: dict):
        """解析終了ログ"""
        elapsed = self.perf_logger.end_timer(analyzer_name)
        self.logger.info(f"解析完了: {analyzer_name}")
        self.logger.debug(f"結果サマリー: {results_summary}")
        
        if elapsed:
            self.logger.info(f"処理時間: {elapsed:.2f}秒")
    
    def log_analysis_error(self, analyzer_name: str, error: Exception):
        """解析エラーログ"""
        self.logger.error(f"解析エラー ({analyzer_name}): {error}")
    
    def log_batch_progress(self, current: int, total: int, current_file: str = ""):
        """バッチ処理進捗ログ"""
        progress = (current / total) * 100
        if current_file:
            self.logger.info(f"バッチ進捗: {current}/{total} ({progress:.1f}%) - {current_file}")
        else:
            self.logger.info(f"バッチ進捗: {current}/{total} ({progress:.1f}%)")