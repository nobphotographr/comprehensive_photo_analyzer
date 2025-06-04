"""
高度バッチ処理モジュール

並列処理、進捗監視、統計分析機能を備えた高度なバッチ処理システム
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import logging
from datetime import datetime
import time

from .image_utils import PerformanceMonitor
from .config_manager import ConfigManager

class AdvancedBatchProcessor:
    """高度バッチ処理クラス"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初期化
        
        Args:
            config: 設定辞書
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.performance_monitor = PerformanceMonitor()
        
        # バッチ処理設定
        batch_config = config.get('batch', {})
        self.max_workers = batch_config.get('max_workers', 4)
        self.chunk_size = batch_config.get('chunk_size', 10)
        self.enable_progress = batch_config.get('enable_progress', True)
        self.enable_statistics = batch_config.get('enable_statistics', True)
        self.output_individual_reports = batch_config.get('output_individual_reports', True)
        
        # 結果格納
        self.batch_results = []
        self.batch_statistics = {}
        self.failed_items = []
    
    def process_directory(self, input_dir: str, output_dir: str = None, 
                         file_patterns: List[str] = None) -> Dict[str, Any]:
        """
        ディレクトリのバッチ処理
        
        Args:
            input_dir: 入力ディレクトリ
            output_dir: 出力ディレクトリ
            file_patterns: ファイルパターンリスト
            
        Returns:
            バッチ処理結果
        """
        try:
            self.logger.info(f"高度バッチ処理開始: {input_dir}")
            self.performance_monitor.start_timer("batch_processing")
            
            # 出力ディレクトリ準備
            if output_dir is None:
                output_dir = str(Path(input_dir).parent / "batch_results")
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 画像ペア検出
            image_pairs = self._discover_image_pairs(input_dir, file_patterns)
            
            if not image_pairs:
                self.logger.warning("処理対象の画像ペアが見つかりません")
                return {"status": "no_pairs_found", "results": []}
            
            self.logger.info(f"発見された画像ペア数: {len(image_pairs)}")
            
            # 並列処理実行
            results = self._process_pairs_parallel(image_pairs, output_path)
            
            # 統計分析
            if self.enable_statistics:
                statistics = self._calculate_batch_statistics(results)
                self._save_batch_statistics(statistics, output_path)
            
            # サマリーレポート生成
            summary = self._generate_batch_summary(results, output_path)
            
            total_time = self.performance_monitor.end_timer("batch_processing")
            
            return {
                "status": "completed",
                "total_pairs": len(image_pairs),
                "successful": len([r for r in results if r.get("status") == "success"]),
                "failed": len([r for r in results if r.get("status") == "failed"]),
                "processing_time": total_time,
                "output_directory": str(output_path),
                "summary_report": summary,
                "statistics": statistics if self.enable_statistics else None
            }
            
        except Exception as e:
            self.logger.error(f"バッチ処理エラー: {e}")
            return {"status": "error", "error": str(e)}
    
    def _discover_image_pairs(self, input_dir: str, file_patterns: List[str] = None) -> List[Dict[str, str]]:
        """画像ペアを自動発見"""
        if file_patterns is None:
            file_patterns = ["*.jpg", "*.jpeg", "*.png", "*.tiff", "*.bmp"]
        
        input_path = Path(input_dir)
        pairs = []
        
        # パターン1: original/processed フォルダ構造
        original_dir = input_path / "original"
        processed_dir = input_path / "processed"
        
        if original_dir.exists() and processed_dir.exists():
            pairs.extend(self._find_pairs_in_folders(original_dir, processed_dir, file_patterns))
        
        # パターン2: _original, _processed サフィックス
        pairs.extend(self._find_pairs_by_suffix(input_path, file_patterns))
        
        # パターン3: pairs.json設定ファイル
        pairs_config = input_path / "pairs.json"
        if pairs_config.exists():
            pairs.extend(self._load_pairs_from_config(pairs_config, input_path))
        
        return pairs
    
    def _find_pairs_in_folders(self, original_dir: Path, processed_dir: Path, 
                              patterns: List[str]) -> List[Dict[str, str]]:
        """フォルダ構造から画像ペアを検索"""
        pairs = []
        
        for pattern in patterns:
            for original_file in original_dir.glob(pattern):
                processed_file = processed_dir / original_file.name
                if processed_file.exists():
                    pairs.append({
                        "original": str(original_file),
                        "processed": str(processed_file),
                        "name": original_file.stem
                    })
        
        return pairs
    
    def _find_pairs_by_suffix(self, base_dir: Path, patterns: List[str]) -> List[Dict[str, str]]:
        """サフィックスによる画像ペア検索"""
        pairs = []
        
        for pattern in patterns:
            for original_file in base_dir.glob(f"*_original{pattern[1:]}"):
                processed_name = original_file.name.replace("_original", "_processed")
                processed_file = base_dir / processed_name
                
                if processed_file.exists():
                    pairs.append({
                        "original": str(original_file),
                        "processed": str(processed_file),
                        "name": original_file.stem.replace("_original", "")
                    })
        
        return pairs
    
    def _load_pairs_from_config(self, config_path: Path, base_dir: Path) -> List[Dict[str, str]]:
        """設定ファイルから画像ペアを読み込み"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            pairs = []
            for pair_config in config.get("pairs", []):
                original_path = base_dir / pair_config["original"]
                processed_path = base_dir / pair_config["processed"]
                
                if original_path.exists() and processed_path.exists():
                    pairs.append({
                        "original": str(original_path),
                        "processed": str(processed_path),
                        "name": pair_config.get("name", original_path.stem)
                    })
            
            return pairs
            
        except Exception as e:
            self.logger.error(f"ペア設定読み込みエラー: {e}")
            return []
    
    def _process_pairs_parallel(self, image_pairs: List[Dict[str, str]], 
                               output_path: Path) -> List[Dict[str, Any]]:
        """並列処理で画像ペアを処理"""
        results = []
        
        # 進捗監視
        if self.enable_progress:
            self._setup_progress_monitoring(len(image_pairs))
        
        # チャンク分割
        chunks = [image_pairs[i:i + self.chunk_size] 
                 for i in range(0, len(image_pairs), self.chunk_size)]
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # 各チャンクを並列処理
            future_to_chunk = {
                executor.submit(self._process_chunk, chunk, output_path, chunk_idx): chunk_idx
                for chunk_idx, chunk in enumerate(chunks)
            }
            
            for future in as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                try:
                    chunk_results = future.result()
                    results.extend(chunk_results)
                    
                    if self.enable_progress:
                        self._update_progress(len(chunk_results))
                        
                except Exception as e:
                    self.logger.error(f"チャンク {chunk_idx} 処理エラー: {e}")
                    # エラーのあったチャンクの項目を失敗として記録
                    for pair in chunks[chunk_idx]:
                        results.append({
                            "pair": pair,
                            "status": "failed",
                            "error": str(e)
                        })
        
        return results
    
    def _process_chunk(self, chunk: List[Dict[str, str]], output_path: Path, 
                      chunk_idx: int) -> List[Dict[str, Any]]:
        """チャンク単位での処理"""
        results = []
        
        for pair in chunk:
            try:
                # 個別の画像ペア処理
                result = self._process_single_pair(pair, output_path)
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"画像ペア処理エラー {pair['name']}: {e}")
                results.append({
                    "pair": pair,
                    "status": "failed",
                    "error": str(e)
                })
        
        return results
    
    def _process_single_pair(self, pair: Dict[str, str], output_path: Path) -> Dict[str, Any]:
        """単一画像ペアの処理"""
        # この部分は実際の解析処理を呼び出す
        # メインの解析システムとの統合が必要
        
        start_time = time.time()
        
        try:
            # 仮の処理（実際にはメイン解析システムを呼び出し）
            pair_result = {
                "pair": pair,
                "status": "success",
                "processing_time": time.time() - start_time,
                "output_files": [
                    str(output_path / f"{pair['name']}.html"),
                    str(output_path / f"{pair['name']}.json")
                ]
            }
            
            # 個別レポート出力（オプション）
            if self.output_individual_reports:
                self._save_individual_report(pair_result, output_path)
            
            return pair_result
            
        except Exception as e:
            return {
                "pair": pair,
                "status": "failed",
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def _calculate_batch_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """バッチ統計の計算"""
        successful_results = [r for r in results if r.get("status") == "success"]
        failed_results = [r for r in results if r.get("status") == "failed"]
        
        processing_times = [r.get("processing_time", 0) for r in successful_results]
        
        statistics = {
            "summary": {
                "total_pairs": len(results),
                "successful": len(successful_results),
                "failed": len(failed_results),
                "success_rate": len(successful_results) / len(results) if results else 0
            },
            "performance": {
                "total_processing_time": sum(processing_times),
                "average_processing_time": np.mean(processing_times) if processing_times else 0,
                "median_processing_time": np.median(processing_times) if processing_times else 0,
                "min_processing_time": min(processing_times) if processing_times else 0,
                "max_processing_time": max(processing_times) if processing_times else 0
            },
            "errors": {
                "error_count": len(failed_results),
                "error_types": self._categorize_errors(failed_results)
            }
        }
        
        return statistics
    
    def _categorize_errors(self, failed_results: List[Dict[str, Any]]) -> Dict[str, int]:
        """エラーの分類"""
        error_categories = {}
        
        for result in failed_results:
            error_msg = result.get("error", "Unknown error")
            # エラーメッセージから簡単な分類
            if "FileNotFoundError" in error_msg:
                category = "file_not_found"
            elif "MemoryError" in error_msg:
                category = "memory_error"
            elif "OpenCV" in error_msg or "cv2" in error_msg:
                category = "opencv_error"
            else:
                category = "other_error"
            
            error_categories[category] = error_categories.get(category, 0) + 1
        
        return error_categories
    
    def _save_batch_statistics(self, statistics: Dict[str, Any], output_path: Path):
        """バッチ統計の保存"""
        stats_file = output_path / "batch_statistics.json"
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(statistics, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"バッチ統計を保存: {stats_file}")
    
    def _generate_batch_summary(self, results: List[Dict[str, Any]], output_path: Path) -> str:
        """バッチサマリーレポート生成"""
        summary_file = output_path / "batch_summary.html"
        
        successful_count = len([r for r in results if r.get("status") == "success"])
        failed_count = len([r for r in results if r.get("status") == "failed"])
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Batch Processing Summary</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .summary {{ background: #f5f5f5; padding: 20px; border-radius: 8px; }}
                .success {{ color: green; }}
                .failed {{ color: red; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Batch Processing Summary</h1>
            <div class="summary">
                <h2>Overall Results</h2>
                <p><strong>Total Pairs:</strong> {len(results)}</p>
                <p class="success"><strong>Successful:</strong> {successful_count}</p>
                <p class="failed"><strong>Failed:</strong> {failed_count}</p>
                <p><strong>Success Rate:</strong> {successful_count/len(results)*100:.1f}%</p>
                <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <h2>Processing Details</h2>
            <table>
                <tr>
                    <th>Pair Name</th>
                    <th>Status</th>
                    <th>Processing Time</th>
                    <th>Notes</th>
                </tr>
        """
        
        for result in results:
            pair = result.get("pair", {})
            name = pair.get("name", "Unknown")
            status = result.get("status", "Unknown")
            proc_time = result.get("processing_time", 0)
            error = result.get("error", "")
            
            status_class = "success" if status == "success" else "failed"
            
            html_content += f"""
                <tr>
                    <td>{name}</td>
                    <td class="{status_class}">{status}</td>
                    <td>{proc_time:.2f}s</td>
                    <td>{error if error else '-'}</td>
                </tr>
            """
        
        html_content += """
            </table>
        </body>
        </html>
        """
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"バッチサマリーを保存: {summary_file}")
        return str(summary_file)
    
    def _save_individual_report(self, result: Dict[str, Any], output_path: Path):
        """個別レポートの保存"""
        pair = result["pair"]
        report_file = output_path / f"{pair['name']}_report.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    
    def _setup_progress_monitoring(self, total_items: int):
        """進捗監視の設定"""
        self.progress_total = total_items
        self.progress_completed = 0
        self.progress_start_time = time.time()
        
        self.logger.info(f"進捗監視開始: 総項目数 {total_items}")
    
    def _update_progress(self, completed_items: int):
        """進捗更新"""
        self.progress_completed += completed_items
        
        if self.progress_total > 0:
            progress_percentage = (self.progress_completed / self.progress_total) * 100
            elapsed_time = time.time() - self.progress_start_time
            
            if self.progress_completed > 0:
                estimated_total_time = elapsed_time * (self.progress_total / self.progress_completed)
                remaining_time = estimated_total_time - elapsed_time
                
                self.logger.info(
                    f"進捗: {self.progress_completed}/{self.progress_total} "
                    f"({progress_percentage:.1f}%) - "
                    f"残り時間: {remaining_time/60:.1f}分"
                )