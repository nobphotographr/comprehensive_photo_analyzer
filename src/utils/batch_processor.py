"""
Batch Processor

バッチ処理モジュール
複数画像ペアの自動解析、進捗管理、エラーハンドリングを行う
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Iterator
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import numpy as np

from utils.logger import get_logger, AnalysisLogger
from core.image_processor import ImageProcessor
from core.color_analyzer import ColorAnalyzer
from visualization.report_generator import ReportGenerator


class BatchProcessor:
    """バッチ処理クラス"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger("batch_processor")
        self.analysis_logger = AnalysisLogger()
        
        # 処理設定
        self.max_workers = config.get("performance", {}).get("max_workers", 4)
        self.multiprocessing = config.get("performance", {}).get("multiprocessing", True)
        
        # 対応する画像形式
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp']
        
    def process_directory(self, input_dir: str, output_dir: str) -> List[Dict[str, Any]]:
        """
        ディレクトリ内の画像ペアを一括処理
        
        Args:
            input_dir: 入力ディレクトリ
            output_dir: 出力ディレクトリ
        
        Returns:
            処理結果のリスト
        """
        try:
            self.logger.info(f"バッチ処理開始: {input_dir} -> {output_dir}")
            
            # 画像ペアの検出
            image_pairs = self._find_image_pairs(input_dir)
            total_pairs = len(image_pairs)
            
            if total_pairs == 0:
                self.logger.warning("処理する画像ペアが見つかりませんでした")
                return []
            
            self.logger.info(f"検出された画像ペア数: {total_pairs}")
            
            # 出力ディレクトリの準備
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 処理実行
            if self.multiprocessing and total_pairs > 1:
                results = self._process_parallel(image_pairs, output_dir)
            else:
                results = self._process_sequential(image_pairs, output_dir)
            
            # バッチサマリーの生成
            self._generate_batch_summary(results, output_dir)
            
            self.logger.info(f"バッチ処理完了: {len(results)} 件処理")
            return results
            
        except Exception as e:
            self.logger.error(f"バッチ処理エラー: {e}")
            raise
    
    def _find_image_pairs(self, input_dir: str) -> List[Tuple[str, str]]:
        """
        画像ペアを検出
        
        Args:
            input_dir: 検索ディレクトリ
        
        Returns:
            (original_path, processed_path) のリスト
        """
        try:
            input_path = Path(input_dir)
            pairs = []
            
            # パターン1: original_*, processed_* のペア
            original_files = list(input_path.glob("original_*"))
            for orig_file in original_files:
                if orig_file.suffix.lower() in self.supported_formats:
                    # 対応する処理済みファイルを検索
                    base_name = orig_file.stem.replace("original_", "")
                    proc_patterns = [
                        f"processed_{base_name}{orig_file.suffix}",
                        f"proc_{base_name}{orig_file.suffix}",
                        f"{base_name}_processed{orig_file.suffix}",
                        f"{base_name}_proc{orig_file.suffix}"
                    ]
                    
                    for pattern in proc_patterns:
                        proc_file = input_path / pattern
                        if proc_file.exists():
                            pairs.append((str(orig_file), str(proc_file)))
                            break
            
            # パターン2: サブディレクトリ構造
            if not pairs:
                pairs.extend(self._find_pairs_in_subdirs(input_path))
            
            # パターン3: ペアリストファイル
            if not pairs:
                pairs.extend(self._find_pairs_from_list(input_path))
            
            self.logger.info(f"検出された画像ペア: {len(pairs)} 件")
            for orig, proc in pairs[:5]:  # 最初の5件をログ出力
                self.logger.debug(f"  {Path(orig).name} <-> {Path(proc).name}")
            
            return pairs
            
        except Exception as e:
            self.logger.error(f"画像ペア検出エラー: {e}")
            return []
    
    def _find_pairs_in_subdirs(self, input_path: Path) -> List[Tuple[str, str]]:
        """サブディレクトリ構造での画像ペア検出"""
        pairs = []
        
        # originals/ と processed/ サブディレクトリを検索
        orig_dir = input_path / "originals"
        proc_dir = input_path / "processed"
        
        if orig_dir.exists() and proc_dir.exists():
            for orig_file in orig_dir.iterdir():
                if orig_file.suffix.lower() in self.supported_formats:
                    proc_file = proc_dir / orig_file.name
                    if proc_file.exists():
                        pairs.append((str(orig_file), str(proc_file)))
        
        return pairs
    
    def _find_pairs_from_list(self, input_path: Path) -> List[Tuple[str, str]]:
        """ペアリストファイルからの画像ペア検出"""
        pairs = []
        
        # pairs.json ファイルを検索
        pairs_file = input_path / "pairs.json"
        if pairs_file.exists():
            try:
                with open(pairs_file, 'r', encoding='utf-8') as f:
                    pairs_data = json.load(f)
                
                for pair in pairs_data.get("pairs", []):
                    orig_path = input_path / pair["original"]
                    proc_path = input_path / pair["processed"]
                    
                    if orig_path.exists() and proc_path.exists():
                        pairs.append((str(orig_path), str(proc_path)))
                        
            except Exception as e:
                self.logger.error(f"ペアリストファイル読み込みエラー: {e}")
        
        return pairs
    
    def _process_sequential(self, image_pairs: List[Tuple[str, str]], output_dir: str) -> List[Dict[str, Any]]:
        """シーケンシャル処理"""
        results = []
        total = len(image_pairs)
        
        for i, (orig_path, proc_path) in enumerate(image_pairs, 1):
            try:
                self.analysis_logger.log_batch_progress(i, total, Path(orig_path).name)
                
                # 単一画像ペアの処理
                result = self._process_single_pair(orig_path, proc_path, output_dir, i)
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"画像ペア処理エラー ({orig_path}): {e}")
                results.append({
                    "original_path": orig_path,
                    "processed_path": proc_path,
                    "status": "error",
                    "error": str(e)
                })
        
        return results
    
    def _process_parallel(self, image_pairs: List[Tuple[str, str]], output_dir: str) -> List[Dict[str, Any]]:
        """並列処理"""
        results = []
        total = len(image_pairs)
        completed = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 全タスクを投入
            future_to_pair = {
                executor.submit(self._process_single_pair, orig, proc, output_dir, i): (orig, proc)
                for i, (orig, proc) in enumerate(image_pairs, 1)
            }
            
            # 完了したタスクから結果を収集
            for future in as_completed(future_to_pair):
                orig_path, proc_path = future_to_pair[future]
                completed += 1
                
                try:
                    result = future.result()
                    results.append(result)
                    self.analysis_logger.log_batch_progress(completed, total, Path(orig_path).name)
                    
                except Exception as e:
                    self.logger.error(f"並列処理エラー ({orig_path}): {e}")
                    results.append({
                        "original_path": orig_path,
                        "processed_path": proc_path,
                        "status": "error",
                        "error": str(e)
                    })
        
        # インデックス順にソート
        results.sort(key=lambda x: x.get("index", 0))
        return results
    
    def _process_single_pair(self, orig_path: str, proc_path: str, output_dir: str, index: int) -> Dict[str, Any]:
        """
        単一画像ペアの処理
        
        Args:
            orig_path: 元画像パス
            proc_path: 処理済み画像パス
            output_dir: 出力ディレクトリ
            index: 処理インデックス
        
        Returns:
            処理結果
        """
        try:
            start_time = time.time()
            
            # 出力ファイル名の生成
            orig_name = Path(orig_path).stem
            output_prefix = f"batch_{index:03d}_{orig_name}"
            
            # 画像処理器の初期化
            processor = ImageProcessor(self.config)
            original_img, processed_img = processor.load_image_pair(orig_path, proc_path)
            
            # 解析実行
            results = {}
            
            # Phase 1-3: 基本色彩解析
            if self.config.get("analysis", {}).get("phase") in ["basic", "1-3", "all"]:
                color_analyzer = ColorAnalyzer(self.config)
                results["color_analysis"] = color_analyzer.analyze(original_img, processed_img)
            
            # レポート生成
            report_generator = ReportGenerator(self.config)
            report_path = report_generator.generate_batch_report(
                results, 
                orig_path, 
                proc_path, 
                output_dir,
                output_prefix
            )
            
            processing_time = time.time() - start_time
            
            return {
                "index": index,
                "original_path": orig_path,
                "processed_path": proc_path,
                "output_prefix": output_prefix,
                "report_path": report_path,
                "processing_time": processing_time,
                "status": "success",
                "summary": results.get("color_analysis", {}).get("summary", {})
            }
            
        except Exception as e:
            self.logger.error(f"単一ペア処理エラー ({orig_path}): {e}")
            raise
    
    def _generate_batch_summary(self, results: List[Dict[str, Any]], output_dir: str) -> Dict[str, Any]:
        """
        バッチ処理サマリーの生成
        
        Args:
            results: 処理結果リスト
            output_dir: 出力ディレクトリ
        
        Returns:
            サマリー情報
        """
        try:
            # 統計情報の集計
            total_processed = len(results)
            successful = len([r for r in results if r.get("status") == "success"])
            failed = total_processed - successful
            
            if successful > 0:
                avg_processing_time = sum(r.get("processing_time", 0) for r in results if r.get("status") == "success") / successful
                total_processing_time = sum(r.get("processing_time", 0) for r in results)
            else:
                avg_processing_time = 0
                total_processing_time = 0
            
            # 解析結果の統計
            analysis_stats = self._aggregate_analysis_results(results)
            
            summary = {
                "batch_info": {
                    "total_pairs": total_processed,
                    "successful": successful,
                    "failed": failed,
                    "success_rate": successful / total_processed if total_processed > 0 else 0,
                    "total_processing_time": total_processing_time,
                    "average_processing_time": avg_processing_time
                },
                "analysis_statistics": analysis_stats,
                "failed_items": [r for r in results if r.get("status") == "error"]
            }
            
            # サマリーファイルの保存
            summary_path = Path(output_dir) / "batch_summary.json"
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"バッチサマリー保存: {summary_path}")
            return summary
            
        except Exception as e:
            self.logger.error(f"バッチサマリー生成エラー: {e}")
            return {"error": str(e)}
    
    def _aggregate_analysis_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """解析結果の統計集計"""
        try:
            successful_results = [r for r in results if r.get("status") == "success" and "summary" in r]
            
            if not successful_results:
                return {"message": "統計可能な結果がありません"}
            
            # 明度・コントラスト変化の統計
            brightness_changes = []
            contrast_changes = []
            quality_scores = []
            
            for result in successful_results:
                summary = result.get("summary", {})
                assessment = summary.get("overall_assessment", {})
                
                if "brightness_change" in assessment:
                    brightness_changes.append(assessment["brightness_change"])
                if "contrast_change" in assessment:
                    contrast_changes.append(assessment["contrast_change"])
                if "quality_score" in assessment:
                    quality_scores.append(assessment["quality_score"])
            
            stats = {}
            
            if brightness_changes:
                stats["brightness_changes"] = {
                    "mean": float(np.mean(brightness_changes)),
                    "std": float(np.std(brightness_changes)),
                    "min": float(np.min(brightness_changes)),
                    "max": float(np.max(brightness_changes))
                }
            
            if contrast_changes:
                stats["contrast_changes"] = {
                    "mean": float(np.mean(contrast_changes)),
                    "std": float(np.std(contrast_changes)),
                    "min": float(np.min(contrast_changes)),
                    "max": float(np.max(contrast_changes))
                }
            
            if quality_scores:
                stats["quality_scores"] = {
                    "mean": float(np.mean(quality_scores)),
                    "std": float(np.std(quality_scores)),
                    "min": float(np.min(quality_scores)),
                    "max": float(np.max(quality_scores))
                }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"統計集計エラー: {e}")
            return {"error": str(e)}