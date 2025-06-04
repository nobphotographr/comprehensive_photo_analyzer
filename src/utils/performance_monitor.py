"""
Performance Monitor

パフォーマンス監視モジュール
処理時間、メモリ使用量、システムリソースの監視
"""

import time
import psutil
import os
import threading
from typing import Dict, Any, Optional, List
import logging
from dataclasses import dataclass, field
from datetime import datetime

from utils.logger import get_logger


@dataclass
class PerformanceMetrics:
    """パフォーマンス指標データクラス"""
    start_time: float
    end_time: Optional[float] = None
    elapsed_time: Optional[float] = None
    peak_memory_mb: float = 0.0
    avg_memory_mb: float = 0.0
    peak_cpu_percent: float = 0.0
    avg_cpu_percent: float = 0.0
    memory_samples: List[float] = field(default_factory=list)
    cpu_samples: List[float] = field(default_factory=list)


class PerformanceMonitor:
    """パフォーマンス監視クラス"""
    
    def __init__(self, sampling_interval: float = 1.0):
        self.logger = get_logger("performance")
        self.sampling_interval = sampling_interval
        self.is_monitoring = False
        self.monitor_thread = None
        self.process = psutil.Process(os.getpid())
        self.metrics = {}
        
    def start_monitoring(self, operation_name: str = "default") -> None:
        """監視開始"""
        try:
            if self.is_monitoring:
                self.logger.warning("すでに監視中です")
                return
            
            self.is_monitoring = True
            self.metrics[operation_name] = PerformanceMetrics(start_time=time.time())
            
            # 監視スレッド開始
            self.monitor_thread = threading.Thread(
                target=self._monitor_loop, 
                args=(operation_name,),
                daemon=True
            )
            self.monitor_thread.start()
            
            self.logger.info(f"パフォーマンス監視開始: {operation_name}")
            
        except Exception as e:
            self.logger.error(f"監視開始エラー: {e}")
    
    def stop_monitoring(self, operation_name: str = "default") -> PerformanceMetrics:
        """監視停止"""
        try:
            if not self.is_monitoring:
                self.logger.warning("監視が開始されていません")
                return None
            
            self.is_monitoring = False
            
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=2.0)
            
            if operation_name in self.metrics:
                metrics = self.metrics[operation_name]
                metrics.end_time = time.time()
                metrics.elapsed_time = metrics.end_time - metrics.start_time
                
                # 統計計算
                if metrics.memory_samples:
                    metrics.peak_memory_mb = max(metrics.memory_samples)
                    metrics.avg_memory_mb = sum(metrics.memory_samples) / len(metrics.memory_samples)
                
                if metrics.cpu_samples:
                    metrics.peak_cpu_percent = max(metrics.cpu_samples)
                    metrics.avg_cpu_percent = sum(metrics.cpu_samples) / len(metrics.cpu_samples)
                
                self.logger.info(f"パフォーマンス監視終了: {operation_name}")
                return metrics
            
        except Exception as e:
            self.logger.error(f"監視停止エラー: {e}")
            return None
    
    def _monitor_loop(self, operation_name: str) -> None:
        """監視ループ（別スレッドで実行）"""
        try:
            while self.is_monitoring:
                # メモリ使用量取得
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                
                # CPU使用率取得
                cpu_percent = self.process.cpu_percent()
                
                # メトリクスに追加
                if operation_name in self.metrics:
                    self.metrics[operation_name].memory_samples.append(memory_mb)
                    self.metrics[operation_name].cpu_samples.append(cpu_percent)
                
                time.sleep(self.sampling_interval)
                
        except Exception as e:
            self.logger.error(f"監視ループエラー: {e}")
    
    def get_current_memory_usage(self) -> float:
        """現在のメモリ使用量を取得（MB）"""
        try:
            return self.process.memory_info().rss / 1024 / 1024
        except Exception as e:
            self.logger.error(f"メモリ使用量取得エラー: {e}")
            return 0.0
    
    def get_current_cpu_usage(self) -> float:
        """現在のCPU使用率を取得（%）"""
        try:
            return self.process.cpu_percent()
        except Exception as e:
            self.logger.error(f"CPU使用率取得エラー: {e}")
            return 0.0
    
    def print_results(self, operation_name: str = "default") -> None:
        """結果の出力"""
        try:
            if operation_name not in self.metrics:
                self.logger.warning(f"指定された操作の結果が見つかりません: {operation_name}")
                return
            
            metrics = self.metrics[operation_name]
            
            print("\\n" + "="*50)
            print(f"パフォーマンス結果: {operation_name}")
            print("="*50)
            print(f"実行時間: {metrics.elapsed_time:.2f} 秒")
            print(f"ピークメモリ使用量: {metrics.peak_memory_mb:.1f} MB")
            print(f"平均メモリ使用量: {metrics.avg_memory_mb:.1f} MB")
            print(f"ピークCPU使用率: {metrics.peak_cpu_percent:.1f} %")
            print(f"平均CPU使用率: {metrics.avg_cpu_percent:.1f} %")
            print("="*50)
            
        except Exception as e:
            self.logger.error(f"結果出力エラー: {e}")
    
    def export_metrics(self, operation_name: str = "default", output_path: str = None) -> Dict[str, Any]:
        """メトリクスをエクスポート"""
        try:
            if operation_name not in self.metrics:
                return {}
            
            metrics = self.metrics[operation_name]
            
            export_data = {
                "operation_name": operation_name,
                "timestamp": datetime.now().isoformat(),
                "elapsed_time_seconds": metrics.elapsed_time,
                "memory_usage": {
                    "peak_mb": metrics.peak_memory_mb,
                    "average_mb": metrics.avg_memory_mb,
                    "samples": metrics.memory_samples
                },
                "cpu_usage": {
                    "peak_percent": metrics.peak_cpu_percent,
                    "average_percent": metrics.avg_cpu_percent,
                    "samples": metrics.cpu_samples
                },
                "system_info": self._get_system_info()
            }
            
            # ファイル出力
            if output_path:
                import json
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
                self.logger.info(f"メトリクスをエクスポート: {output_path}")
            
            return export_data
            
        except Exception as e:
            self.logger.error(f"メトリクスエクスポートエラー: {e}")
            return {}
    
    def _get_system_info(self) -> Dict[str, Any]:
        """システム情報の取得"""
        try:
            return {
                "cpu_count": psutil.cpu_count(),
                "cpu_count_logical": psutil.cpu_count(logical=True),
                "total_memory_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024,
                "available_memory_gb": psutil.virtual_memory().available / 1024 / 1024 / 1024,
                "disk_usage_gb": psutil.disk_usage('/').free / 1024 / 1024 / 1024,
                "python_process_threads": self.process.num_threads()
            }
        except Exception as e:
            self.logger.error(f"システム情報取得エラー: {e}")
            return {}


class BenchmarkRunner:
    """ベンチマーク実行クラス"""
    
    def __init__(self):
        self.logger = get_logger("benchmark")
        self.results = {}
    
    def run_benchmark(self, func, *args, benchmark_name: str = None, iterations: int = 1, **kwargs) -> Dict[str, Any]:
        """ベンチマーク実行"""
        try:
            if benchmark_name is None:
                benchmark_name = func.__name__
            
            self.logger.info(f"ベンチマーク開始: {benchmark_name} (iterations: {iterations})")
            
            iteration_results = []
            
            for i in range(iterations):
                monitor = PerformanceMonitor()
                monitor.start_monitoring(f"{benchmark_name}_iter_{i}")
                
                try:
                    result = func(*args, **kwargs)
                    success = True
                except Exception as e:
                    self.logger.error(f"ベンチマーク実行エラー (iteration {i}): {e}")
                    result = None
                    success = False
                
                metrics = monitor.stop_monitoring(f"{benchmark_name}_iter_{i}")
                
                iteration_results.append({
                    "iteration": i,
                    "success": success,
                    "result": result,
                    "metrics": metrics
                })
            
            # 統計計算
            successful_iterations = [r for r in iteration_results if r["success"]]
            
            if successful_iterations:
                times = [r["metrics"].elapsed_time for r in successful_iterations]
                memories = [r["metrics"].peak_memory_mb for r in successful_iterations]
                
                benchmark_summary = {
                    "benchmark_name": benchmark_name,
                    "total_iterations": iterations,
                    "successful_iterations": len(successful_iterations),
                    "execution_time": {
                        "min": min(times),
                        "max": max(times),
                        "mean": sum(times) / len(times),
                        "total": sum(times)
                    },
                    "memory_usage": {
                        "min": min(memories),
                        "max": max(memories),
                        "mean": sum(memories) / len(memories)
                    },
                    "iterations": iteration_results
                }
            else:
                benchmark_summary = {
                    "benchmark_name": benchmark_name,
                    "total_iterations": iterations,
                    "successful_iterations": 0,
                    "error": "すべてのイテレーションが失敗しました"
                }
            
            self.results[benchmark_name] = benchmark_summary
            self.logger.info(f"ベンチマーク完了: {benchmark_name}")
            
            return benchmark_summary
            
        except Exception as e:
            self.logger.error(f"ベンチマーク実行エラー: {e}")
            return {"error": str(e)}
    
    def print_benchmark_results(self, benchmark_name: str = None) -> None:
        """ベンチマーク結果の出力"""
        try:
            if benchmark_name:
                results_to_print = {benchmark_name: self.results.get(benchmark_name, {})}
            else:
                results_to_print = self.results
            
            for name, result in results_to_print.items():
                if "error" in result:
                    print(f"\\nベンチマーク: {name} - エラー: {result['error']}")
                    continue
                
                print(f"\\n{'='*60}")
                print(f"ベンチマーク結果: {name}")
                print('='*60)
                print(f"実行回数: {result['successful_iterations']}/{result['total_iterations']}")
                
                if result['successful_iterations'] > 0:
                    exec_time = result['execution_time']
                    memory = result['memory_usage']
                    
                    print(f"実行時間:")
                    print(f"  最短: {exec_time['min']:.3f}秒")
                    print(f"  最長: {exec_time['max']:.3f}秒")
                    print(f"  平均: {exec_time['mean']:.3f}秒")
                    print(f"  合計: {exec_time['total']:.3f}秒")
                    
                    print(f"メモリ使用量:")
                    print(f"  最小: {memory['min']:.1f}MB")
                    print(f"  最大: {memory['max']:.1f}MB")
                    print(f"  平均: {memory['mean']:.1f}MB")
                
                print('='*60)
                
        except Exception as e:
            self.logger.error(f"ベンチマーク結果出力エラー: {e}")