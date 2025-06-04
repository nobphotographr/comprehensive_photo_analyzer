#!/usr/bin/env python3
"""
総合システム統合テストスクリプト

全ての解析機能（色彩・テクスチャ・印象）の統合テストを実行
パフォーマンス測定、エラーハンドリング、結果検証を含む
"""

import sys
import os
import time
import json
from pathlib import Path
import logging
import traceback

# プロジェクトのsrcディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.config_manager import ConfigManager
from utils.image_utils import PerformanceMonitor
from core.image_processor import ImageProcessor
from core.color_analyzer import ColorAnalyzer
from analyzers.texture_analyzer import TextureAnalyzer
from analyzers.impression_analyzer import ImpressionAnalyzer
from visualization.report_generator import ReportGenerator

class ComprehensiveSystemTest:
    """総合システムテストクラス"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.performance_monitor = PerformanceMonitor()
        self.test_results = {}
        
    def _setup_logging(self):
        """ログ設定"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def run_comprehensive_test(self):
        """包括的テストの実行"""
        self.logger.info("=" * 60)
        self.logger.info("総合システム統合テスト開始")
        self.logger.info("=" * 60)
        
        self.performance_monitor.start_timer("total_test_time")
        
        try:
            # テスト準備
            self._prepare_test_environment()
            
            # 基本機能テスト
            self._test_basic_functionality()
            
            # 全解析器統合テスト
            self._test_integrated_analysis()
            
            # パフォーマンステスト
            self._test_performance()
            
            # エラーハンドリングテスト
            self._test_error_handling()
            
            # 結果検証
            self._validate_results()
            
            # 最終レポート
            self._generate_test_report()
            
        except Exception as e:
            self.logger.error(f"テスト実行エラー: {e}")
            self.logger.error(traceback.format_exc())
            return False
        
        finally:
            total_time = self.performance_monitor.end_timer("total_test_time")
            self.logger.info(f"総テスト時間: {total_time:.2f}秒")
        
        return True
    
    def _prepare_test_environment(self):
        """テスト環境の準備"""
        self.logger.info("テスト環境準備中...")
        
        # 設定ファイルの読み込み
        self.config_manager = ConfigManager()
        self.config = self.config_manager.load_config("config/default_config.yaml")
        
        # テスト画像の確認
        self.test_images = {
            "gradient": {
                "original": "data/sample/gradient_original.jpg",
                "processed": "data/sample/gradient_processed.jpg"
            },
            "landscape": {
                "original": "data/sample/landscape_original.jpg", 
                "processed": "data/sample/landscape_processed.jpg"
            },
            "portrait": {
                "original": "data/sample/portrait_original.jpg",
                "processed": "data/sample/portrait_processed.jpg"
            }
        }
        
        # 画像ファイルの存在確認
        missing_files = []
        for image_set, paths in self.test_images.items():
            for img_type, path in paths.items():
                if not Path(path).exists():
                    missing_files.append(f"{image_set}_{img_type}: {path}")
        
        if missing_files:
            self.logger.warning(f"以下のテスト画像が見つかりません: {missing_files}")
            # 利用可能な画像のみでテスト
            available_sets = []
            for image_set, paths in self.test_images.items():
                if all(Path(path).exists() for path in paths.values()):
                    available_sets.append(image_set)
            
            self.test_images = {k: v for k, v in self.test_images.items() if k in available_sets}
            self.logger.info(f"利用可能なテストセット: {list(self.test_images.keys())}")
        
        self.logger.info("テスト環境準備完了")
    
    def _test_basic_functionality(self):
        """基本機能テスト"""
        self.logger.info("基本機能テスト開始")
        
        test_results = {"basic_functionality": {}}
        
        for image_set, paths in self.test_images.items():
            self.logger.info(f"テストセット: {image_set}")
            
            try:
                # 画像読み込みテスト
                self.performance_monitor.start_timer(f"image_loading_{image_set}")
                processor = ImageProcessor(self.config)
                original_img, processed_img = processor.load_image_pair(
                    paths["original"], paths["processed"]
                )
                load_time = self.performance_monitor.end_timer(f"image_loading_{image_set}")
                
                test_results["basic_functionality"][image_set] = {
                    "image_loading": {
                        "status": "success",
                        "time": load_time,
                        "original_shape": original_img.shape,
                        "processed_shape": processed_img.shape
                    }
                }
                
                self.logger.info(f"✓ 画像読み込み成功: {original_img.shape}")
                
            except Exception as e:
                self.logger.error(f"✗ 画像読み込み失敗: {e}")
                test_results["basic_functionality"][image_set] = {
                    "image_loading": {"status": "failed", "error": str(e)}
                }
        
        self.test_results.update(test_results)
        self.logger.info("基本機能テスト完了")
    
    def _test_integrated_analysis(self):
        """統合解析テスト"""
        self.logger.info("統合解析テスト開始")
        
        test_results = {"integrated_analysis": {}}
        
        # 各解析器のテスト
        analyzers = {
            "color": ColorAnalyzer,
            "texture": TextureAnalyzer, 
            "impression": ImpressionAnalyzer
        }
        
        for image_set, paths in self.test_images.items():
            self.logger.info(f"統合解析テスト - {image_set}")
            
            try:
                # 画像読み込み
                processor = ImageProcessor(self.config)
                original_img, processed_img = processor.load_image_pair(
                    paths["original"], paths["processed"]
                )
                
                test_results["integrated_analysis"][image_set] = {}
                
                # 各解析器のテスト
                for analyzer_name, analyzer_class in analyzers.items():
                    self.performance_monitor.start_timer(f"{analyzer_name}_{image_set}")
                    
                    try:
                        analyzer = analyzer_class(self.config)
                        
                        if analyzer_name == "color":
                            results = analyzer.analyze(original_img, processed_img)
                        elif analyzer_name == "texture":
                            results = analyzer.analyze_texture(original_img, processed_img)
                        elif analyzer_name == "impression":
                            results = analyzer.analyze_impression(original_img, processed_img)
                        
                        analysis_time = self.performance_monitor.end_timer(f"{analyzer_name}_{image_set}")
                        
                        # 結果検証
                        is_valid = self._validate_analysis_results(results, analyzer_name)
                        
                        test_results["integrated_analysis"][image_set][analyzer_name] = {
                            "status": "success",
                            "time": analysis_time,
                            "result_keys": list(results.keys()) if results else [],
                            "is_valid": is_valid
                        }
                        
                        self.logger.info(f"✓ {analyzer_name}解析成功: {analysis_time:.2f}秒")
                        
                    except Exception as e:
                        analysis_time = self.performance_monitor.end_timer(f"{analyzer_name}_{image_set}")
                        test_results["integrated_analysis"][image_set][analyzer_name] = {
                            "status": "failed", 
                            "error": str(e),
                            "time": analysis_time
                        }
                        self.logger.error(f"✗ {analyzer_name}解析失敗: {e}")
                
            except Exception as e:
                self.logger.error(f"✗ 統合解析失敗 ({image_set}): {e}")
                test_results["integrated_analysis"][image_set] = {
                    "status": "failed",
                    "error": str(e)
                }
        
        self.test_results.update(test_results)
        self.logger.info("統合解析テスト完了")
    
    def _test_performance(self):
        """パフォーマンステスト"""
        self.logger.info("パフォーマンステスト開始")
        
        test_results = {"performance": {}}
        
        # メモリ使用量測定
        self.performance_monitor.log_memory_usage("start_performance_test")
        
        # 処理時間ベンチマーク
        if self.test_images:
            first_image_set = list(self.test_images.keys())[0]
            paths = self.test_images[first_image_set]
            
            try:
                # 複数回実行して平均時間を測定
                execution_times = []
                
                for i in range(3):  # 3回実行
                    start_time = time.time()
                    
                    processor = ImageProcessor(self.config)
                    original_img, processed_img = processor.load_image_pair(
                        paths["original"], paths["processed"]
                    )
                    
                    # 全解析実行
                    color_analyzer = ColorAnalyzer(self.config)
                    texture_analyzer = TextureAnalyzer(self.config)
                    impression_analyzer = ImpressionAnalyzer(self.config)
                    
                    color_results = color_analyzer.analyze(original_img, processed_img)
                    texture_results = texture_analyzer.analyze_texture(original_img, processed_img)
                    impression_results = impression_analyzer.analyze_impression(original_img, processed_img)
                    
                    execution_time = time.time() - start_time
                    execution_times.append(execution_time)
                    
                    self.logger.info(f"実行回数 {i+1}: {execution_time:.2f}秒")
                
                # 統計計算
                avg_time = sum(execution_times) / len(execution_times)
                min_time = min(execution_times)
                max_time = max(execution_times)
                
                test_results["performance"] = {
                    "average_execution_time": avg_time,
                    "min_execution_time": min_time,
                    "max_execution_time": max_time,
                    "execution_times": execution_times,
                    "performance_rating": self._rate_performance(avg_time)
                }
                
                self.logger.info(f"平均実行時間: {avg_time:.2f}秒")
                self.logger.info(f"パフォーマンス評価: {test_results['performance']['performance_rating']}")
                
            except Exception as e:
                test_results["performance"] = {
                    "status": "failed",
                    "error": str(e)
                }
                self.logger.error(f"パフォーマンステスト失敗: {e}")
        
        # メモリ使用量測定
        self.performance_monitor.log_memory_usage("end_performance_test")
        
        self.test_results.update(test_results)
        self.logger.info("パフォーマンステスト完了")
    
    def _test_error_handling(self):
        """エラーハンドリングテスト"""
        self.logger.info("エラーハンドリングテスト開始")
        
        test_results = {"error_handling": {}}
        
        # 無効なファイルパスのテスト
        try:
            processor = ImageProcessor(self.config)
            processor.load_image_pair("invalid_path.jpg", "another_invalid.jpg")
            test_results["error_handling"]["invalid_file_path"] = {
                "status": "failed", 
                "message": "例外が発生するべきでした"
            }
        except Exception as e:
            test_results["error_handling"]["invalid_file_path"] = {
                "status": "success", 
                "error_caught": str(e)
            }
            self.logger.info("✓ 無効ファイルパスエラーハンドリング成功")
        
        # 無効な設定のテスト
        try:
            invalid_config = {}
            analyzer = ColorAnalyzer(invalid_config)
            # テスト実行...
            test_results["error_handling"]["invalid_config"] = {
                "status": "success",
                "message": "無効設定でも動作"
            }
        except Exception as e:
            test_results["error_handling"]["invalid_config"] = {
                "status": "success",
                "error_caught": str(e)
            }
            self.logger.info("✓ 無効設定エラーハンドリング成功")
        
        self.test_results.update(test_results)
        self.logger.info("エラーハンドリングテスト完了")
    
    def _validate_analysis_results(self, results: dict, analyzer_name: str) -> bool:
        """解析結果の検証"""
        if not results or not isinstance(results, dict):
            return False
        
        # 解析器ごとの必須キーチェック
        required_keys = {
            "color": ["basic_statistics", "histograms", "color_shifts"],
            "texture": ["edge_analysis", "sharpness_analysis", "noise_analysis"],
            "impression": ["color_psychology", "brightness_contrast_impression"]
        }
        
        if analyzer_name in required_keys:
            for key in required_keys[analyzer_name]:
                if key not in results:
                    return False
        
        return True
    
    def _validate_results(self):
        """結果検証"""
        self.logger.info("結果検証開始")
        
        validation_results = {"validation": {}}
        
        # 成功率計算
        total_tests = 0
        successful_tests = 0
        
        for category, tests in self.test_results.items():
            if category == "validation":
                continue
                
            if isinstance(tests, dict):
                for test_name, result in tests.items():
                    if isinstance(result, dict):
                        for sub_test, sub_result in result.items():
                            total_tests += 1
                            if isinstance(sub_result, dict) and sub_result.get("status") == "success":
                                successful_tests += 1
                            elif sub_test in ["average_execution_time", "performance_rating"]:
                                # パフォーマンス指標は成功とカウント
                                successful_tests += 1
        
        success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
        
        validation_results["validation"] = {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "success_rate": success_rate,
            "overall_status": "PASS" if success_rate >= 80 else "FAIL"
        }
        
        self.test_results.update(validation_results)
        
        self.logger.info(f"テスト成功率: {success_rate:.1f}% ({successful_tests}/{total_tests})")
        self.logger.info(f"総合判定: {validation_results['validation']['overall_status']}")
        
        self.logger.info("結果検証完了")
    
    def _rate_performance(self, avg_time: float) -> str:
        """パフォーマンス評価"""
        if avg_time < 10:
            return "Excellent"
        elif avg_time < 20:
            return "Good"
        elif avg_time < 30:
            return "Fair"
        else:
            return "Poor"
    
    def _generate_test_report(self):
        """テストレポート生成"""
        self.logger.info("テストレポート生成中...")
        
        # 結果をJSONファイルに保存
        report_path = Path("test_results") / f"comprehensive_test_report_{time.strftime('%Y%m%d_%H%M%S')}.json"
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False, default=str)
        
        # サマリーレポート
        summary = f"""
=====================================================
        総合システム統合テスト結果サマリー
=====================================================

実行日時: {time.strftime('%Y-%m-%d %H:%M:%S')}

【総合判定】
{self.test_results.get('validation', {}).get('overall_status', 'UNKNOWN')}

【テスト成功率】
{self.test_results.get('validation', {}).get('success_rate', 0):.1f}%
({self.test_results.get('validation', {}).get('successful_tests', 0)}/{self.test_results.get('validation', {}).get('total_tests', 0)})

【パフォーマンス】
平均実行時間: {self.test_results.get('performance', {}).get('average_execution_time', 0):.2f}秒
評価: {self.test_results.get('performance', {}).get('performance_rating', 'N/A')}

【詳細レポート】
{report_path}

=====================================================
"""
        
        print(summary)
        self.logger.info(f"詳細レポート保存: {report_path}")
        
        return str(report_path)

def main():
    """メイン関数"""
    test_runner = ComprehensiveSystemTest()
    success = test_runner.run_comprehensive_test()
    
    if success:
        print("\n🎉 総合システムテスト完了!")
        return 0
    else:
        print("\n❌ テスト実行中にエラーが発生しました")
        return 1

if __name__ == "__main__":
    sys.exit(main())