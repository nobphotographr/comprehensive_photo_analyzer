"""
Configuration Manager

設定管理モジュール
YAML設定ファイルの読み込み、デフォルト設定の管理を行う
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging


class ConfigManager:
    """設定管理クラス"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config_dir = Path(__file__).parent.parent.parent / "config"
        
    def get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定を取得"""
        return {
            "analysis": {
                "phase": "basic",  # basic, intermediate, advanced, all
                "analyzers": ["color"],  # 使用する解析器
                "color_spaces": ["RGB", "HSV", "LAB"],  # 解析する色空間
                "precision": "standard",  # standard, high, ultra
            },
            "processing": {
                "max_image_size": [7000, 7000],  # 最大画像サイズ
                "resize_for_analysis": True,  # 解析用リサイズ
                "analysis_size": [2000, 2000],  # 解析用サイズ
                "preserve_aspect_ratio": True,
            },
            "output": {
                "format": "html",  # html, json, csv, all
                "directory": "data/output",
                "include_raw_data": True,
                "include_visualizations": True,
                "compression": False,
            },
            "performance": {
                "multiprocessing": True,
                "max_workers": 4,
                "memory_limit": "4GB",
                "gpu_acceleration": False,
            },
            "logging": {
                "level": "INFO",
                "file_output": True,
                "console_output": True,
                "log_directory": "logs",
            }
        }
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """設定ファイルを読み込み"""
        try:
            config_file = Path(config_path)
            
            if not config_file.exists():
                self.logger.warning(f"設定ファイルが見つかりません: {config_path}")
                return self.get_default_config()
            
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # デフォルト設定とマージ
            default_config = self.get_default_config()
            merged_config = self._merge_configs(default_config, config)
            
            self.logger.info(f"設定ファイルを読み込みました: {config_path}")
            return merged_config
            
        except Exception as e:
            self.logger.error(f"設定ファイル読み込みエラー: {e}")
            return self.get_default_config()
    
    def save_config(self, config: Dict[str, Any], config_path: str) -> bool:
        """設定をファイルに保存"""
        try:
            config_file = Path(config_path)
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            
            self.logger.info(f"設定を保存しました: {config_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"設定保存エラー: {e}")
            return False
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """設定の妥当性チェック"""
        try:
            # 必須キーの存在チェック
            required_keys = ["analysis", "processing", "output"]
            for key in required_keys:
                if key not in config:
                    self.logger.error(f"必須設定キーが不足: {key}")
                    return False
            
            # Phase設定の妥当性チェック
            valid_phases = ["basic", "1-3", "intermediate", "4-6", "advanced", "7-9", "all"]
            if config["analysis"]["phase"] not in valid_phases:
                self.logger.error(f"無効なPhase設定: {config['analysis']['phase']}")
                return False
            
            # 出力形式の妥当性チェック
            valid_formats = ["html", "json", "csv", "all"]
            if config["output"]["format"] not in valid_formats:
                self.logger.error(f"無効な出力形式: {config['output']['format']}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"設定検証エラー: {e}")
            return False
    
    def get_phase_config(self, phase: str) -> Dict[str, Any]:
        """Phase別の設定を取得"""
        phase_configs = {
            "basic": {
                "analyzers": ["color"],
                "precision": "standard",
                "features": ["histogram", "statistics", "color_shift"]
            },
            "1-3": {
                "analyzers": ["color"],
                "precision": "standard", 
                "features": ["histogram", "statistics", "color_shift"]
            },
            "intermediate": {
                "analyzers": ["color", "texture"],
                "precision": "high",
                "features": ["histogram", "statistics", "color_shift", "grain", "edges"]
            },
            "4-6": {
                "analyzers": ["color", "texture", "optical"],
                "precision": "high",
                "features": ["histogram", "statistics", "color_shift", "grain", "edges", "lens"]
            },
            "advanced": {
                "analyzers": ["color", "texture", "optical", "impression"],
                "precision": "ultra",
                "features": ["all"]
            },
            "7-9": {
                "analyzers": ["color", "texture", "optical", "impression", "frequency"],
                "precision": "ultra",
                "features": ["all"]
            },
            "all": {
                "analyzers": ["all"],
                "precision": "ultra",
                "features": ["all"]
            }
        }
        
        return phase_configs.get(phase, phase_configs["basic"])
    
    def _merge_configs(self, default: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
        """設定をマージ（再帰的）"""
        result = default.copy()
        
        for key, value in user.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result