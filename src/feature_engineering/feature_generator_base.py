# src/feature_engineering/feature_generator_base.py
from abc import ABC, abstractmethod
import pandas as pd
from .feature_utils import parse_behavior_end_date # 假设你 utils/config_loader 不再由基类使用

class FeatureGeneratorBase(ABC):
    def __init__(self, generator_name: str,
                 generator_specific_config: dict,
                 global_feature_engineering_config: dict,
                 user_log_df: pd.DataFrame, items_df: pd.DataFrame,
                 project_root_path: str # <--- 确认这个参数在这里
                 ):
        self.generator_name = generator_name
        self.config = generator_specific_config # 这是该生成器的特定配置部分 (包含 module, class, params)
        self.global_fe_config = global_feature_engineering_config # 全局设置，现在包含了 project_root_path_placeholder
        self.user_log_df = user_log_df
        self.items_df = items_df
        # project_root_path 现在由 run_feature_engineer.py 直接传递
        self.project_root_path = project_root_path # 保存 project_root

        self.feature_names_generated = []

        if not self.config:
            print(f"警告: 生成器 '{generator_name}' 的特定配置为空。")
        # 子类将从 self.config.get("params", {}) 获取它们的参数

    @abstractmethod
    def generate_features(self, candidates_df: pd.DataFrame, behavior_end_date: pd.Timestamp) -> pd.DataFrame:
        pass

    def get_generated_feature_names(self) -> list:
        return list(set(self.feature_names_generated))

    def _filter_log_by_time_window(self, base_log_df: pd.DataFrame,
                                   end_time: pd.Timestamp,
                                   days: int = None, hours: int = None) -> pd.DataFrame:
        if days is not None and days > 0:
            start_time = end_time - pd.Timedelta(days=days)
        elif hours is not None and hours > 0:
            start_time = end_time - pd.Timedelta(hours=hours)
        else:
            return base_log_df[base_log_df['datetime'] <= end_time].copy()
        return base_log_df[(base_log_df['datetime'] <= end_time) & (base_log_df['datetime'] > start_time)].copy()

    def _add_feature_name(self, col_name: str):
        if col_name not in self.feature_names_generated:
            self.feature_names_generated.append(col_name)