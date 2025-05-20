# src/feature_engineering/temporal_features.py
import pandas as pd
from .feature_generator_base import FeatureGeneratorBase # 确保正确导入基类

class TemporalFeatures(FeatureGeneratorBase):
    def __init__(self, generator_name: str,
                 generator_specific_config: dict,
                 global_feature_engineering_config: dict,
                 user_log_df: pd.DataFrame, items_df: pd.DataFrame,
                 project_root_path: str): # <--- 添加 project_root_path
        # 调用父类的 __init__ 方法，并将所有必要的参数传递过去
        super().__init__(generator_name,
                         generator_specific_config,
                         global_feature_engineering_config,
                         user_log_df, items_df,
                         project_root_path) # <--- 传递 project_root_path
        self.params = self.config.get("params", {})

    def generate_features(self, candidates_df: pd.DataFrame, behavior_end_date: pd.Timestamp) -> pd.DataFrame:
        prediction_datetime = behavior_end_date + pd.Timedelta(days=1)
        # print(f"  正在为预测日期生成时间特征: {prediction_datetime.date()}") # 这句日志在 run_feature_engineer 中有了

        num_candidates = len(candidates_df)
        new_features_dict = {}

        if self.params.get("include_day_of_week"):
            col_name = "pred_day_of_week"
            day_of_week_value = prediction_datetime.dayofweek
            new_features_dict[col_name] = pd.Series([day_of_week_value] * num_candidates, index=candidates_df.index)
            self._add_feature_name(col_name)

        if self.params.get("include_is_weekend"):
            col_name = "pred_is_weekend"
            is_weekend_int = int(prediction_datetime.dayofweek >= 5)
            new_features_dict[col_name] = pd.Series([is_weekend_int] * num_candidates, index=candidates_df.index)
            self._add_feature_name(col_name)

        if self.params.get("include_hour_of_day"):
            col_name = "pred_hour_of_day"
            hour_of_day_value = prediction_datetime.hour
            new_features_dict[col_name] = pd.Series([hour_of_day_value] * num_candidates, index=candidates_df.index)
            self._add_feature_name(col_name)

        result_df = candidates_df[['user_id', 'item_id']].copy()
        if new_features_dict:
            # result_df = pd.concat([result_df, pd.DataFrame(new_features_dict, index=candidates_df.index)], axis=1)
            # 更安全的做法，避免因索引不完全匹配导致concat行为异常 (尽管这里index应该是一样的)
            for col, series_val in new_features_dict.items():
                result_df[col] = series_val

        # print(f"    已生成时间特征: {self.get_generated_feature_names()}") # 这句日志在 run_feature_engineer 中有了
        return result_df