# src/recall/strategies/popular_items_strategy.py
import pandas as pd
from src.recall.recall_strategy import RecallStrategy
from datetime import timedelta

class GlobalPopularItemsStrategy(RecallStrategy):
    def __init__(self, strategy_name: str, processed_data_path: str,
                 strategy_specific_config: dict,
                 user_log_grouped: dict = None, # 接收，但本策略通常不直接用
                 items_df_global: pd.DataFrame = None,
                 precomputed_popular_items_series: pd.Series = None): # 接收预计算结果
        super().__init__(strategy_name, processed_data_path, strategy_specific_config,
                         user_log_grouped, items_df_global)
        
        # self.days_window_popularity 和 self.min_interactions 由 calculate_popular_items_logic 的调用方 (run_recall.py) 控制
        self.popular_items_series = precomputed_popular_items_series # 使用预计算结果

    @staticmethod
    def calculate_popular_items_logic(
            user_log_df_full: pd.DataFrame,
            item_pool_set_to_filter: set,
            behavior_data_end_date_str: str, # 这个参数现在其实不直接用了，因为窗口天数从外部传
            days_window: int, # 改为接收具体的窗口天数
            min_interactions_threshold: int,
            strategy_name_for_log: str = "PopularItemsLogic"
        ) -> pd.Series:
        if user_log_df_full.empty:
            # print(f"Strategy '{strategy_name_for_log}': User log data is empty for popularity calculation.")
            return pd.Series(dtype='float64')

        end_date_inclusive = pd.to_datetime(behavior_data_end_date_str) + pd.Timedelta(hours=23, minutes=59, seconds=59)
        start_date = end_date_inclusive - timedelta(days=days_window) + pd.Timedelta(seconds=1)

        if 'datetime' in user_log_df_full.columns and not pd.api.types.is_datetime64_any_dtype(user_log_df_full['datetime']):
            user_log_df_full['datetime'] = pd.to_datetime(user_log_df_full['datetime'])

        relevant_logs = user_log_df_full[
            (user_log_df_full['datetime'] >= start_date) &
            (user_log_df_full['datetime'] <= end_date_inclusive)
        ]

        if relevant_logs.empty:
            return pd.Series(dtype='float64')

        item_counts = relevant_logs['item_id'].value_counts()
        popular_items_series = item_counts[item_counts >= min_interactions_threshold]
        
        if item_pool_set_to_filter and not popular_items_series.empty:
            popular_items_series = popular_items_series[popular_items_series.index.isin(item_pool_set_to_filter)]
        
        return popular_items_series.sort_values(ascending=False)

    def get_candidates(self, user_id: int, behavior_data_end_date_str: str, N: int) -> pd.DataFrame:
        # behavior_data_end_date_str 在此策略中不直接使用，因为热门榜是预计算的
        if self.popular_items_series is None or self.popular_items_series.empty:
             # print(f"Strategy '{self.strategy_name}': Precomputed popular items series is None or empty.")
             return pd.DataFrame(columns=['user_id', 'item_id', 'recall_score', 'recall_source'])

        hot_item_ids_with_scores = self.popular_items_series.head(N)

        if hot_item_ids_with_scores.empty:
            return pd.DataFrame(columns=['user_id', 'item_id', 'recall_score', 'recall_source'])

        output_df = pd.DataFrame({
            'user_id': user_id,
            'item_id': hot_item_ids_with_scores.index,
            'recall_score': hot_item_ids_with_scores.values,
            'recall_source': self.strategy_name
        })
        return output_df