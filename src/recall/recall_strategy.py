# src/recall/recall_strategy.py
from abc import ABC, abstractmethod
import pandas as pd
import os
from datetime import timedelta

class RecallStrategy(ABC):
    def __init__(self, strategy_name: str, processed_data_path: str,
                 strategy_specific_config: dict,
                 user_log_grouped: dict = None, # {user_id: user_df}
                 items_df_global: pd.DataFrame = None,
                 ):
        self.strategy_name = strategy_name
        self.processed_data_path = processed_data_path
        self.config = strategy_specific_config

        self.user_log_grouped = user_log_grouped if user_log_grouped is not None else {}
        self.items_df = items_df_global if items_df_global is not None else pd.DataFrame()

        # if self.user_log_grouped: # 日志可能过多，只在必要时打印或在工厂处打印一次
            # print(f"Strategy '{self.strategy_name}': Using provided pre-grouped user log for {len(self.user_log_grouped)} users.")

        if not self.items_df.empty:
            self.item_pool_set = set(self.items_df['item_id'].unique())
        else:
            self.item_pool_set = set()

    def _get_user_log_for_user(self, user_id: int) -> pd.DataFrame:
        user_df = self.user_log_grouped.get(user_id)
        if user_df is None or user_df.empty:
            return pd.DataFrame(columns=['user_id', 'item_id', 'behavior_type', 'datetime', 'item_category'])
        
        # 确保datetime列是datetime类型 (应该在预分组前就处理好，这里是后备)
        if 'datetime' in user_df.columns and not pd.api.types.is_datetime64_any_dtype(user_df['datetime']):
             user_df['datetime'] = pd.to_datetime(user_df['datetime'])
        return user_df.copy() # 返回副本

    def _get_user_interactions(self, user_id: int, end_date_str: str, days_window: int = None) -> pd.DataFrame:
        user_interactions_df = self._get_user_log_for_user(user_id)

        if user_interactions_df.empty:
            return user_interactions_df

        effective_end_datetime_inclusive = pd.to_datetime(end_date_str) + pd.Timedelta(hours=23, minutes=59, seconds=59)
        user_interactions_df = user_interactions_df[user_interactions_df['datetime'] <= effective_end_datetime_inclusive]

        if days_window is not None and days_window > 0:
            start_date = effective_end_datetime_inclusive - pd.Timedelta(days=days_window) + pd.Timedelta(seconds=1)
            user_interactions_df = user_interactions_df[user_interactions_df['datetime'] >= start_date]
        
        return user_interactions_df

    @abstractmethod
    def get_candidates(self, user_id: int, behavior_data_end_date_str: str, N: int) -> pd.DataFrame:
        pass

    def _filter_by_item_pool(self, item_ids: list) -> list:
        if not self.item_pool_set:
            return item_ids
        return [item_id for item_id in item_ids if item_id in self.item_pool_set]