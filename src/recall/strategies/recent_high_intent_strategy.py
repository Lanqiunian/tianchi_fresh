# src/recall/strategies/recent_high_intent_strategy.py
import pandas as pd
from datetime import timedelta 
from src.recall.recall_strategy import RecallStrategy

class RecentHighIntentStrategy(RecallStrategy):
    def __init__(self, strategy_name: str, processed_data_path: str,
                 strategy_specific_config: dict,
                 user_log_grouped: dict = None, 
                 items_df_global: pd.DataFrame = None):
        # 注意：此策略不接收 user_specific_preprocessed_history 或 precomputed_popular_items_series
        super().__init__(strategy_name, processed_data_path, strategy_specific_config,
                         user_log_grouped, items_df_global)
        
        self.days_window = self.config.get("days_window", 7)
        self.behavior_weights = self.config.get("behavior_weights", None)
        
        if self.behavior_weights:
            self.behavior_weights = {int(k): v for k, v in self.behavior_weights.items()}
            self.behavior_types_target = list(self.behavior_weights.keys())
        else: 
            self.behavior_types_target = self.config.get("behavior_types", [1, 2, 3])
            self.behavior_weights = {int(bt): 1.0 for bt in self.behavior_types_target}

    def get_candidates(self, user_id: int, behavior_data_end_date_str: str, N: int) -> pd.DataFrame:
        user_interactions = super()._get_user_interactions(
            user_id=user_id,
            end_date_str=behavior_data_end_date_str, # 基类会处理这个日期
            days_window=self.days_window # 使用本策略配置的窗口
        )

        if user_interactions.empty:
            return pd.DataFrame(columns=['user_id', 'item_id', 'recall_score', 'recall_source'])
            
        high_intent_actions = user_interactions[
            user_interactions['behavior_type'].isin(self.behavior_types_target)
        ].copy()

        if high_intent_actions.empty:
            return pd.DataFrame(columns=['user_id', 'item_id', 'recall_score', 'recall_source'])

        high_intent_actions['base_score'] = high_intent_actions['behavior_type'].map(self.behavior_weights)
        
        behavior_data_end_datetime = pd.to_datetime(behavior_data_end_date_str) + pd.Timedelta(hours=23, minutes=59, seconds=59)
        window_in_seconds = self.days_window * 24 * 60 * 60
        
        high_intent_actions['age_in_seconds'] = (behavior_data_end_datetime - high_intent_actions['datetime']).dt.total_seconds()
        high_intent_actions['age_in_seconds'] = high_intent_actions['age_in_seconds'].clip(lower=0)
        
        if window_in_seconds > 0:
            high_intent_actions['time_weight'] = (window_in_seconds - high_intent_actions['age_in_seconds'].clip(upper=window_in_seconds)) / window_in_seconds
        else: 
            high_intent_actions['time_weight'] = (high_intent_actions['age_in_seconds'] == 0).astype(float)
            
        high_intent_actions['time_weight'] = high_intent_actions['time_weight'].clip(lower=0, upper=1)
        high_intent_actions['interaction_score'] = high_intent_actions['base_score'] * high_intent_actions['time_weight']
        
        if high_intent_actions.empty or 'item_id' not in high_intent_actions.columns:
             return pd.DataFrame(columns=['user_id', 'item_id', 'recall_score', 'recall_source'])

        item_scores_df = high_intent_actions.sort_values(
            ['item_id', 'interaction_score'], ascending=[True, False]
        ).drop_duplicates(subset=['item_id'], keep='first')

        recalled_candidates = item_scores_df.sort_values(
            by='interaction_score', ascending=False
        ).head(N)

        if recalled_candidates.empty:
            return pd.DataFrame(columns=['user_id', 'item_id', 'recall_score', 'recall_source'])

        output_df = pd.DataFrame({
            'user_id': user_id,
            'item_id': recalled_candidates['item_id'],
            'recall_score': recalled_candidates['interaction_score'],
            'recall_source': self.strategy_name
        })
        return output_df