# src/feature_engineering/user_behavior_features.py
import pandas as pd
import numpy as np
from tqdm import tqdm
import gc
from .feature_generator_base import FeatureGeneratorBase
from .feature_utils import time_decay_weight, safe_division, split_into_sessions

class UserBehaviorFeatures(FeatureGeneratorBase):
    def __init__(self, generator_name: str,
                 generator_specific_config: dict,
                 global_feature_engineering_config: dict,
                 user_log_df: pd.DataFrame, items_df: pd.DataFrame,
                 project_root_path: str):
        super().__init__(generator_name,
                         generator_specific_config,
                         global_feature_engineering_config,
                         user_log_df, items_df,
                         project_root_path)
        self.params = self.config.get("params", {})
        self.windows_days = self.params.get("windows_days", [1, 3, 7, 15, 30])
        self.hour_windows = self.params.get("hour_windows", [1, 3, 6, 12, 24])
        self.behavior_types = self.params.get("behavior_types", [1, 2, 3, 4])
        self.gen_decayed_counts = self.params.get("generate_decayed_counts", True)
        self.gen_distinct_item_counts = self.params.get("generate_distinct_item_counts", True)
        self.gen_distinct_category_counts = self.params.get("generate_distinct_category_counts", True)
        self.gen_conversion_rates = self.params.get("generate_conversion_rates", True)
        self.gen_activity_features = self.params.get("generate_activity_features", True)
        self.gen_session_features = self.params.get("generate_session_features", False)
        self.time_decay_lambda = self.params.get("time_decay_lambda", 0.01)
        self.session_timeout_minutes = self.params.get("session_timeout_minutes", 30)

        all_possible_names = self._get_all_expected_feature_names()
        for name in all_possible_names:
            self._add_feature_name(name)

    def _get_all_expected_feature_names(self) -> list:
        temp_feature_names = []
        time_windows_config = [(d, 'd') for d in self.windows_days] + \
                              [(h, 'h') for h in self.hour_windows]

        for window_val, window_unit in time_windows_config:
            for b_type in self.behavior_types:
                temp_feature_names.append(f"user_cnt_b{b_type}_last_{window_val}{window_unit}")
                if self.gen_decayed_counts:
                    temp_feature_names.append(f"user_decayed_cnt_b{b_type}_last_{window_val}{window_unit}")
            if self.gen_distinct_item_counts:
                temp_feature_names.append(f"user_distinct_items_last_{window_val}{window_unit}")
            if self.gen_distinct_category_counts:
                temp_feature_names.append(f"user_distinct_categories_last_{window_val}{window_unit}")

        if self.gen_conversion_rates:
            temp_feature_names.extend([
                "user_cvr_1_to_3_all_time", "user_cvr_1_to_2_all_time",
                "user_cvr_3_to_4_all_time", "user_cvr_2_to_4_all_time", "user_cvr_1_to_4_all_time"
            ])
        if self.gen_activity_features:
            max_window_for_activity = max(self.windows_days) if self.windows_days else 30
            temp_feature_names.extend([
                f"user_active_days_{max_window_for_activity}d",
                f"user_total_actions_{max_window_for_activity}d",
                f"user_avg_daily_actions_{max_window_for_activity}d",
                f"user_purchase_days_{max_window_for_activity}d"
            ])
        if self.gen_session_features:
            temp_feature_names.extend([
                "user_num_sessions_total",
                "user_avg_session_duration_hours", "user_avg_session_actions",
                "user_avg_session_distinct_items", "user_avg_session_distinct_categories",
                "user_last_session_duration_hours", "user_last_session_actions",
                "user_last_session_distinct_items", "user_last_session_distinct_categories"
            ])
        return list(set(temp_feature_names))


    def generate_features(self, candidates_df: pd.DataFrame, behavior_end_date: pd.Timestamp) -> pd.DataFrame:
        print(f"  生成 UserBehaviorFeatures (用户全局行为特征) 直到 {behavior_end_date.strftime('%Y-%m-%d %H:%M:%S')}")

        unique_user_ids_series = candidates_df['user_id'].unique()
        num_unique_users = len(unique_user_ids_series)

        if num_unique_users == 0:
            print("    候选集中没有用户，UserBehaviorFeatures 将返回一个包含预期列的空 DataFrame。")
            expected_cols = ['user_id'] + self.get_generated_feature_names()
            return pd.DataFrame(columns=expected_cols)

        # 1. 初始化最终的特征 DataFrame (Fix for PerformanceWarning)
        feature_cols_init = self.get_generated_feature_names()
        # Create a DataFrame with 0.0s directly, more efficient
        final_user_features_df = pd.DataFrame(0.0, index=unique_user_ids_series, columns=feature_cols_init)
        final_user_features_df.index.name = 'user_id'


        user_log_for_features = self.user_log_df[
            (self.user_log_df['user_id'].isin(unique_user_ids_series)) &
            (self.user_log_df['datetime'] <= behavior_end_date)
        ].copy()

        if user_log_for_features.empty:
            print("    没有找到相关用户的历史行为日志，UserBehaviorFeatures 将主要为0。")
            return final_user_features_df.reset_index() # Ensure user_id is a column

        if self.gen_decayed_counts:
            user_log_for_features['delta_hours'] = \
                (behavior_end_date - user_log_for_features['datetime']).dt.total_seconds() / 3600
            user_log_for_features['decay_weight'] = \
                time_decay_weight(user_log_for_features['delta_hours'], self.time_decay_lambda)
        
        gc.collect()

        time_windows_config = [(d, 'd', pd.Timedelta(days=d)) for d in self.windows_days] + \
                              [(h, 'h', pd.Timedelta(hours=h)) for h in self.hour_windows]

        for window_val, window_unit, timedelta_obj in tqdm(time_windows_config, desc="  UF: Processing time windows", total=len(time_windows_config)):
            window_start_time = behavior_end_date - timedelta_obj + pd.Timedelta(seconds=1)
            log_in_window = user_log_for_features[
                (user_log_for_features['datetime'] >= window_start_time)
            ].copy() # Use .copy() when slicing for modification or further complex ops

            if log_in_window.empty:
                continue

            grouped_by_user_in_window = log_in_window.groupby('user_id')

            for b_type in self.behavior_types:
                feat_name_count = f"user_cnt_b{b_type}_last_{window_val}{window_unit}"
                counts = log_in_window[log_in_window['behavior_type'] == b_type].groupby('user_id').size()
                if not counts.empty:
                    final_user_features_df[feat_name_count] = final_user_features_df[feat_name_count].add(counts, fill_value=0)

            if self.gen_decayed_counts and 'decay_weight' in log_in_window.columns:
                for b_type in self.behavior_types:
                    feat_name_decayed = f"user_decayed_cnt_b{b_type}_last_{window_val}{window_unit}"
                    decayed_counts = log_in_window[log_in_window['behavior_type'] == b_type].groupby('user_id')['decay_weight'].sum()
                    if not decayed_counts.empty:
                        final_user_features_df[feat_name_decayed] = final_user_features_df[feat_name_decayed].add(decayed_counts, fill_value=0)

            if self.gen_distinct_item_counts:
                feat_name_items = f"user_distinct_items_last_{window_val}{window_unit}"
                distinct_items = grouped_by_user_in_window['item_id'].nunique()
                if not distinct_items.empty:
                    final_user_features_df[feat_name_items] = final_user_features_df[feat_name_items].add(distinct_items, fill_value=0)

            if self.gen_distinct_category_counts:
                feat_name_cats = f"user_distinct_categories_last_{window_val}{window_unit}"
                distinct_cats = grouped_by_user_in_window['item_category'].nunique()
                if not distinct_cats.empty:
                    final_user_features_df[feat_name_cats] = final_user_features_df[feat_name_cats].add(distinct_cats, fill_value=0)
            
            del log_in_window, grouped_by_user_in_window
            gc.collect()

        if self.gen_conversion_rates and not user_log_for_features.empty:
            pivot_counts_all_time = user_log_for_features.groupby(['user_id', 'behavior_type']).size().unstack(fill_value=0)
            
            # Ensure all behavior type columns [1, 2, 3, 4] exist in pivot_counts_all_time
            for b_type_col in self.behavior_types:
                if b_type_col not in pivot_counts_all_time.columns:
                    pivot_counts_all_time[b_type_col] = 0 # Add missing column with zeros

            # Prepare Series for safe_division, ensuring they align with final_user_features_df's index
            # The .get(col, pd.Series(...)) pattern is good, but ensure the default Series aligns correctly.
            # A simpler way is to reindex the pivot table to match all unique users.
            pivot_counts_all_time = pivot_counts_all_time.reindex(final_user_features_df.index, fill_value=0)

            final_user_features_df["user_cvr_1_to_3_all_time"] = safe_division(pivot_counts_all_time[3], pivot_counts_all_time[1])
            final_user_features_df["user_cvr_1_to_2_all_time"] = safe_division(pivot_counts_all_time[2], pivot_counts_all_time[1])
            final_user_features_df["user_cvr_3_to_4_all_time"] = safe_division(pivot_counts_all_time[4], pivot_counts_all_time[3])
            final_user_features_df["user_cvr_2_to_4_all_time"] = safe_division(pivot_counts_all_time[4], pivot_counts_all_time[2])
            final_user_features_df["user_cvr_1_to_4_all_time"] = safe_division(pivot_counts_all_time[4], pivot_counts_all_time[1])
            del pivot_counts_all_time
            gc.collect()

        if self.gen_activity_features:
            max_window_days = max(self.windows_days) if self.windows_days else 30
            activity_start_time = behavior_end_date - pd.Timedelta(days=max_window_days) + pd.Timedelta(seconds=1)
            log_for_activity = user_log_for_features[user_log_for_features['datetime'] >= activity_start_time].copy()

            if not log_for_activity.empty:
                log_for_activity.loc[:, 'date_normalized'] = log_for_activity['datetime'].dt.normalize()
                
                user_activity_stats_list = []
                # Active days and total actions
                agg_active_total = log_for_activity.groupby('user_id').agg(
                    active_days=(pd.NamedAgg(column='date_normalized', aggfunc='nunique')),
                    total_actions=(pd.NamedAgg(column='datetime', aggfunc='count'))
                ).rename(columns={
                    'active_days': f"user_active_days_{max_window_days}d",
                    'total_actions': f"user_total_actions_{max_window_days}d"
                })
                user_activity_stats_list.append(agg_active_total)

                # Purchase days
                user_purchase_days = log_for_activity[log_for_activity['behavior_type'] == 4].groupby('user_id')['date_normalized'].nunique().rename(f"user_purchase_days_{max_window_days}d")
                user_activity_stats_list.append(user_purchase_days)
                
                # Join all activity stats
                if user_activity_stats_list:
                    user_activity_stats = pd.concat(user_activity_stats_list, axis=1)
                    # Fill NaN for users who might not have purchases etc.
                    activity_feature_names = [
                        f"user_active_days_{max_window_days}d",
                        f"user_total_actions_{max_window_days}d",
                        f"user_purchase_days_{max_window_days}d"
                    ]
                    for col_name in activity_feature_names: # Ensure columns exist before filling
                        if col_name not in user_activity_stats.columns:
                            user_activity_stats[col_name] = 0
                    user_activity_stats = user_activity_stats.fillna(0)


                    user_activity_stats[f"user_avg_daily_actions_{max_window_days}d"] = safe_division(
                        user_activity_stats[f"user_total_actions_{max_window_days}d"],
                        user_activity_stats[f"user_active_days_{max_window_days}d"]
                    )
                    
                    for col in user_activity_stats.columns:
                        if col in final_user_features_df.columns:
                             final_user_features_df[col] = final_user_features_df[col].add(user_activity_stats[col], fill_value=0)
                    del user_activity_stats
                del user_activity_stats_list
            del log_for_activity
            gc.collect()

        if self.gen_session_features and not user_log_for_features.empty:
            print("    UF: Calculating session features...")
            user_log_with_sessions = split_into_sessions(
                user_log_for_features.copy(),
                self.session_timeout_minutes,
                user_col='user_id', time_col='datetime'
            )
            if 'session_id' in user_log_with_sessions.columns and not user_log_with_sessions.empty:
                user_num_sessions = user_log_with_sessions.groupby('user_id')['session_id'].nunique().rename("user_num_sessions_total")
                if not user_num_sessions.empty:
                     final_user_features_df["user_num_sessions_total"] = final_user_features_df["user_num_sessions_total"].add(user_num_sessions, fill_value=0)

                session_level_stats = user_log_with_sessions.groupby(['user_id', 'session_id']).agg(
                    session_start_time=('datetime', 'min'),
                    session_end_time=('datetime', 'max'),
                    session_actions_count=('datetime', 'count'),
                    session_distinct_items=('item_id', 'nunique'),
                    session_distinct_categories=('item_category', 'nunique')
                )

                session_level_stats['session_duration_seconds'] = \
                    (session_level_stats['session_end_time'] - session_level_stats['session_start_time']).dt.total_seconds()
                session_level_stats.loc[session_level_stats['session_actions_count'] <= 1, 'session_duration_seconds'] = 0.0
                session_level_stats['session_duration_hours'] = session_level_stats['session_duration_seconds'] / 3600
                
                user_avg_session_stats = session_level_stats.groupby('user_id').agg(
                    user_avg_session_duration_hours=('session_duration_hours', 'mean'),
                    user_avg_session_actions=('session_actions_count', 'mean'),
                    user_avg_session_distinct_items=('session_distinct_items', 'mean'),
                    user_avg_session_distinct_categories=('session_distinct_categories', 'mean')
                )
                for col in user_avg_session_stats.columns:
                    if col in final_user_features_df.columns:
                        final_user_features_df[col] = final_user_features_df[col].add(user_avg_session_stats[col], fill_value=0)
                
                session_level_stats_reset = session_level_stats.reset_index() # Need user_id as column for idxmax
                last_session_indices = session_level_stats_reset.loc[session_level_stats_reset.groupby('user_id')['session_start_time'].idxmax()]
                user_last_session_df = last_session_indices.set_index('user_id')


                if not user_last_session_df.empty:
                    final_user_features_df['user_last_session_duration_hours'] = final_user_features_df['user_last_session_duration_hours'].add(user_last_session_df['session_duration_hours'], fill_value=0)
                    final_user_features_df['user_last_session_actions'] = final_user_features_df['user_last_session_actions'].add(user_last_session_df['session_actions_count'], fill_value=0)
                    final_user_features_df['user_last_session_distinct_items'] = final_user_features_df['user_last_session_distinct_items'].add(user_last_session_df['session_distinct_items'], fill_value=0)
                    final_user_features_df['user_last_session_distinct_categories'] = final_user_features_df['user_last_session_distinct_categories'].add(user_last_session_df['session_distinct_categories'], fill_value=0)

                del user_log_with_sessions, session_level_stats, user_avg_session_stats, user_last_session_df, session_level_stats_reset
                gc.collect()
            print("    UF: Session features calculated.")

        final_user_features_df = final_user_features_df.reset_index()
        
        # Ensure all expected columns are present and fill NaNs that might have occurred
        # (though add with fill_value=0 and safe_division should prevent most NaNs)
        expected_feature_cols_final = self.get_generated_feature_names()
        for col in expected_feature_cols_final:
            if col not in final_user_features_df.columns:
                final_user_features_df[col] = 0.0
        final_user_features_df.fillna(0.0, inplace=True)

        final_cols_order = ['user_id'] + expected_feature_cols_final
        # Ensure no duplicate columns before reordering (just in case)
        final_user_features_df = final_user_features_df.loc[:,~final_user_features_df.columns.duplicated()]
        final_user_features_df = final_user_features_df[final_cols_order]


        print(f"  UserBehaviorFeatures 生成完毕. 最终形状: {final_user_features_df.shape}")
        return final_user_features_df