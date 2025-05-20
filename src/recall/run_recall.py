# src/recall/run_recall.py
import pandas as pd
import yaml
import os
import time
from datetime import datetime, timedelta
from tqdm import tqdm # type: ignore
import importlib
import gc

# 使用相对导入，因为通常通过 python -m src.recall.run_recall 运行
from .factory import RecallStrategyFactory
from .strategies.popular_items_strategy import GlobalPopularItemsStrategy
from .strategies.itemcf_strategy import ItemCFRecallStrategy # 需要导入以进行 isinstance 检查

def run_recall_for_user_group(
    target_user_ids: list,
    active_strategies: list, # 策略实例现在持有预处理或共享的数据
    global_config: dict,
) -> pd.DataFrame:
    all_user_final_candidates = []
    user_candidate_limit = global_config.get("user_candidate_limit", 200)
    target_prediction_date_str = global_config.get("target_prediction_date", "2014-12-19")
    # 行为数据的最后一天，传递给策略的 get_candidates
    behavior_data_end_date_str = (pd.to_datetime(target_prediction_date_str) - timedelta(days=1)).strftime('%Y-%m-%d')

    for user_id in tqdm(target_user_ids, desc="Recalling users"):
        user_all_strategy_candidates = []
        for strategy in active_strategies:
            try:
                strategy_top_n = strategy.config.get("top_n_recall", 50)
                candidates_df = strategy.get_candidates(user_id, behavior_data_end_date_str, strategy_top_n)

                if candidates_df is not None and not candidates_df.empty:
                    expected_cols = ['user_id', 'item_id', 'recall_score', 'recall_source']
                    if not all(col in candidates_df.columns for col in expected_cols):
                        continue # 跳过格式不正确的
                    
                    # 确保只添加属于当前处理用户的数据
                    correct_user_candidates_df = candidates_df[candidates_df['user_id'] == user_id]
                    if not correct_user_candidates_df.empty:
                        user_all_strategy_candidates.append(correct_user_candidates_df)
            except Exception as e:
                # import traceback # 调试时取消注释
                # traceback.print_exc()
                print(f"  Error running strategy '{strategy.strategy_name}' for user {user_id}: {type(e).__name__} - {e}")
        
        if not user_all_strategy_candidates:
            continue

        merged_candidates_df = pd.concat(user_all_strategy_candidates, ignore_index=True)
        if 'recall_score' in merged_candidates_df.columns:
            merged_candidates_df.sort_values(by='recall_score', ascending=False, inplace=True)
        
        merged_candidates_df.drop_duplicates(subset=['user_id', 'item_id'], keep='first', inplace=True)
        final_user_candidates_df = merged_candidates_df.head(user_candidate_limit)
        all_user_final_candidates.append(final_user_candidates_df)

    if not all_user_final_candidates:
        print("No candidates were recalled for any of the target users.")
        return pd.DataFrame(columns=['user_id', 'item_id', 'recall_score', 'recall_source'])
    
    return pd.concat(all_user_final_candidates, ignore_index=True)


if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    CONFIG_FILE = os.path.join(PROJECT_ROOT, 'conf', 'recall', 'recall_config.yaml')

    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Config File: {CONFIG_FILE}")

    if not os.path.exists(CONFIG_FILE):
        print(f"FATAL: Recall configuration file not found at {CONFIG_FILE}")
        exit(1)
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        main_config_data = yaml.safe_load(f)
    global_recall_config = main_config_data.get("global_settings", {})
    processed_data_dir_cfg = global_recall_config.get("processed_data_path", "data/1_interim/")
    processed_data_dir_abs = os.path.join(PROJECT_ROOT, processed_data_dir_cfg)

    print("\nLoading global dataframes for pre-processing...")
    user_log_path_abs = os.path.join(processed_data_dir_abs, "processed_user_log_p_related.parquet")
    items_path_abs = os.path.join(processed_data_dir_abs, "processed_items.parquet")

    if not os.path.exists(user_log_path_abs) or not os.path.exists(items_path_abs):
        print(f"FATAL: User log ({user_log_path_abs}) or items file ({items_path_abs}) not found.")
        exit(1)
    
    try:
        # 这个 df_all_user_logs_global 主要用于 ItemCF历史预处理 和 热门商品预计算
        df_all_user_logs_global_for_preprocessing = pd.read_parquet(user_log_path_abs)
        df_items_global = pd.read_parquet(items_path_abs)
    except Exception as e:
        print(f"FATAL: Error loading global parquet files: {e}")
        exit(1)

    if 'datetime' in df_all_user_logs_global_for_preprocessing.columns and \
       not pd.api.types.is_datetime64_any_dtype(df_all_user_logs_global_for_preprocessing['datetime']):
        print("Converting 'datetime' column in global user log for pre-processing...")
        df_all_user_logs_global_for_preprocessing['datetime'] = pd.to_datetime(df_all_user_logs_global_for_preprocessing['datetime'])
    print(f"Global user log for pre-processing loaded: {df_all_user_logs_global_for_preprocessing.shape}")
    print(f"Global items df loaded: {df_items_global.shape}")
    item_pool_set_global = set(df_items_global['item_id'].unique()) if not df_items_global.empty else set()


    # --- 预处理用户行为数据 ---
    target_prediction_date_str_from_config = global_recall_config.get("target_prediction_date", "2014-12-19")
    behavior_data_end_date_for_pp_str = (pd.to_datetime(target_prediction_date_str_from_config) - timedelta(days=1)).strftime('%Y-%m-%d')
    effective_end_dt_for_pp = pd.to_datetime(behavior_data_end_date_for_pp_str) + pd.Timedelta(hours=23, minutes=59, seconds=59)

    max_days_window_needed = 0
    itemcf_config = main_config_data.get("strategies", {}).get("item_cf", {}).get("params", {})
    itemcf_user_history_bh_types = itemcf_config.get("user_history_behavior_types")
    itemcf_max_hist_items = itemcf_config.get("max_user_history_items", 50)
    itemcf_days_window = itemcf_config.get("days_window_user_history", 30)
    if itemcf_days_window > max_days_window_needed: max_days_window_needed = itemcf_days_window
    
    # 其他策略也可能需要历史，找到最大的窗口需求
    for strat_name, strat_cfg in main_config_data.get("strategies", {}).items():
        if strat_cfg.get("enabled", False) and strat_name != "item_cf":
            days_w = strat_cfg.get("params", {}).get("days_window", 0) # e.g. recent_high_intent
            if days_w > max_days_window_needed: max_days_window_needed = days_w
    if max_days_window_needed == 0: max_days_window_needed = 30 # Default
    
    start_dt_for_pp = effective_end_dt_for_pp - pd.Timedelta(days=max_days_window_needed) + pd.Timedelta(seconds=1)

    print(f"\nFiltering global logs to max window ({max_days_window_needed} days) for pre-processing...")
    logs_in_max_window_for_all = df_all_user_logs_global_for_preprocessing[
        (df_all_user_logs_global_for_preprocessing['datetime'] >= start_dt_for_pp) &
        (df_all_user_logs_global_for_preprocessing['datetime'] <= effective_end_dt_for_pp)
    ].copy() # 使用副本进行后续操作
    del df_all_user_logs_global_for_preprocessing
    gc.collect()
    print(f"Logs within max window for all pre-processing: {logs_in_max_window_for_all.shape}")


    # 1. ItemCF专属的用户历史预处理
    user_preprocessed_itemcf_histories = {}
    if main_config_data.get("strategies", {}).get("item_cf", {}).get("enabled", False):
        print("\nPre-processing user histories specifically for ItemCF...")
        itemcf_hist_logs_source = logs_in_max_window_for_all # 从已经按最大窗口过滤的日志开始
        if itemcf_user_history_bh_types:
            try:
                valid_bh = [int(bt) for bt in itemcf_user_history_bh_types]
                itemcf_hist_logs_source = itemcf_hist_logs_source[itemcf_hist_logs_source['behavior_type'].isin(valid_bh)]
            except ValueError:
                print(f"Warning: Invalid ItemCF user_history_behavior_types: {itemcf_user_history_bh_types}")

        for user_id, user_df in tqdm(itemcf_hist_logs_source.groupby('user_id'), desc="ItemCF Histories"):
            # 确保item_id在pool中（理论上P相关日志已保证，双保险）
            # user_df_in_pool = user_df[user_df['item_id'].isin(item_pool_set_global)] if item_pool_set_global else user_df
            # 改为：ItemCF的交互历史物品也应在P中，所以直接使用user_df (它来自P相关日志)
            sorted_items = user_df.sort_values(by='datetime', ascending=False)['item_id']
            unique_items = sorted_items.unique()[:itemcf_max_hist_items]
            if len(unique_items) > 0:
                user_preprocessed_itemcf_histories[user_id] = list(unique_items)
        print(f"Pre-processed ItemCF histories for {len(user_preprocessed_itemcf_histories)} users.")
    

    # 2. 其他策略通用的预分组日志 (只按user_id分组，保留需要的列，已按最大时间窗口过滤)
    print("\nPre-grouping user logs for other strategies...")
    cols_for_grouping = ['user_id', 'item_id', 'behavior_type', 'datetime', 'item_category']
    actual_cols_for_grouping = [col for col in cols_for_grouping if col in logs_in_max_window_for_all.columns]
    user_log_grouped_for_other_strategies = {}
    if actual_cols_for_grouping and 'user_id' in actual_cols_for_grouping:
        for uid_grp, df_grp in tqdm(logs_in_max_window_for_all[actual_cols_for_grouping].groupby('user_id'), desc="Grouping for Others"):
            user_log_grouped_for_other_strategies[uid_grp] = df_grp.copy() 
    print(f"User logs grouped for other strategies: {len(user_log_grouped_for_other_strategies)} users.")
    del logs_in_max_window_for_all # 清理
    gc.collect()


    # 3. 确定最终目标用户
    num_test_users_config = global_recall_config.get("num_test_users_for_run_recall", 1000000) # 从配置读取
    
    # target_users 应基于 user_log_grouped_for_other_strategies 的键，因为所有策略都会用它
    # ItemCF 如果对某个用户没有预处理历史，它会在 get_candidates 中返回空
    all_available_users = list(user_log_grouped_for_other_strategies.keys())
    
    if num_test_users_config == -1:
        target_users = all_available_users
    elif len(all_available_users) > num_test_users_config:
        target_users = all_available_users[:num_test_users_config]
    else:
        target_users = all_available_users
    print(f"\nFinal target users for recall run: {len(target_users)}")
    if not target_users:
        print("No target users to process. Exiting.")
        exit(1)


    # 4. 预计算热门商品
    precomputed_pop_items_series = None
    pop_strategy_config_entry = main_config_data.get("strategies", {}).get("global_popular_items", {})
    if pop_strategy_config_entry.get("enabled", False):
        print("\nPre-calculating global popular items...")
        pop_params = pop_strategy_config_entry.get("params",{})
        pop_days_window = pop_params.get("days_window_for_popularity", 7)
        
        # 为热门商品计算准备数据 (需要完整的、仅按其自身窗口过滤的日志)
        df_logs_for_pop_calc_fresh = pd.read_parquet(user_log_path_abs) # 重新加载
        if 'datetime' in df_logs_for_pop_calc_fresh.columns and \
           not pd.api.types.is_datetime64_any_dtype(df_logs_for_pop_calc_fresh['datetime']):
            df_logs_for_pop_calc_fresh['datetime'] = pd.to_datetime(df_logs_for_pop_calc_fresh['datetime'])

        pop_effective_end_dt = pd.to_datetime(behavior_data_end_date_for_pp_str) + pd.Timedelta(hours=23, minutes=59, seconds=59)
        pop_start_dt = pop_effective_end_dt - pd.Timedelta(days=pop_days_window) + pd.Timedelta(seconds=1)
        
        df_logs_for_pop_calc_final = df_logs_for_pop_calc_fresh[
            (df_logs_for_pop_calc_fresh['datetime'] >= pop_start_dt) &
            (df_logs_for_pop_calc_fresh['datetime'] <= pop_effective_end_dt)
        ]
        del df_logs_for_pop_calc_fresh; gc.collect()

        precomputed_pop_items_series = GlobalPopularItemsStrategy.calculate_popular_items_logic(
            user_log_df_full=df_logs_for_pop_calc_final,
            item_pool_set_to_filter=item_pool_set_global, # 使用全局 item_pool_set
            behavior_data_end_date_str=behavior_data_end_date_for_pp_str, # calculate_popular_items_logic 内部会处理
            days_window=pop_days_window,
            min_interactions_threshold=pop_params.get("min_interactions_for_hot", 1),
            strategy_name_for_log="GlobalPopularItemsPreCalc"
        )
        if precomputed_pop_items_series is not None and not precomputed_pop_items_series.empty:
            print(f"Global popular items precomputed. Top item count: {precomputed_pop_items_series.iloc[0] if not precomputed_pop_items_series.empty else 'N/A'}")
        del df_logs_for_pop_calc_final; gc.collect()


    # 5. 获取激活的召回策略实例
    print("\nLoading recall strategies...")
    active_strategies = RecallStrategyFactory.get_active_strategies(
        CONFIG_FILE,
        user_log_grouped_for_other_strategies, # 给通用策略
        df_items_global,                       # 全局物品DF
        precomputed_popular_items=precomputed_pop_items_series,
        user_preprocessed_itemcf_histories=user_preprocessed_itemcf_histories # ItemCF专属历史
    )
    if not active_strategies:
        print("No active recall strategies were loaded. Exiting.")
        exit(1)
    print(f"Loaded {len(active_strategies)} active strategies.\n")
    # 工厂内部已经处理了将 user_preprocessed_itemcf_histories 注入ItemCF实例的逻辑 (需要factory.py配合修改)
    # 如果工厂没有特殊处理，可以在这里手动注入：
    # for strat in active_strategies:
    #     if isinstance(strat, ItemCFRecallStrategy):
    #         strat.user_specific_preprocessed_history = user_preprocessed_itemcf_histories
    #         print(f"Manually Injected preprocessed histories into ItemCF strategy for {len(user_preprocessed_itemcf_histories)} users.")


    # 6. 运行召回流程
    print("Starting recall process...")
    overall_start_time = time.time()
    final_candidates_df = run_recall_for_user_group(
        target_user_ids=target_users,
        active_strategies=active_strategies,
        global_config=global_recall_config
    )
    overall_end_time = time.time()
    total_duration = overall_end_time - overall_start_time
    print(f"\nRecall process finished. Total time: {total_duration:.2f}s")
    if not final_candidates_df.empty and target_users:
        avg_time_per_user = total_duration / len(target_users)
        print(f"Average time per user: {avg_time_per_user:.4f}s")

    # 7. 保存召回结果
    if not final_candidates_df.empty:
        recall_output_dir_abs = os.path.join(PROJECT_ROOT, global_recall_config.get("recall_output_path", "data/1_interim/"))
        os.makedirs(recall_output_dir_abs, exist_ok=True)
        pred_date_for_filename = global_recall_config.get("target_prediction_date", "YYYYMMDD").replace('-', '')
        output_filename = f"recall_candidates_for_{pred_date_for_filename}.parquet"
        output_file_path_abs = os.path.join(recall_output_dir_abs, output_filename)
        try:
            final_candidates_df.to_parquet(output_file_path_abs, index=False)
            print(f"Successfully saved {len(final_candidates_df)} recall candidates to: {output_file_path_abs}")
            print("\nSample of recalled candidates (first 5 rows):")
            print(final_candidates_df.head().to_string())
        except Exception as e:
            print(f"Error saving recall candidates to {output_file_path_abs}: {e}")
    else:
        print("No recall candidates were generated overall.")