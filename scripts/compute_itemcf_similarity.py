# scripts/compute_itemcf_similarity.py
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity # type: ignore
from scipy.sparse import csr_matrix # type: ignore
import os
import sys
import pickle
import yaml
from tqdm import tqdm # type: ignore

# 将项目根目录添加到sys.path
PROJECT_ROOT_FOR_SCRIPT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT_FOR_SCRIPT)

def calculate_and_save_item_similarity(
    user_log_path: str,
    items_path: str,
    output_similarity_path: str,
    behavior_types_for_similarity: list = None,
    offline_similarity_top_n: int = 100, # 新增参数：离线截断数量
    item_col='item_id', 
    user_col='user_id',
    behavior_col='behavior_type'
    ):
    print(f"Loading user log from: {user_log_path}")
    user_log_df = pd.read_parquet(user_log_path)
    print(f"Loading items from: {items_path}")
    items_df = pd.read_parquet(items_path)
    
    item_pool_set = set(items_df[item_col].unique())
    print(f"Item pool size: {len(item_pool_set)}")

    if 'datetime' in user_log_df.columns and not pd.api.types.is_datetime64_any_dtype(user_log_df['datetime']):
        user_log_df['datetime'] = pd.to_datetime(user_log_df['datetime'])

    similarity_logs = user_log_df.copy() # 使用副本
    del user_log_df # 尽早释放内存
    import gc
    gc.collect()

    if behavior_types_for_similarity:
        valid_behavior_types = [int(bt) for bt in behavior_types_for_similarity]
        similarity_logs = similarity_logs[similarity_logs[behavior_col].isin(valid_behavior_types)]
    
    if similarity_logs.empty:
        print(f"No logs found for specified behavior types. Cannot compute similarity.")
        return
    print(f"Number of logs for similarity computation: {len(similarity_logs)}")

    user_item_interaction = similarity_logs[[user_col, item_col]].drop_duplicates()
    del similarity_logs # 释放内存
    gc.collect()

    user_item_interaction = user_item_interaction[user_item_interaction[item_col].isin(item_pool_set)]
    
    if user_item_interaction.empty:
        print(f"No user-item interactions found within the item pool. Cannot compute similarity.")
        return
    print(f"Number of unique user-item interactions in P for similarity: {len(user_item_interaction)}")

    user_ids_all = user_item_interaction[user_col].unique()
    item_ids_all = user_item_interaction[item_col].unique()

    if len(item_ids_all) < 2 or len(user_ids_all) == 0:
        print(f"Not enough unique items in P ({len(item_ids_all)}) or users ({len(user_ids_all)}) for similarity.")
        return

    user_id_map = {uid: i for i, uid in enumerate(user_ids_all)}
    item_id_map = {iid: i for i, iid in enumerate(item_ids_all)}
    item_idx_to_id_map = {i: iid for iid, i in item_id_map.items()}
    
    del user_ids_all # 释放内存
    # item_ids_all 保留，或者 item_idx_to_id_map 已经包含了所有信息
    gc.collect()


    rows = user_item_interaction[user_col].map(user_id_map)
    cols = user_item_interaction[item_col].map(item_id_map)
    data = np.ones(len(user_item_interaction))
    del user_item_interaction # 释放内存
    gc.collect()

    user_item_matrix_sparse = csr_matrix((data, (rows, cols)), shape=(len(user_id_map), len(item_id_map)))
    del rows, cols, data, user_id_map # item_id_map 还需要用于 item_idx_to_id_map
    gc.collect()

    item_user_matrix_sparse = user_item_matrix_sparse.T
    del user_item_matrix_sparse
    gc.collect()
    
    print("Computing cosine similarity...")
    item_similarity_matrix_sparse = cosine_similarity(item_user_matrix_sparse, dense_output=False)
    del item_user_matrix_sparse
    gc.collect()
    print(f"Item similarity matrix computed (shape: {item_similarity_matrix_sparse.shape}). Converting to dict...")

    item_similarity_final_dict = {}
    num_items = item_similarity_matrix_sparse.shape[0] # 这应该是 len(item_id_map)

    for i in tqdm(range(num_items), desc="Building similarity dictionary"):
        source_item_id = item_idx_to_id_map.get(i)
        if source_item_id is None: continue

        row_slice = item_similarity_matrix_sparse[i]
        similar_indices = row_slice.nonzero()[1]
        similarity_scores = row_slice.data
        
        current_item_sims = {}
        for j_idx, score in zip(similar_indices, similarity_scores):
            if i == j_idx or score <= 1e-6:
                continue
            target_item_id = item_idx_to_id_map.get(j_idx)
            if target_item_id is None: continue
            current_item_sims[target_item_id] = score
        
        if current_item_sims:
            sorted_sims_list = sorted(current_item_sims.items(), key=lambda item: item[1], reverse=True)
            # ---- 重要修改：离线截断 ----
            item_similarity_final_dict[source_item_id] = dict(sorted_sims_list[:offline_similarity_top_n])
    
    print(f"Item similarity dictionary created. Number of items with similarities: {len(item_similarity_final_dict)}")
    
    try:
        os.makedirs(os.path.dirname(output_similarity_path), exist_ok=True)
        with open(output_similarity_path, 'wb') as f:
            pickle.dump(item_similarity_final_dict, f)
        print(f"Successfully saved item similarity dictionary to: {output_similarity_path}")
    except Exception as e:
        print(f"Error saving item similarity dictionary: {e}")

if __name__ == "__main__":
    config_path = os.path.join(PROJECT_ROOT_FOR_SCRIPT, 'conf', 'recall', 'recall_config.yaml')
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        sys.exit(1)
        
    with open(config_path, 'r') as f:
        recall_config = yaml.safe_load(f)

    processed_data_path_cfg = recall_config.get("global_settings", {}).get("processed_data_path", "data/1_interim/")
    processed_data_dir = os.path.join(PROJECT_ROOT_FOR_SCRIPT, processed_data_path_cfg)

    user_log_file = os.path.join(processed_data_dir, "processed_user_log_p_related.parquet")
    items_file = os.path.join(processed_data_dir, "processed_items.parquet")

    itemcf_params_cfg = recall_config.get("strategies", {}).get("item_cf", {}).get("params", {})
    output_filename = itemcf_params_cfg.get("similarity_matrix_filename", "itemcf_similarity_default.pkl")
    output_sim_path = os.path.join(processed_data_dir, output_filename)
    
    behav_types_sim_cfg = itemcf_params_cfg.get("similarity_behavior_types", None)
    offline_top_n_cfg = itemcf_params_cfg.get("offline_similarity_top_n", 100) # 从配置读取

    print("--- Starting ItemCF Similarity Matrix Computation ---")
    print(f"User Log: {user_log_file}")
    print(f"Items File: {items_file}")
    print(f"Output Path: {output_sim_path}")
    print(f"Behavior types for similarity: {behav_types_sim_cfg if behav_types_sim_cfg else 'All'}")
    print(f"Offline Top-N similar items per item: {offline_top_n_cfg}")


    calculate_and_save_item_similarity(
        user_log_path=user_log_file,
        items_path=items_file,
        output_similarity_path=output_sim_path,
        behavior_types_for_similarity=behav_types_sim_cfg,
        offline_similarity_top_n=offline_top_n_cfg # 传递给函数
    )
    print("--- ItemCF Similarity Matrix Computation Finished ---")