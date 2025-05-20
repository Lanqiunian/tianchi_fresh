# src/recall/strategies/itemcf_strategy.py
import pandas as pd
from collections import defaultdict
import os
import pickle
import heapq # 确保导入

from src.recall.recall_strategy import RecallStrategy

class ItemCFRecallStrategy(RecallStrategy):
    def __init__(self, strategy_name: str, processed_data_path: str,
                 strategy_specific_config: dict,
                 user_log_grouped: dict = None, # 这个参数基类会接收，但ItemCF不直接用
                 items_df_global: pd.DataFrame = None,
                 # 新增：接收专属的预处理历史 {user_id: [item_id1, item_id2, ...]}
                 user_specific_preprocessed_history: dict = None):
        
        super().__init__(strategy_name, processed_data_path, strategy_specific_config,
                         user_log_grouped, items_df_global) # 调用基类初始化
        
        # ItemCF 自身的参数
        self.top_k_similar_items = self.config.get("top_k_similar_items", 10)
        # days_window_user_history 和 max_user_history_items 现在主要由 run_recall.py 中的预处理逻辑使用
        
        similarity_matrix_filename = self.config.get("similarity_matrix_filename", f"{self.strategy_name}_similarity.pkl")
        self.item_similarity_path = os.path.join(self.processed_data_path, similarity_matrix_filename)
        self.item_similarity_dict = self._load_similarity_matrix()

        # 存储由 run_recall.py 注入的预处理历史
        self.user_specific_preprocessed_history = user_specific_preprocessed_history if user_specific_preprocessed_history is not None else {}
        if self.user_specific_preprocessed_history:
             print(f"Strategy '{self.strategy_name}': Received preprocessed user histories for {len(self.user_specific_preprocessed_history)} users.")


    def _load_similarity_matrix(self) -> dict:
        if os.path.exists(self.item_similarity_path):
            try:
                with open(self.item_similarity_path, 'rb') as f:
                    sim_dict = pickle.load(f)
                    print(f"Strategy '{self.strategy_name}': Successfully loaded item similarity. Items with sims: {len(sim_dict)}")
                    return sim_dict
            except Exception as e:
                print(f"Strategy '{self.strategy_name}': CRITICAL - Error loading similarity matrix from {self.item_similarity_path}. Error: {e}.")
                return {}
        else:
            print(f"Strategy '{self.strategy_name}': CRITICAL - Precomputed item similarity matrix not found at {self.item_similarity_path}. ItemCF will not work. Please run the offline similarity calculation script first.")
            return {}

    def get_candidates(self, user_id: int, behavior_data_end_date_str: str, N: int) -> pd.DataFrame:
        # behavior_data_end_date_str 在此策略中不再直接使用，因为历史已经按需预处理
        
        if not self.item_similarity_dict:
            return pd.DataFrame(columns=['user_id', 'item_id', 'recall_score', 'recall_source'])

        # ---- 直接从预处理历史中获取用户的交互物品列表 ----
        # user_interacted_items 是一个已经排序、去重、截断的 item_id 列表
        user_interacted_items = self.user_specific_preprocessed_history.get(user_id, []) 
        
        if not user_interacted_items:
            return pd.DataFrame(columns=['user_id', 'item_id', 'recall_score', 'recall_source'])
        
        user_interacted_set = set(user_interacted_items)

        candidate_item_scores = defaultdict(float)
        for interacted_item_id in user_interacted_items: # 循环次数已由预处理的 max_user_history_items 控制
            if interacted_item_id not in self.item_similarity_dict:
                continue 
            
            # similar_items_for_interacted 是一个已截断(offline_similarity_top_n)并排序的字典
            similar_items_for_interacted = self.item_similarity_dict[interacted_item_id] 
            
            count = 0
            # 内层循环次数受 offline_similarity_top_n 和 top_k_similar_items 双重控制
            for similar_item_id, similarity_score in similar_items_for_interacted.items(): 
                if count >= self.top_k_similar_items: 
                    break
                if similar_item_id not in user_interacted_set: 
                    # 相似物品本身应该已经在P中 (因为相似度矩阵是基于P中物品计算的)
                    # if self.item_pool_set and similar_item_id not in self.item_pool_set:
                    #     continue
                    candidate_item_scores[similar_item_id] += similarity_score
                count += 1
        
        if not candidate_item_scores:
            return pd.DataFrame(columns=['user_id', 'item_id', 'recall_score', 'recall_source'])

        # 使用 heapq.nlargest 获取Top-N候选
        top_n_candidates_list = heapq.nlargest(N, candidate_item_scores.items(), key=lambda item: item[1])
        
        if not top_n_candidates_list:
             return pd.DataFrame(columns=['user_id', 'item_id', 'recall_score', 'recall_source'])

        recalled_items_data = [{'item_id': item_id, 'recall_score': score} for item_id, score in top_n_candidates_list]
        output_df = pd.DataFrame(recalled_items_data)

        if output_df.empty:
            return pd.DataFrame(columns=['user_id', 'item_id', 'recall_score', 'recall_source'])

        output_df['user_id'] = user_id
        output_df['recall_source'] = self.strategy_name
        return output_df[['user_id', 'item_id', 'recall_score', 'recall_source']]