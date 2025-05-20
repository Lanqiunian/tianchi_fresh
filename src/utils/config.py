class Config:
    def __init__(self):
        # 数据路径
        self.user_behavior_path_partA = "dataset/tianchi_fresh_comp_train_user_online_partA.txt"
        self.user_behavior_path_partB = "dataset/tianchi_fresh_comp_train_user_online_partB.txt"
        self.item_path = "dataset/tianchi_fresh_comp_train_item_online.txt"
        
        # 时间相关参数
        self.train_start_date = "2014-11-18"
        self.train_end_date = "2014-12-18"
        self.predict_date = "2014-12-19"
        
        # 特征工程参数
        self.time_windows = [1, 3, 7, 14, 30]  # 时间窗口（天）
        self.geohash_precision = 6  # 地理位置哈希精度
        
        # 模型参数
        self.model_params = {
            "lgb": {
                "objective": "binary",
                "metric": "auc",
                "boosting_type": "gbdt",
                "num_leaves": 31,
                "learning_rate": 0.05,
                "feature_fraction": 0.9
            }
        }
        
        # 评估参数
        self.eval_metrics = ["precision", "recall", "f1"] 