import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.model = None
        
    def train_and_predict(self, features):
        """训练模型并预测"""
        # 准备训练数据
        X = features.drop(['user_id', 'item_id', 'label'], axis=1, errors='ignore')
        y = features['label'] if 'label' in features.columns else None
        
        # 划分训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 训练模型
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        self.model = lgb.train(
            self.config.model_params['lgb'],
            train_data,
            valid_sets=[train_data, val_data],
            num_boost_round=1000,
            early_stopping_rounds=50,
            verbose_eval=100
        )
        
        # 预测
        predictions = self.model.predict(X)
        
        # 生成提交结果
        submission = pd.DataFrame({
            'user_id': features['user_id'],
            'item_id': features['item_id'],
            'score': predictions
        })
        
        # 选择每个用户得分最高的N个商品
        submission = submission.sort_values(['user_id', 'score'], ascending=[True, False])
        submission = submission.groupby('user_id').head(10)  # 每个用户推荐10个商品
        
        return submission[['user_id', 'item_id']]
    
    def evaluate(self, y_true, y_pred):
        """评估模型性能"""
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        } 