import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class FeatureEngineering:
    def __init__(self, config):
        self.config = config
        
    def create_features(self, data):
        """创建特征"""
        user_behavior = data['user_behavior']
        items = data['items']
        
        # 用户特征
        user_features = self._create_user_features(user_behavior)
        
        # 商品特征
        item_features = self._create_item_features(user_behavior, items)
        
        # 用户-商品交互特征
        interaction_features = self._create_interaction_features(user_behavior, items)
        
        # 合并所有特征
        features = pd.merge(user_features, item_features, on=['user_id', 'item_id'], how='outer')
        features = pd.merge(features, interaction_features, on=['user_id', 'item_id'], how='outer')
        
        return features
    
    def _create_user_features(self, user_behavior):
        """创建用户特征"""
        user_features = []
        
        # 用户行为统计
        for window in self.config.time_windows:
            window_end = pd.to_datetime(self.config.train_end_date)
            window_start = window_end - timedelta(days=window)
            window_data = user_behavior[user_behavior['time'] >= window_start]
            
            # 计算用户行为统计
            user_stats = window_data.groupby('user_id').agg({
                'behavior_type': ['count', 'nunique'],
                'item_id': 'nunique',
                'item_category': 'nunique'
            }).reset_index()
            
            user_stats.columns = ['user_id', f'behavior_count_{window}d', 
                                f'behavior_type_count_{window}d',
                                f'item_count_{window}d',
                                f'category_count_{window}d']
            user_features.append(user_stats)
        
        # 合并所有时间窗口的特征
        user_features = pd.concat(user_features, axis=1)
        user_features = user_features.loc[:,~user_features.columns.duplicated()]
        
        return user_features
    
    def _create_item_features(self, user_behavior, items):
        """创建商品特征"""
        item_features = []
        
        # 商品行为统计
        for window in self.config.time_windows:
            window_end = pd.to_datetime(self.config.train_end_date)
            window_start = window_end - timedelta(days=window)
            window_data = user_behavior[user_behavior['time'] >= window_start]
            
            # 计算商品行为统计
            item_stats = window_data.groupby('item_id').agg({
                'behavior_type': ['count', 'nunique'],
                'user_id': 'nunique'
            }).reset_index()
            
            item_stats.columns = ['item_id', f'item_behavior_count_{window}d',
                                f'item_behavior_type_count_{window}d',
                                f'item_user_count_{window}d']
            item_features.append(item_stats)
        
        # 合并所有时间窗口的特征
        item_features = pd.concat(item_features, axis=1)
        item_features = item_features.loc[:,~item_features.columns.duplicated()]
        
        # 合并商品基本信息
        item_features = pd.merge(item_features, items[['item_id', 'item_category']], 
                               on='item_id', how='left')
        
        return item_features
    
    def _create_interaction_features(self, user_behavior, items):
        """创建用户-商品交互特征"""
        interaction_features = []
        
        # 用户-商品交互统计
        for window in self.config.time_windows:
            window_end = pd.to_datetime(self.config.train_end_date)
            window_start = window_end - timedelta(days=window)
            window_data = user_behavior[user_behavior['time'] >= window_start]
            
            # 计算用户-商品交互统计
            interaction_stats = window_data.groupby(['user_id', 'item_id']).agg({
                'behavior_type': ['count', 'nunique'],
                'time': ['min', 'max']
            }).reset_index()
            
            interaction_stats.columns = ['user_id', 'item_id', 
                                      f'interaction_count_{window}d',
                                      f'interaction_type_count_{window}d',
                                      f'first_interaction_{window}d',
                                      f'last_interaction_{window}d']
            interaction_features.append(interaction_stats)
        
        # 合并所有时间窗口的特征
        interaction_features = pd.concat(interaction_features, axis=1)
        interaction_features = interaction_features.loc[:,~interaction_features.columns.duplicated()]
        
        return interaction_features 