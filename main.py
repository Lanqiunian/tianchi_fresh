import os
import pandas as pd
from src.data_processing.processor import DataProcessor
from src.feature_engineering.features import FeatureEngineering
from src.models.trainer import ModelTrainer
from src.utils.config import Config

def main():
    # 配置参数
    config = Config()
    
    # 数据预处理
    data_processor = DataProcessor(config)
    user_behavior_df, item_df = data_processor.load_data()
    processed_data = data_processor.preprocess(user_behavior_df, item_df)
    
    # 特征工程
    feature_engineering = FeatureEngineering(config)
    features = feature_engineering.create_features(processed_data)
    
    # 模型训练和预测
    model_trainer = ModelTrainer(config)
    predictions = model_trainer.train_and_predict(features)
    
    # 保存结果
    predictions.to_csv('submission.csv', index=False, sep='\t')

if __name__ == "__main__":
    main()