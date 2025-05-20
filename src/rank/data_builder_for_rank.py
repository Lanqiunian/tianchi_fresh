# src/rank/data_builder_for_rank.py
import pandas as pd
import os
import yaml
from ..utils.config_loader import load_config # 假设已存在
from .utils_rank import get_label # 从同级目录导入

def build_rank_samples(
    project_root: str,
    rank_config_path: str,
    dataset_key: str # 例如 "train_s1", "valid_s1", "test_target"
):
    """
    构建用于排序模型训练、验证或测试的样本数据。
    它会加载特征工程的结果，并（如果不是测试集）关联真实标签。
    """
    print(f"\n开始为数据集 '{dataset_key}' 构建排序样本...")

    rank_config = load_config(rank_config_path)
    global_conf = rank_config.get("global_settings", {})
    
    # 修改开始
    all_dataset_configs = rank_config.get("dataset_config", {})
    if dataset_key in ["train_s1", "valid_s1"]: # 检查是否是嵌套在 train_valid_split 下的键
        dataset_conf = all_dataset_configs.get("train_valid_split", {}).get(dataset_key)
    else: # 对于 train_full, test_target 等直接在 dataset_config 下的键
        dataset_conf = all_dataset_configs.get(dataset_key)
    # 修改结束

    if not dataset_conf:
        raise ValueError(f"在 rank_config.yaml 的 dataset_config 或 dataset_config.train_valid_split 下未找到数据集 '{dataset_key}' 的配置。")
    feature_data_dir = os.path.join(project_root, global_conf["feature_data_path"])
    processed_sample_dir = os.path.join(project_root, global_conf["processed_sample_path"])
    user_log_full_path = os.path.join(project_root, global_conf["user_log_path"])
    os.makedirs(processed_sample_dir, exist_ok=True)

    # 构造特征文件名
    # 假设特征文件名格式为 ranking_features_manual_{feature_file_suffix}.parquet
    feature_filename = f"ranking_features_manual_{dataset_conf['feature_file_suffix']}.parquet"
    feature_file_path = os.path.join(feature_data_dir, feature_filename)

    if not os.path.exists(feature_file_path):
        raise FileNotFoundError(f"特征文件未找到: {feature_file_path}。请先运行特征工程。")

    print(f"  加载特征数据从: {feature_file_path}")
    features_df = pd.read_parquet(feature_file_path)
    print(f"    特征数据形状: {features_df.shape}")

    if features_df.empty:
        print(f"警告: 为 '{dataset_key}' 加载的特征数据为空。")
        # 保存一个空的parquet文件以避免后续流程出错，或者直接报错
        output_sample_filename = f"{dataset_key}_samples.parquet"
        output_sample_path = os.path.join(processed_sample_dir, output_sample_filename)
        features_df.to_parquet(output_sample_path, index=False)
        print(f"  已保存空的样本文件到: {output_sample_path}")
        return output_sample_path


    # 如果不是最终的测试集 (test_target 通常没有真实标签可获取)
    # 并且 prediction_date 在配置中存在，则获取标签
    # 对于天池比赛的测试集，我们没有未来的标签
    final_samples_df = features_df.copy()
    if dataset_key != "test_target" and "prediction_date" in dataset_conf:
        prediction_date = dataset_conf["prediction_date"]
        print(f"  为 '{dataset_key}' (预测日期: {prediction_date}) 生成标签...")
        try:
            user_log_df = pd.read_parquet(user_log_full_path)
            if 'datetime' not in user_log_df.columns and 'time' in user_log_df.columns:
                user_log_df['datetime'] = pd.to_datetime(user_log_df['time'])
            elif 'datetime' in user_log_df.columns and not pd.api.types.is_datetime64_any_dtype(user_log_df['datetime']):
                user_log_df['datetime'] = pd.to_datetime(user_log_df['datetime'])

            final_samples_df = get_label(user_log_df, features_df, prediction_date)
        except FileNotFoundError:
            print(f"错误: 用户行为日志文件未找到于 {user_log_full_path}。无法生成标签。")
            # 根据策略，可以选择是否继续（如果允许无标签数据），或者直接报错
            # 对于训练/验证，标签是必须的
            raise
        except Exception as e:
            print(f"生成标签时发生错误: {e}")
            raise
    elif dataset_key == "test_target":
        print(f"  数据集 '{dataset_key}' 是测试集，不生成标签。")
        # 为测试集添加一个全为0的 'label' 列，以便后续代码统一处理，但在模型评估时不会使用
        final_samples_df['label'] = 0 # 或者一个特殊值如 -1
    else:
        print(f"警告: 数据集 '{dataset_key}' 的配置中缺少 'prediction_date'，无法生成标签。")
        # 同样，可以添加一个全为0的 'label' 列
        final_samples_df['label'] = -1 # 特殊值表示未知标签


    # 保存处理后的样本数据
    output_sample_filename = f"{dataset_key}_samples.parquet"
    output_sample_path = os.path.join(processed_sample_dir, output_sample_filename)
    final_samples_df.to_parquet(output_sample_path, index=False)
    print(f"  排序样本已构建并保存到: {output_sample_path}")
    print(f"    最终样本形状: {final_samples_df.shape}")
    if 'label' in final_samples_df.columns:
        print(f"    标签分布 (如果适用):\n{final_samples_df['label'].value_counts(normalize=True, dropna=False)}")

    return output_sample_path


if __name__ == "__main__":
    # 假设此脚本在 src/rank/ 目录下
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    RANK_CONFIG = os.path.join(PROJECT_ROOT, 'conf', 'rank', 'rank_config.yaml')

    # 示例：构建训练集、验证集和测试集样本
    # 确保你的 rank_config.yaml 中的 feature_file_suffix 与你实际的特征文件名匹配
    # datasets_to_build = ["train_s1", "valid_s1", "train_full", "test_target"]
    datasets_to_build = ["train_s1", "valid_s1"]
    # datasets_to_build = ["test_target"] # 只构建测试集

    for ds_key in datasets_to_build:
        try:
            build_rank_samples(PROJECT_ROOT, RANK_CONFIG, ds_key)
        except Exception as e:
            print(f"构建数据集 '{ds_key}' 时失败: {e}")
            # 可以选择继续构建其他数据集或中断
            # break