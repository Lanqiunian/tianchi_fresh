# src/rank/predictor.py
import pandas as pd
import os
import joblib
import numpy as np
import time
from datetime import datetime

from ..utils.config_loader import load_config
from .utils_rank import find_best_f1_threshold, calculate_f1_metrics # F1相关
from .model_ensembler import ModelEnsembler # 用于加载和使用融合模型

class Predictor:
    def __init__(self, project_root: str, rank_config_path: str):
        self.project_root = project_root
        self.rank_config = load_config(rank_config_path)

        self.global_conf = self.rank_config.get("global_settings", {})
        self.pred_conf = self.rank_config.get("prediction", {})
        self.ensemble_conf = self.rank_config.get("model_ensemble", {})

        self.processed_sample_dir = os.path.join(self.project_root, self.global_conf["processed_sample_path"])
        self.model_output_dir = os.path.join(self.project_root, self.global_conf["model_output_path"])
        self.submission_output_dir = os.path.join(self.project_root, self.global_conf["submission_output_path"])
        os.makedirs(self.submission_output_dir, exist_ok=True)

        self.model = None
        self.features_to_use = None
        self.best_f1_threshold = None # 从文件加载或在验证集上重新计算

    def _load_single_model(self, model_identifier: str = "main_v1"): # 或 "final_submission_model"
        """加载单个LGBM模型及其相关文件。"""
        model_filename = f"lgbm_model_{model_identifier}.pkl"
        model_path = os.path.join(self.model_output_dir, model_filename)
        feature_list_filename = f"lgbm_features_{model_identifier}.txt"
        feature_list_path = os.path.join(self.model_output_dir, feature_list_filename)
        threshold_filename = f"best_f1_threshold_{model_identifier}.txt"
        threshold_path = os.path.join(self.model_output_dir, threshold_filename)

        if not os.path.exists(model_path) or not os.path.exists(feature_list_path):
            raise FileNotFoundError(f"模型文件 ({model_path}) 或特征列表文件 ({feature_list_path}) 未找到。")

        print(f"  加载模型从: {model_path}")
        self.model = joblib.load(model_path)
        with open(feature_list_path, 'r') as f:
            self.features_to_use = [line.strip() for line in f if line.strip()]
        print(f"  加载特征列表 ({len(self.features_to_use)}个特征)。")

        if os.path.exists(threshold_path):
            with open(threshold_path, 'r') as f:
                self.best_f1_threshold = float(f.read().strip())
            print(f"  加载已保存的最佳F1阈值: {self.best_f1_threshold:.4f}")
        else:
            print(f"  警告: 未找到已保存的最佳F1阈值文件 ({threshold_path})。可能需要在验证集上重新优化。")
            self.best_f1_threshold = 0.1 # 一个备用默认值，强烈建议优化

    def _load_ensemble_model(self):
        """加载融合模型 (通过 ModelEnsembler)。"""
        print("  加载融合模型...")
        self.model_ensemble = ModelEnsembler(self.project_root,
                                             os.path.join(self.project_root, 'conf', 'rank', 'rank_config.yaml'))
        # ModelEnsembler 的 _load_bagging_models_if_empty 会尝试加载
        self.model_ensemble._load_bagging_models_if_empty()
        if not self.model_ensemble.models:
            raise ValueError("无法加载融合模型，或融合模型中不包含任何基模型。")
        
        # 对于融合模型，特征列表通常基于第一个基模型（假设一致）
        self.features_to_use = self.model_ensemble.feature_lists[0]
        print(f"  融合模型加载完成，使用 {len(self.features_to_use)} 个特征。")
        
        # 融合模型的阈值通常需要在验证集上对融合后的概率进行优化
        # 这里简单地使用第一个基模型的阈值，或者重新优化
        if self.model_ensemble.best_thresholds:
            self.best_f1_threshold = np.mean(self.model_ensemble.best_thresholds) # 或者取第一个，或者重新优化
            print(f"  使用融合模型的（平均/首个）最佳F1阈值: {self.best_f1_threshold:.4f}")
        else:
            self.best_f1_threshold = 0.1
            print(f"  警告: 未找到融合模型的最佳F1阈值。使用默认值 {self.best_f1_threshold:.4f}。")


    def predict_on_test_set(self, test_set_key: str = "test_target", model_id_for_single: str = "final_submission_model"):
        """
        在测试集上进行预测并生成提交文件。
        """
        print(f"\n开始在测试集 '{test_set_key}' 上进行预测...")

        # 加载模型（单个或融合）
        if self.ensemble_conf.get("enabled", False):
            self._load_ensemble_model()
        else:
            self._load_single_model(model_identifier=model_id_for_single) # 使用为提交训练的模型ID

        if self.model is None and (self.model_ensemble is None or not self.model_ensemble.models):
            print("错误: 模型未能加载。无法进行预测。")
            return

        # 加载测试集样本数据
        test_sample_file = os.path.join(self.processed_sample_dir, f"{test_set_key}_samples.parquet")
        if not os.path.exists(test_sample_file):
            raise FileNotFoundError(f"测试样本文件未找到: {test_sample_file}。请先运行 data_builder_for_rank.py。")

        print(f"  加载测试样本数据从: {test_sample_file}")
        df_test = pd.read_parquet(test_sample_file)
        print(f"    测试样本形状: {df_test.shape}")

        if df_test.empty:
            print("警告: 测试样本数据为空。无法生成预测。")
            return

        # 准备特征
        missing_features = set(self.features_to_use) - set(df_test.columns)
        if missing_features:
            raise ValueError(f"测试数据中缺少以下必要的特征: {missing_features}")
        X_test = df_test[self.features_to_use]

        # 进行概率预测
        print("  进行概率预测...")
        if self.ensemble_conf.get("enabled", False) and self.model_ensemble and self.model_ensemble.models:
            y_pred_proba_test = self.model_ensemble.predict_proba(X_test)
        elif self.model:
            y_pred_proba_test = self.model.predict_proba(X_test)[:, 1]
        else:
            print("错误: 无法确定使用哪个模型进行预测。")
            return

        df_test['pred_proba'] = y_pred_proba_test
        print(f"    预测概率 (前5个): {df_test['pred_proba'].head().values}")

        # 如果没有预设的阈值，或者想要在验证集（如果有标签的话）上重新优化
        # 对于最终提交，通常使用在验证集上找到的最佳阈值
        # 这里假设 self.best_f1_threshold 已经是最终阈值
        final_threshold_to_use = self.best_f1_threshold
        if final_threshold_to_use is None:
            # 这种情况不应该发生，因为 _load_model 会设置一个默认值或从文件加载
            # 如果真的需要，可以尝试在某个验证集上临时找一个，但这不规范
            print("警告: 未找到最终F1阈值，将使用默认值0.1。强烈建议进行阈值优化。")
            final_threshold_to_use = 0.1


        print(f"  使用最终阈值 {final_threshold_to_use:.4f} 生成提交结果...")
        # 根据阈值进行二分类
        df_test['prediction'] = (df_test['pred_proba'] >= final_threshold_to_use).astype(int)

        # 筛选出预测为1的 (user_id, item_id) 对
        submission_df = df_test[df_test['prediction'] == 1][['user_id', 'item_id']]
        print(f"  预测购买的 (user, item) 对数量: {len(submission_df)}")

        if submission_df.empty:
            print("警告: 没有预测任何购买行为。生成的提交文件将为空。")

        # 生成提交文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pred_date_for_filename = self.rank_config.get("dataset_config",{}).get(test_set_key,{}).get("prediction_date","").replace("-","")
        submission_filename_base = self.pred_conf.get("submission_file_prefix", "submission")
        submission_filename = f"{submission_filename_base}_{pred_date_for_filename}_{timestamp}.txt"
        submission_path = os.path.join(self.submission_output_dir, submission_filename)

        submission_df.to_csv(submission_path, sep='\t', header=False, index=False)
        print(f"  提交文件已保存到: {submission_path}")
        print(f"  提交文件内容 (前5行):\n{submission_df.head().to_string(index=False)}")

        return submission_path


if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    RANK_CONFIG = os.path.join(PROJECT_ROOT, 'conf', 'rank', 'rank_config.yaml')

    predictor = Predictor(project_root=PROJECT_ROOT, rank_config_path=RANK_CONFIG)

    try:
        # 假设 test_target 样本已通过 data_builder_for_rank.py 生成
        # 并且模型 (单个或融合的) 已通过 lgbm_trainer.py 或 model_ensembler.py 训练并保存
        # model_id_for_single 应与你训练最终提交模型时使用的标识符一致
        # 例如，如果你用 "train_full" 数据集训练了一个名为 "final_model_v1" 的模型：
        submission_file = predictor.predict_on_test_set(test_set_key="test_target", model_id_for_single="final_submission_model")
        # 如果你使用了ensemble，并且ensemble配置中enabled=true, predictor会自动尝试加载ensemble
        # predictor.predict_on_test_set(test_set_key="test_target")

        print(f"\nPredictor.predict_on_test_set() 执行完毕。提交文件: {submission_file}")

    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请确保已运行 src/rank/data_builder_for_rank.py 生成测试样本，并已训练和保存了模型。")
    except ValueError as e:
        print(f"错误: {e}")
    except Exception as e:
        import traceback
        print(f"预测过程中发生未知错误: {e}")
        traceback.print_exc()