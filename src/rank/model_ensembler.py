# src/rank/model_ensembler.py
import pandas as pd
import os
import yaml
import joblib
import numpy as np
from ..utils.config_loader import load_config
from .lgbm_trainer import LGBMTrainer # 用于训练多个模型

class ModelEnsembler:
    def __init__(self, project_root: str, rank_config_path: str):
        self.project_root = project_root
        self.rank_config = load_config(rank_config_path)
        self.ensemble_conf = self.rank_config.get("model_ensemble", {})
        self.model_output_dir = os.path.join(self.project_root,
                                             self.rank_config.get("global_settings", {})["model_output_path"])
        self.models = []
        self.feature_lists = [] # 每个模型对应的特征列表
        self.best_thresholds = [] # 每个模型对应的最佳阈值

    def train_bagging_models(self, train_key: str, valid_key: str, use_optuna_for_first_only: bool = True):
        """
        训练多个LGBM模型用于Bagging。
        """
        if not self.ensemble_conf.get("enabled") or \
           self.ensemble_conf.get("method") != "bagging_average":
            print("  模型Bagging未启用或方法不是 bagging_average。跳过训练Bagging模型。")
            return

        bagging_params = self.ensemble_conf.get("bagging_params", {})
        n_models = bagging_params.get("n_models", 1) # 如果n_models为1，则与单模型效果一样
        # seeds = bagging_params.get("random_seeds", [i * 100 for i in range(n_models)])
        # if len(seeds) < n_models:
        #     seeds.extend([ (i + len(seeds)) * 100 for i in range(n_models - len(seeds))])


        print(f"\n开始训练 {n_models} 个LGBM模型用于Bagging平均...")

        for i in range(n_models):
            model_id = f"bagging_model_{i+1}"
            print(f"\n  训练模型: {model_id}")
            # 修改LGBM配置中的随机种子
            # 注意：这需要LGBMTrainer能够接收并使用变化的种子，或者修改配置文件副本
            # 一个简单的方法是，LGBMTrainer 的 base_params.seed 可以被外部覆盖
            # 或者，我们可以为每个trainer创建一个略微不同的配置文件副本或修改参数
            current_seed = self.rank_config.get("lgbm_training",{}).get("base_params",{}).get("seed", 2024) + i * 100 # 生成不同种子

            # 创建一个新的配置字典，只修改种子
            temp_rank_config = self.rank_config.copy() # 深拷贝可能更好，但这里简单处理
            if "lgbm_training" not in temp_rank_config: temp_rank_config["lgbm_training"] = {}
            if "base_params" not in temp_rank_config["lgbm_training"]: temp_rank_config["lgbm_training"]["base_params"] = {}
            temp_rank_config["lgbm_training"]["base_params"]["seed"] = current_seed
            # 可以将修改后的配置保存到临时文件，或者直接传递字典（如果LGBMTrainer支持）
            # 为了简单，我们假设LGBMTrainer可以被实例化多次，并且其内部会重新加载配置或我们修改其属性
            # 更优雅的方式是让LGBMTrainer的train方法能接收覆盖参数。
            # 这里我们简单地为每个模型重新实例化Trainer，并传递不同的model_identifier
            # 并假设LGBMTrainer会使用其model_identifier来保存不同的模型文件

            # 写入临时配置文件，或者LGBMTrainer支持直接传入params_override
            temp_conf_path = os.path.join(self.project_root, "conf", "rank", f"temp_rank_conf_bag_{i}.yaml")
            with open(temp_conf_path, 'w') as f_temp:
                 yaml.dump(temp_rank_config, f_temp)


            trainer = LGBMTrainer(project_root=self.project_root,
                                  rank_config_path=temp_conf_path, # 使用临时配置
                                  model_identifier=model_id)
            
            # Optuna只对第一个模型运行（如果配置了），后续模型使用找到的最佳参数+不同种子
            use_optuna_current = self.rank_config.get("lgbm_training",{}).get("optuna_tuning",{}).get("enabled",False)
            if i > 0 and use_optuna_for_first_only:
                use_optuna_current = False # 后续模型不再进行Optuna调优
                # 如果第一个模型调优了，后续模型应该使用调优后的参数
                if hasattr(self, 'best_optuna_params_for_bagging') and self.best_optuna_params_for_bagging:
                    print(f"    模型 {model_id} 使用之前Optuna找到的最佳参数: {self.best_optuna_params_for_bagging}")
                    # 需要一种方式将这些参数传递给LGBMTrainer或覆盖其配置
                    # 这里简化：假设LGBMTrainer的配置中base_params会被 Optuna 的结果覆盖
                    # 但这里我们是为每个模型创建新的trainer和新的（临时的）config...
                    # 一个更实际的做法是在第一次Optuna后保存最佳参数，然后修改后续模型的base_params
                    # 为此，LGBMTrainer.train需要能返回最佳参数，或者我们从study对象获取

            model, features, threshold = trainer.train(train_key=train_key, valid_key=valid_key, use_optuna=use_optuna_current)
            
            if i == 0 and use_optuna_current: # 保存第一次Optuna的结果供后续使用
                # 这个逻辑需要调整，因为study对象在trainer内部
                # 暂时简化，假设我们手动将第一次的最佳参数更新到主配置文件或一个共享状态
                pass


            self.models.append(model) # 保存的是训练好的模型对象
            self.feature_lists.append(features)
            self.best_thresholds.append(threshold) # 每个模型在验证集上的最佳阈值

            os.remove(temp_conf_path) # 清理临时配置文件

        if self.models:
            print(f"\n成功训练 {len(self.models)} 个模型用于Bagging。")
            # 可以考虑保存一个ensemble的配置文件或元数据
            ensemble_meta = {
                "num_models": len(self.models),
                "model_identifiers": [f"bagging_model_{j+1}" for j in range(len(self.models))],
                "feature_list_一致性检查": "需要确保所有模型的特征列表一致才能简单平均"
            }
            meta_path = os.path.join(self.model_output_dir, "ensemble_bagging_meta.json")
            with open(meta_path, 'w') as f_meta:
                import json
                json.dump(ensemble_meta, f_meta, indent=2)
            print(f"  Bagging元数据已保存到: {meta_path}")


    def predict_proba(self, X_data: pd.DataFrame) -> np.ndarray:
        """
        使用融合模型进行概率预测。
        目前只实现了Bagging平均。
        """
        if not self.models: # 如果没有训练或加载模型
            print("错误: 没有模型可用于预测。请先训练或加载模型。")
            # 尝试加载之前保存的bagging模型
            self._load_bagging_models_if_empty()
            if not self.models:
                 raise ValueError("没有模型可用于预测，并且无法从磁盘加载。")


        # 确保所有模型的特征列表是兼容的（理想情况下是相同的）
        # 这里简化，假设它们是相同的，并且X_data包含了所有需要的特征
        # 实际上，应该使用第一个模型的特征列表来选择X_data的列
        if not self.feature_lists:
            raise ValueError("模型特征列表未定义。")

        # 使用第一个模型的特征列表来筛选输入数据
        # (假设所有bagging模型的特征顺序和名称一致)
        try:
            X_data_filtered = X_data[self.feature_lists[0]]
        except KeyError as e:
            missing_cols = set(self.feature_lists[0]) - set(X_data.columns)
            raise ValueError(f"输入数据X_data中缺少必要的特征列: {missing_cols}. 错误: {e}")


        all_probas = []
        for i, model in enumerate(self.models):
            # 如果特征列表不一致，这里需要使用 self.feature_lists[i]
            probas = model.predict_proba(X_data_filtered)[:, 1]
            all_probas.append(probas)

        if not all_probas:
            raise ValueError("未能从任何模型获得预测概率。")

        # Bagging平均
        avg_probas = np.mean(all_probas, axis=0)
        return avg_probas

    def _load_bagging_models_if_empty(self):
        """如果self.models为空，尝试从磁盘加载之前训练的bagging模型"""
        if self.models: # 如果已有模型，则不加载
            return

        print("  当前无已加载模型，尝试从磁盘加载Bagging模型...")
        bagging_params = self.ensemble_conf.get("bagging_params", {})
        n_expected_models = bagging_params.get("n_models", 0)
        if n_expected_models == 0:
            print("    配置文件中 n_models 为0或未配置，无法加载。")
            return

        loaded_models = []
        loaded_features = []
        loaded_thresholds = []
        for i in range(n_expected_models):
            model_id = f"bagging_model_{i+1}"
            model_path = os.path.join(self.model_output_dir, f"lgbm_model_{model_id}.pkl")
            feature_path = os.path.join(self.model_output_dir, f"lgbm_features_{model_id}.txt")
            threshold_path = os.path.join(self.model_output_dir, f"best_f1_threshold_{model_id}.txt")

            if os.path.exists(model_path) and os.path.exists(feature_path) and os.path.exists(threshold_path):
                try:
                    model = joblib.load(model_path)
                    with open(feature_path, 'r') as f_feat:
                        features = [line.strip() for line in f_feat if line.strip()]
                    with open(threshold_path, 'r') as f_thresh:
                        threshold = float(f_thresh.read().strip())

                    loaded_models.append(model)
                    loaded_features.append(features)
                    loaded_thresholds.append(threshold)
                    print(f"    成功加载模型: {model_id}")
                except Exception as e:
                    print(f"    加载模型 {model_id} 文件时出错: {e}")
                    loaded_models.clear(); loaded_features.clear(); loaded_thresholds.clear() # 一旦出错则不加载任何
                    break
            else:
                print(f"    模型 {model_id} 的部分或全部文件未找到，无法加载Bagging模型。")
                loaded_models.clear(); loaded_features.clear(); loaded_thresholds.clear()
                break
        
        if loaded_models:
            # 检查所有加载模型的特征列表是否一致 (简单检查长度和第一个/最后一个)
            first_feature_list = loaded_features[0]
            consistent = True
            for fl in loaded_features[1:]:
                if len(fl) != len(first_feature_list) or fl[0] != first_feature_list[0] or fl[-1] != first_feature_list[-1]:
                    consistent = False
                    break
            if not consistent:
                print("警告：加载的Bagging模型的特征列表不一致！融合预测可能不准确。")
                # 可以选择不加载，或者只使用第一个模型的特征列表（有风险）

            self.models = loaded_models
            self.feature_lists = loaded_features # 存储每个模型的特征列表
            self.best_thresholds = loaded_thresholds
            print(f"  成功从磁盘加载 {len(self.models)} 个Bagging模型。")


if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    RANK_CONFIG = os.path.join(PROJECT_ROOT, 'conf', 'rank', 'rank_config.yaml')

    ensembler = ModelEnsembler(project_root=PROJECT_ROOT, rank_config_path=RANK_CONFIG)
    
    # 训练Bagging模型 (假设train_s1, valid_s1样本已准备好)
    # 注意：这会运行多次LGBM训练，可能耗时较长
    if ensembler.ensemble_conf.get("enabled"):
        try:
            ensembler.train_bagging_models(train_key="train_s1", valid_key="valid_s1")
            print("\nModelEnsembler.train_bagging_models() 执行完毕。")

            # 示例：加载测试数据并进行预测 (假设 test_target_samples.parquet 已生成)
            test_sample_file = os.path.join(PROJECT_ROOT, ensembler.global_conf["processed_sample_path"], "test_target_samples.parquet")
            if os.path.exists(test_sample_file) and ensembler.models:
                print("\n加载测试数据进行融合预测示例...")
                df_test_samples = pd.read_parquet(test_sample_file)
                # 假设 df_test_samples 包含所有必要的特征列
                # 对于测试集，我们没有 y_true
                # X_test = df_test_samples.drop(columns=['user_id', 'item_id', 'label'] + rank_config.get("exclude_features_from_model",[]), errors='ignore')
                # 这里简化，假设df_test_samples 就是 X_test
                
                # 确保测试集包含模型训练时使用的所有特征
                # 我们使用第一个训练好的模型的特征列表来筛选
                if ensembler.feature_lists:
                    test_feature_cols = ensembler.feature_lists[0]
                    missing_in_test = set(test_feature_cols) - set(df_test_samples.columns)
                    if missing_in_test:
                        print(f"错误: 测试样本中缺少以下特征: {missing_in_test}")
                    else:
                        X_test_for_pred = df_test_samples[test_feature_cols]
                        ensemble_probas = ensembler.predict_proba(X_test_for_pred)
                        print(f"融合预测概率 (前5个): {ensemble_probas[:5]}")
                else:
                    print("错误：无法确定用于预测的特征列表。")

            elif not ensembler.models:
                print("未训练或加载任何模型，无法进行预测。")
            else:
                print(f"测试样本文件未找到: {test_sample_file}")

        except FileNotFoundError as e:
            print(f"错误: {e}")
            print("请确保已运行 src/rank/data_builder_for_rank.py 来生成所需的 _samples.parquet 文件。")
        except Exception as e:
            import traceback
            print(f"融合模型训练或预测过程中发生未知错误: {e}")
            traceback.print_exc()
    else:
        print("模型融合未在配置文件中启用。")