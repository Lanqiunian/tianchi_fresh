# src/rank/lgbm_trainer.py
import pandas as pd
import lightgbm as lgb
import optuna
import os
import joblib 
from sklearn.metrics import roc_auc_score
import yaml 
import numpy as np
import gc # Import garbage collector

from ..utils.config_loader import load_config
from .utils_rank import calculate_f1_metrics, find_best_f1_threshold

class LGBMTrainer:
    def __init__(self, project_root: str, rank_config_path: str, model_identifier: str = "primary"):
        self.project_root = project_root
        try:
            if not os.path.isabs(rank_config_path):
                effective_config_path = os.path.join(self.project_root, rank_config_path)
            else:
                effective_config_path = rank_config_path
            self.rank_config = load_config(effective_config_path)
        except FileNotFoundError:
            if os.path.isabs(rank_config_path) and os.path.exists(rank_config_path):
                 print(f"    LGBMTrainer: Loading config from temporary path: {rank_config_path}")
                 with open(rank_config_path, 'r', encoding='utf-8') as f_temp_conf:
                    self.rank_config = yaml.safe_load(f_temp_conf)
            else:
                print(f"Error: Config file not found at {effective_config_path} (or original {rank_config_path})")
                raise
        
        self.model_identifier = model_identifier
        self.global_conf = self.rank_config.get("global_settings", {})
        self.lgbm_conf = self.rank_config.get("lgbm_training", {})
        self.dataset_conf = self.rank_config.get("dataset_config", {})

        self.processed_sample_dir = os.path.join(self.project_root, self.global_conf.get("processed_sample_path", "data/2_processed/"))
        self.model_output_dir = os.path.join(self.project_root, self.global_conf.get("model_output_path", "models/rank/"))
        os.makedirs(self.model_output_dir, exist_ok=True)

        self.exclude_cols = ['user_id', 'item_id', 'label'] + \
                             self.rank_config.get("exclude_features_from_model", [])
        self.scale_pos_weight_calculated = 1.0

    def _convert_param_types(self, params_dict: dict):
        if params_dict is None:
            return {}
        converted_params = params_dict.copy()
        float_params = ['learning_rate', 'feature_fraction', 'bagging_fraction',
                        'lambda_l1', 'lambda_l2', 'min_child_weight', 'scale_pos_weight']
        int_params = ['num_leaves', 'max_depth', 'n_estimators', 
                      'seed', 'n_jobs', 'verbose', 'bagging_freq', 'min_child_samples',
                      'early_stopping_rounds']

        for p_name, p_value in converted_params.items():
            if p_value is None: continue
            original_type = type(p_value)
            try:
                if p_name in float_params:
                    if p_name == 'scale_pos_weight' and str(p_value).lower() == 'auto':
                        converted_params[p_name] = 'auto'
                    elif not isinstance(p_value, (float, int)):
                        converted_params[p_name] = float(p_value)
                    elif isinstance(p_value, int):
                        converted_params[p_name] = float(p_value)
                elif p_name in int_params:
                    if not isinstance(p_value, int):
                        converted_params[p_name] = int(float(p_value))
                # No print here to reduce log spam, conversion is best effort
            except (ValueError, TypeError):
                print(f"    WARNING: Could not convert param '{p_name}' (value: '{p_value}', type: {original_type}) to expected numeric type. Will use original or let LightGBM handle.")
        return converted_params

    def _load_data(self, train_key: str, valid_key: str):
        train_valid_split_conf = self.dataset_conf.get("train_valid_split", {})
        train_conf = train_valid_split_conf.get(train_key)
        valid_conf = train_valid_split_conf.get(valid_key)

        if not train_conf: train_conf = self.dataset_conf.get(train_key)
        if not valid_conf: 
            valid_conf = self.dataset_conf.get(valid_key)
            if not valid_conf and valid_key in train_valid_split_conf:
                valid_conf = train_valid_split_conf.get(valid_key)

        if not train_conf: raise ValueError(f"Config for TRAIN key '{train_key}' not found.")
        if not valid_conf: raise ValueError(f"Config for VALID key '{valid_key}' not found.")

        train_sample_filename = f"{train_key}_samples.parquet"
        valid_sample_filename = f"{valid_key}_samples.parquet"
        train_file = os.path.join(self.processed_sample_dir, train_sample_filename)
        valid_file = os.path.join(self.processed_sample_dir, valid_sample_filename)

        if not os.path.exists(train_file): raise FileNotFoundError(f"训练样本文件未找到: {train_file}")
        if not os.path.exists(valid_file): raise FileNotFoundError(f"验证样本文件未找到: {valid_file}")

        print(f"  加载训练数据: {train_file}")
        df_train = pd.read_parquet(train_file)
        print(f"  加载验证数据: {valid_file}")
        df_valid = pd.read_parquet(valid_file)
        
        print(f"    训练集原始形状 (加载后): {df_train.shape}, 验证集原始形状: {df_valid.shape}")
        if 'label' not in df_train.columns or 'label' not in df_valid.columns:
            raise ValueError("错误: 训练集或验证集缺少 'label' 列。")
            
        print(f"    训练集原始标签分布:\n{df_train['label'].value_counts(normalize=True, dropna=False)}")
        print(f"    验证集标签分布:\n{df_valid['label'].value_counts(normalize=True, dropna=False)}")

        # --- BEGIN: Negative Undersampling for Training Data ---
        undersampling_conf = self.lgbm_conf.get("undersampling", {})
        apply_undersampling = undersampling_conf.get("enabled", False)
        # Undersampling only applies to the training set identified by train_key,
        # NOT to validation or test sets.
        if apply_undersampling and train_key.startswith("train"): # Be more specific if needed
            target_ratio = undersampling_conf.get("ratio", None)
            sampling_seed = undersampling_conf.get("random_seed", 2024)

            if target_ratio is not None and target_ratio > 0:
                positive_samples = df_train[df_train['label'] == 1]
                negative_samples = df_train[df_train['label'] == 0]
                num_positive = len(positive_samples)
                num_negative_to_keep = int(num_positive * target_ratio)

                if len(negative_samples) > num_negative_to_keep:
                    print(f"  对训练集 '{train_key}' 进行负样本随机欠采样：正样本数={num_positive}, 负样本数从{len(negative_samples)}采样到{num_negative_to_keep} (目标比例 1:{target_ratio})")
                    negative_samples_sampled = negative_samples.sample(n=num_negative_to_keep, random_state=sampling_seed)
                    df_train = pd.concat([positive_samples, negative_samples_sampled]).sample(frac=1, random_state=sampling_seed).reset_index(drop=True) # Shuffle
                    print(f"    采样后训练集形状: {df_train.shape}, 新的标签分布:\n{df_train['label'].value_counts(normalize=True, dropna=False)}")
                    gc.collect()
                else:
                    print(f"  训练集 '{train_key}' 负样本数量 ({len(negative_samples)}) 已少于或等于目标采样数 ({num_negative_to_keep})，不进行欠采样。")
            else:
                print(f"  训练集 '{train_key}' 负样本欠采样已启用但目标比例无效 ({target_ratio})，不进行采样。")
        # --- END: Negative Undersampling ---

        explicit_exclude_list = self.exclude_cols + self.rank_config.get("exclude_features_from_model",[])
        feature_columns = [col for col in df_train.columns if col not in explicit_exclude_list]
        feature_columns = [col for col in feature_columns if col in df_valid.columns]
        print(f"  使用的特征数量: {len(feature_columns)}")
        if not feature_columns: raise ValueError("没有可用的特征列进行训练。")

        X_train = df_train[feature_columns]
        y_train = df_train['label'] # y_train is now from potentially undersampled df_train
        X_valid = df_valid[feature_columns]
        y_valid = df_valid['label']

        # Recalculate scale_pos_weight based on (potentially undersampled) y_train
        base_params_conf = self._convert_param_types(self.lgbm_conf.get("base_params", {}))
        scale_pos_weight_setting = base_params_conf.get("scale_pos_weight")
        is_unbalance_setting = base_params_conf.get("is_unbalance", False)

        if str(scale_pos_weight_setting).lower() == "auto" or \
           (scale_pos_weight_setting is None and is_unbalance_setting is True):
            num_negative_train = (y_train == 0).sum() # Use current y_train
            num_positive_train = (y_train == 1).sum() # Use current y_train
            if num_positive_train > 0:
                self.scale_pos_weight_calculated = float(num_negative_train) / num_positive_train
                print(f"  自动计算 scale_pos_weight (基于当前训练集 y_train): {self.scale_pos_weight_calculated:.2f} (neg={num_negative_train}, pos={num_positive_train})")
            else:
                self.scale_pos_weight_calculated = 1.0 
                print("  警告: 当前训练集 y_train 中无正样本，scale_pos_weight_calculated 设为1.0")
        elif isinstance(scale_pos_weight_setting, (int, float)):
            self.scale_pos_weight_calculated = float(scale_pos_weight_setting)
            print(f"  使用配置中明确的 scale_pos_weight: {self.scale_pos_weight_calculated:.2f}")
        else: 
            self.scale_pos_weight_calculated = 1.0
            print(f"  scale_pos_weight 未在配置中指定为'auto'或数字，将使用默认值: {self.scale_pos_weight_calculated:.2f}")
            if scale_pos_weight_setting is not None:
                 print(f"    (配置文件中的原始值为: '{scale_pos_weight_setting}')")
        return X_train, y_train, X_valid, y_valid, feature_columns

    def _objective(self, trial: optuna.Trial, X_train, y_train, X_valid, y_valid):
        print(f"--- Starting Optuna Trial {trial.number} ---") 
        optuna_params_conf = self.lgbm_conf.get("optuna_tuning", {}).get("param_distributions", {})
        params = self._convert_param_types(self.lgbm_conf.get("base_params", {}).copy())

        if 'scale_pos_weight' not in optuna_params_conf: 
            if str(params.get('scale_pos_weight')).lower() == 'auto' or 'scale_pos_weight' not in params or params.get('scale_pos_weight') is None :
                params['scale_pos_weight'] = self.scale_pos_weight_calculated
        
        print("--- DEBUG: Optuna _objective ---")
        for param_name, dist_conf in optuna_params_conf.items():
            param_type = dist_conf.get("type")
            try:
                current_low = float(dist_conf["low"])
                current_high = float(dist_conf["high"])
            except (TypeError, ValueError) as e:
                print(f"ERROR: Invalid low/high for Optuna param '{param_name}'. Values: '{dist_conf.get('low')}', '{dist_conf.get('high')}'. Error: {e}")
                raise
            if param_type == "float":
                params[param_name] = trial.suggest_float(param_name, current_low, current_high, log=dist_conf.get("log", False))
            elif param_type == "int":
                params[param_name] = trial.suggest_int(param_name, int(current_low), int(current_high), log=dist_conf.get("log", False))
        
        # Ensure critical numeric params are indeed numeric before passing to LGBM
        for p_key in ['scale_pos_weight', 'min_child_weight', 'num_leaves', 'min_child_samples', 'learning_rate']:
            if p_key in params and params[p_key] is not None: # Check for None as well
                if not isinstance(params[p_key], (int, float)):
                    try:
                        if p_key in ['num_leaves', 'min_child_samples', 'bagging_freq']: # Params that must be int
                            params[p_key] = int(float(params[p_key]))
                        elif p_key == 'scale_pos_weight' and str(params[p_key]).lower() == 'auto':
                            params[p_key] = self.scale_pos_weight_calculated # Convert 'auto' if it slipped through
                        else: # Floats
                            params[p_key] = float(params[p_key])
                    except (ValueError, TypeError) as e_conv:
                        print(f"    Trial {trial.number} ERROR: Failed to convert param '{p_key}' (value: '{params[p_key]}') to numeric: {e_conv}. Defaulting or stopping might be needed.")
                        # Fallback for critical params if conversion fails
                        if p_key == 'scale_pos_weight': params[p_key] = 1.0
                        elif p_key == 'min_child_weight': params[p_key] = 1e-3
                        else: pass # Let LightGBM potentially error on other types

        if 'scale_pos_weight' in params: print(f"    Trial {trial.number} Effective scale_pos_weight for LGBM: {params.get('scale_pos_weight', 1.0):.2f}")
        if 'min_child_weight' in params: print(f"    Trial {trial.number} Effective min_child_weight for LGBM: {params.get('min_child_weight', 1e-3):.2e}")

        print(f"    Trial {trial.number} 参数 (进入LGBM): {params}")

        try:
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train,
                      eval_set=[(X_valid, y_valid)],
                      eval_metric=params.get('metric', 'auc'),
                      callbacks=[lgb.early_stopping(self.lgbm_conf.get("early_stopping_rounds", 50), verbose=False)]
                     )
        except Exception as e:
            print(f"ERROR during model.fit in Trial {trial.number}: {e}")
            print(f"      Parameters used: {params}")
            raise 

        y_pred_proba_valid = model.predict_proba(X_valid)[:, 1]
        
        print(f"    Trial {trial.number} - Valid y_pred_proba (sample of 5): {y_pred_proba_valid[:5]}")
        print(f"    Trial {trial.number} - Valid y_pred_proba min: {np.min(y_pred_proba_valid):.6f}, max: {np.max(y_pred_proba_valid):.6f}, mean: {np.mean(y_pred_proba_valid):.6f}")

        optuna_objective_metric = self.lgbm_conf.get("optuna_tuning", {}).get("objective_metric", "f1").lower()

        if optuna_objective_metric == "f1":
            f1_opt_conf = self.rank_config.get("prediction",{}).get("f1_threshold_optimization",{})
            optuna_f1_start = f1_opt_conf.get("search_range_start_optuna", 0.001)
            optuna_f1_end = f1_opt_conf.get("search_range_end_optuna", 0.99) 
            optuna_f1_step = f1_opt_conf.get("search_step_optuna", 0.01)

            best_thr, current_f1, current_p, current_r = find_best_f1_threshold(
                                                            y_valid, y_pred_proba_valid,
                                                            start=optuna_f1_start, end=optuna_f1_end, step=optuna_f1_step)
            print(f"    Trial {trial.number} - F1 search (Thr {best_thr:.4f}) -> P: {current_p:.4f}, R: {current_r:.4f}, F1: {current_f1:.4f}")
            return current_f1
        elif optuna_objective_metric == "auc":
            auc = roc_auc_score(y_valid, y_pred_proba_valid)
            print(f"    Trial {trial.number} - 验证集 AUC: {auc:.4f}")
            return auc
        else:
            raise ValueError(f"未知的 Optuna 优化指标: {optuna_objective_metric}")

    def train(self, train_key: str = "train_s1", valid_key: str = "valid_s1", use_optuna: bool = True):
        print(f"\n开始训练LGBM模型 (标识: {self.model_identifier})...")
        # _load_data now handles undersampling and recalculates self.scale_pos_weight_calculated
        X_train, y_train, X_valid, y_valid, feature_columns = self._load_data(train_key, valid_key)

        best_params = self._convert_param_types(self.lgbm_conf.get("base_params", {}).copy())
        
        # Ensure scale_pos_weight for best_params is correctly set using the (potentially re-calculated) class attribute
        if str(best_params.get("scale_pos_weight")).lower() == "auto" or \
           "scale_pos_weight" not in best_params or \
           best_params.get("scale_pos_weight") is None:
            best_params["scale_pos_weight"] = self.scale_pos_weight_calculated
            print(f"  Initial best_params (for final model) using calculated scale_pos_weight: {best_params['scale_pos_weight']:.2f}")
        elif isinstance(best_params.get("scale_pos_weight"), (int, float)):
             print(f"  Initial best_params (for final model) using explicitly configured scale_pos_weight: {best_params['scale_pos_weight']:.2f}")
        
        if "min_child_weight" not in best_params or not isinstance(best_params.get("min_child_weight"), (int,float)):
            default_mcw = 1e-3 # LightGBM default
            print(f"  'min_child_weight' not numeric or missing in base_params for final model, defaulting to {default_mcw}.")
            best_params["min_child_weight"] = default_mcw

        if use_optuna and self.lgbm_conf.get("optuna_tuning", {}).get("enabled", False):
            print("  使用 Optuna 进行超参数调优...")
            optuna_conf = self.lgbm_conf.get("optuna_tuning", {})
            study = optuna.create_study(direction=optuna_conf.get("direction", "maximize"))
            
            study.optimize(lambda trial: self._objective(trial, X_train, y_train, X_valid, y_valid),
                           n_trials=optuna_conf.get("n_trials", 50),
                           show_progress_bar=True) 

            print(f"  Optuna 调优完成。最佳 Trial: {study.best_trial.number}")
            print(f"  最佳值 ({optuna_conf.get('objective_metric', 'f1')}): {study.best_value:.4f}")
            print(f"  最佳参数 (来自Optuna): {study.best_params}")
            
            for key, value in study.best_params.items(): # Optuna's suggestions take precedence
                best_params[key] = value 
            
            # Log final effective scale_pos_weight and min_child_weight for best_params after Optuna
            print(f"  Optuna后, best_params中的 scale_pos_weight: {best_params.get('scale_pos_weight', self.scale_pos_weight_calculated):.2f}")
            print(f"  Optuna后, best_params中的 min_child_weight: {best_params.get('min_child_weight', 1e-3):.2e}")
        else:
            print("  不使用 Optuna，使用配置文件中的基础参数（已应用计算的scale_pos_weight和默认min_child_weight）进行训练。")
            print(f"  最终模型将使用的 scale_pos_weight: {best_params.get('scale_pos_weight', self.scale_pos_weight_calculated):.2f}")
            print(f"  最终模型将使用的 min_child_weight: {best_params.get('min_child_weight', 1e-3):.2e}")

        print("\n  使用最终参数训练最终模型...")
        final_model_params = self._convert_param_types(best_params) # Ensure types one last time
        print(f"  最终参数 (进入LGBM): {final_model_params}")
        final_model = lgb.LGBMClassifier(**final_model_params)
        
        final_model.fit(X_train, y_train,
                        eval_set=[(X_valid, y_valid)],
                        eval_metric=final_model_params.get('metric', 'auc'), 
                        callbacks=[lgb.early_stopping(int(self.lgbm_conf.get("early_stopping_rounds", 50)), verbose=True)] # ensure int
                       )

        model_filename = f"lgbm_model_{self.model_identifier}.pkl"
        model_path = os.path.join(self.model_output_dir, model_filename)
        joblib.dump(final_model, model_path)
        print(f"  模型已保存到: {model_path}")

        feature_list_filename = f"lgbm_features_{self.model_identifier}.txt"
        feature_list_path = os.path.join(self.model_output_dir, feature_list_filename)
        with open(feature_list_path, 'w', encoding='utf-8') as f:
            for feature in feature_columns:
                f.write(f"{feature}\n")
        print(f"  特征列表已保存到: {feature_list_path}")

        y_pred_proba_valid_final = final_model.predict_proba(X_valid)[:, 1]
        f1_opt_conf_final = self.rank_config.get("prediction",{}).get("f1_threshold_optimization",{})
        best_threshold, best_f1, final_p, final_r = find_best_f1_threshold(
            y_valid, y_pred_proba_valid_final,
            start=f1_opt_conf_final.get("search_range_start", 0.01),
            end=f1_opt_conf_final.get("search_range_end", 0.5),
            step=f1_opt_conf_final.get("search_step", 0.005)
        )
        
        threshold_filename = f"best_f1_threshold_{self.model_identifier}.txt"
        threshold_path = os.path.join(self.model_output_dir, threshold_filename)
        with open(threshold_path, 'w', encoding='utf-8') as f:
            f.write(str(best_threshold))
        print(f"  最终模型在验证集上的评估 (阈值 {best_threshold:.4f}): P: {final_p:.4f}, R: {final_r:.4f}, F1: {best_f1:.4f}")
        print(f"  最佳F1阈值 ({best_threshold:.4f}) 已保存到: {threshold_path}")

        return final_model, feature_columns, best_threshold

if __name__ == "__main__":

    try:
        # 尝试创建一个简单的LGBM GPU Dataset 或 Booster，看是否会抛出异常
        # 这只是一个简单的探测，实际的GPU使用发生在 .fit()
        print("尝试探测LightGBM GPU支持...")
        # LightGBM的GPU支持是在编译时确定的，运行时通过device_type参数指定。
        # 直接检查可能比较困难，但可以看device_type='gpu'时是否报错。
        # 或者检查是否有相关的环境变量
        print(f"CUDA_VISIBLE_DEVICES in Python: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
        # 也可以尝试用torch等库检查GPU是否可见
        # import torch
        # print(f"Torch CUDA available: {torch.cuda.is_available()}")
        # print(f"Torch CUDA device count: {torch.cuda.device_count()}")
    except Exception as e_gpu_check:
        print(f"GPU支持探测时发生错误: {e_gpu_check}")
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    DEFAULT_RANK_CONFIG_PATH = os.path.join(PROJECT_ROOT, 'conf', 'rank', 'rank_config.yaml')

    RANK_CONFIG_TO_USE = DEFAULT_RANK_CONFIG_PATH
    MODEL_ID_TO_USE = "train_s1_model_v_debug" 
    TRAIN_KEY_TO_USE = "train_s1"
    VALID_KEY_TO_USE = "valid_s1"
    
    trainer = LGBMTrainer(project_root=PROJECT_ROOT, 
                          rank_config_path=RANK_CONFIG_TO_USE, 
                          model_identifier=MODEL_ID_TO_USE)
    
    try:
        print(f"Manually running LGBMTrainer for model_identifier: {trainer.model_identifier}")
        print(f"Using training data key: '{TRAIN_KEY_TO_USE}', validation data key: '{VALID_KEY_TO_USE}'")
        
        use_optuna_for_manual_run = trainer.lgbm_conf.get("optuna_tuning", {}).get("enabled", False)
        print(f"Optuna enabled for this manual run (from config): {use_optuna_for_manual_run}")

        trained_model, features, threshold = trainer.train(
            train_key=TRAIN_KEY_TO_USE, 
            valid_key=VALID_KEY_TO_USE, 
            use_optuna=use_optuna_for_manual_run
        )
        print("\nLGBMTrainer.train() (manual run) 执行完毕。")
        print(f"  模型对象: {type(trained_model)}")
        print(f"  特征数量: {len(features)}")
        print(f"  找到的最佳阈值 (基于最终模型在验证集上): {threshold:.4f}")

    except FileNotFoundError as e:
        print(f"错误 (manual run): {e}")
        print("请确保已运行 src/rank/data_builder_for_rank.py 来生成所需的 _samples.parquet 文件。")
        print("并检查 rank_config.yaml 中的数据集键名是否正确。")
    except ValueError as e:
        print(f"配置或数据错误 (manual run): {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        import traceback
        print(f"训练过程中发生未知错误 (manual run): {e}")
        traceback.print_exc()