# src/feature_engineering/target_encode_features.py
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import os
import joblib
import gc

from .feature_generator_base import FeatureGeneratorBase

class TargetEncodeFeatures(FeatureGeneratorBase):
    def __init__(self, generator_name: str,
                 generator_specific_config: dict,
                 global_feature_engineering_config: dict,
                 user_log_df: pd.DataFrame,
                 items_df: pd.DataFrame,
                 project_root_path: str,
                 is_training_run: bool,
                 target_col_name: str = 'label'):
        super().__init__(generator_name,
                         generator_specific_config,
                         global_feature_engineering_config,
                         user_log_df, items_df,
                         project_root_path)
        self.params = self.config.get("params", {})
        self.is_training_run = is_training_run
        self.target_col_name = target_col_name
        self.smoothing_alpha = self.params.get("alpha", 20)
        self.n_folds = self.params.get("cv_folds", 5)
        self.features_to_encode_config = self.params.get("features_to_encode", [])

        self.learned_encodings_dir = os.path.join(
            self.project_root_path,
            "models", "rank", "target_encodings"
        )
        os.makedirs(self.learned_encodings_dir, exist_ok=True)

        for cols_to_encode_list in self.features_to_encode_config:
            if not isinstance(cols_to_encode_list, list) or not cols_to_encode_list:
                continue
            group_key = "_".join(cols_to_encode_list)
            te_col_name = f'te_{group_key}'
            self._add_feature_name(te_col_name)

    def _get_encoding_filepath(self, group_key: str) -> str:
        return os.path.join(self.learned_encodings_dir, f"te_mapping_{group_key}.pkl")

    def _calculate_and_apply_encoding(self,
                                      data_to_encode_on: pd.DataFrame,
                                      data_to_apply_to: pd.DataFrame,
                                      group_key_col_name: str,
                                      new_feature_name: str):
        if self.target_col_name not in data_to_encode_on.columns or data_to_encode_on[self.target_col_name].empty:
            fold_global_mean_target = 0.01
        else:
            fold_global_mean_target = data_to_encode_on[self.target_col_name].mean()
            if pd.isna(fold_global_mean_target):
                fold_global_mean_target = 0.01
        
        if data_to_encode_on.empty or group_key_col_name not in data_to_encode_on.columns:
            target_mean_smooth = pd.Series(dtype=float)
        else:
            agg = data_to_encode_on.groupby(group_key_col_name)[self.target_col_name].agg(['sum', 'count'])
            if agg.empty:
                target_mean_smooth = pd.Series(dtype=float)
            else:
                target_mean_smooth = (agg['sum'] + fold_global_mean_target * self.smoothing_alpha) / \
                                     (agg['count'] + self.smoothing_alpha)
        
        if new_feature_name not in data_to_apply_to.columns:
            data_to_apply_to[new_feature_name] = np.nan

        if not target_mean_smooth.empty and group_key_col_name in data_to_apply_to.columns:
            data_to_apply_to[new_feature_name] = data_to_apply_to[group_key_col_name].map(target_mean_smooth)
            # FIX: Avoid inplace=True for fillna
            data_to_apply_to[new_feature_name] = data_to_apply_to[new_feature_name].fillna(fold_global_mean_target)
        else:
            data_to_apply_to[new_feature_name] = fold_global_mean_target
            
        return data_to_apply_to

    def generate_features(self, master_df_with_features_and_labels: pd.DataFrame, behavior_end_date: pd.Timestamp) -> pd.DataFrame:
        print(f"  生成 TargetEncodeFeatures (is_training_run={self.is_training_run})...")
        
        df_processed = master_df_with_features_and_labels 

        all_base_cols_needed_for_te = set()
        if self.is_training_run:
            all_base_cols_needed_for_te.add(self.target_col_name)

        valid_encoding_configs = [] 

        for original_cols_list in self.features_to_encode_config:
            if not isinstance(original_cols_list, list) or not original_cols_list:
                continue
            
            if not all(col in df_processed.columns for col in original_cols_list):
                missing_cols = [col for col in original_cols_list if col not in df_processed.columns]
                temp_group_key = "_".join(original_cols_list)
                temp_te_name = f'te_{temp_group_key}'
                print(f"警告: TE 的基础列 {missing_cols} 在输入DataFrame中缺失。将跳过TE特征 '{temp_te_name}'。")
                if temp_te_name in self.feature_names_generated:
                    self.feature_names_generated.remove(temp_te_name)
                continue

            for base_col in original_cols_list:
                all_base_cols_needed_for_te.add(base_col)
            
            combined_key_name = "_".join(original_cols_list)
            te_feature_name = f'te_{combined_key_name}'
            valid_encoding_configs.append((original_cols_list, combined_key_name, te_feature_name))
            
            if te_feature_name not in df_processed.columns:
                df_processed[te_feature_name] = np.nan

        if self.is_training_run:
            print(f"    为训练数据生成 Target Encoding (使用 {self.n_folds}-Fold CV)...")
            if self.target_col_name not in df_processed.columns:
                raise ValueError(f"错误: 目标列 '{self.target_col_name}' 在 DataFrame 中未找到。")

            df_for_kfold_base = df_processed[list(all_base_cols_needed_for_te)].copy()
            gc.collect()

            created_combined_keys = set()
            for original_cols_list, combined_key_name, _ in valid_encoding_configs:
                if len(original_cols_list) > 1 and combined_key_name not in created_combined_keys:
                    if combined_key_name not in df_for_kfold_base.columns:
                         df_for_kfold_base[combined_key_name] = df_for_kfold_base[original_cols_list].astype(str).apply(lambda row: '_'.join(row.values), axis=1)
                         created_combined_keys.add(combined_key_name)
            gc.collect()
            
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.global_fe_config.get("seed", 2024) + 2)

            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(df_for_kfold_base)):
                df_train_fold_subset = df_for_kfold_base.iloc[train_idx].copy()
                df_val_fold_subset = df_for_kfold_base.iloc[val_idx].copy()   
                
                for original_cols_list, combined_key_name, te_feature_name in valid_encoding_configs:
                    key_col_for_op = combined_key_name if len(original_cols_list) > 1 else original_cols_list[0]
                    
                    df_val_fold_subset = self._calculate_and_apply_encoding(
                        df_train_fold_subset, 
                        df_val_fold_subset,
                        key_col_for_op,
                        te_feature_name
                    )
                
                for _, _, te_feature_name in valid_encoding_configs:
                    if te_feature_name in df_val_fold_subset.columns:
                        df_processed.loc[df_processed.index[val_idx], te_feature_name] = df_val_fold_subset[te_feature_name].values
                
                del df_train_fold_subset, df_val_fold_subset
                gc.collect()
            
            del df_for_kfold_base
            gc.collect()

            print("    在整个训练集上计算并保存最终的 Target Encoding 映射...")
            overall_global_mean = df_processed[self.target_col_name].mean() if self.target_col_name in df_processed else 0.01
            if pd.isna(overall_global_mean): overall_global_mean = 0.01

            df_for_saving_map_base = df_processed.copy()

            for original_cols_list, combined_key_name_for_file, _ in valid_encoding_configs:
                key_col_for_saving_op = combined_key_name_for_file if len(original_cols_list) > 1 else original_cols_list[0]
                
                if len(original_cols_list) > 1 and key_col_for_saving_op not in df_for_saving_map_base.columns:
                    df_for_saving_map_base[key_col_for_saving_op] = df_for_saving_map_base[original_cols_list].astype(str).apply(lambda row: '_'.join(row.values), axis=1)

                if df_for_saving_map_base.empty or key_col_for_saving_op not in df_for_saving_map_base.columns:
                    learned_mapping_global = pd.Series(dtype=float)
                else:
                    agg_global = df_for_saving_map_base.groupby(key_col_for_saving_op)[self.target_col_name].agg(['sum', 'count'])
                    if agg_global.empty:
                        learned_mapping_global = pd.Series(dtype=float)
                    else:
                        learned_mapping_global = (agg_global['sum'] + overall_global_mean * self.smoothing_alpha) / \
                                                 (agg_global['count'] + self.smoothing_alpha)
                
                filepath_to_save = self._get_encoding_filepath(combined_key_name_for_file)
                try:
                    joblib.dump({"mapping": learned_mapping_global, "global_mean_for_fill": overall_global_mean}, filepath_to_save)
                except Exception as e:
                    print(f"错误: 保存全局编码映射 '{combined_key_name_for_file}' 失败: {e}")
            del df_for_saving_map_base
            gc.collect()

        else: 
            print("    为验证/测试数据生成 Target Encoding (加载已学习的映射)...")
            df_apply_te_non_train = df_processed.copy()

            for original_cols_list, combined_key_name_for_file, te_feature_name in valid_encoding_configs:
                key_col_for_apply_op = combined_key_name_for_file if len(original_cols_list) > 1 else original_cols_list[0]
                
                if len(original_cols_list) > 1 and key_col_for_apply_op not in df_apply_te_non_train.columns:
                    df_apply_te_non_train[key_col_for_apply_op] = df_apply_te_non_train[original_cols_list].astype(str).apply(lambda row: '_'.join(row.values), axis=1)
                
                mapping_filepath = self._get_encoding_filepath(combined_key_name_for_file)
                if os.path.exists(mapping_filepath):
                    try:
                        saved_map_data = joblib.load(mapping_filepath)
                        loaded_mapping = saved_map_data["mapping"]
                        mean_for_fill = saved_map_data.get("global_mean_for_fill", 0.01)

                        if loaded_mapping.empty or key_col_for_apply_op not in df_apply_te_non_train.columns:
                            df_processed[te_feature_name] = mean_for_fill
                        else:
                            df_processed[te_feature_name] = df_apply_te_non_train[key_col_for_apply_op].map(loaded_mapping)
                            # FIX: Avoid inplace=True for fillna
                            df_processed[te_feature_name] = df_processed[te_feature_name].fillna(mean_for_fill)
                    except Exception as e:
                        print(f"错误: 加载或应用编码映射 {mapping_filepath} 失败: {e}。TE特征 '{te_feature_name}' 将填充默认值。")
                        df_processed[te_feature_name] = 0.01
                else:
                    print(f"警告: 未找到已学习的编码映射文件: {mapping_filepath}。TE特征 '{te_feature_name}' 将填充默认值。")
                    df_processed[te_feature_name] = 0.01
            
            del df_apply_te_non_train
            gc.collect()
        
        final_return_cols = []
        if 'user_id' in df_processed.columns: final_return_cols.append('user_id')
        if 'item_id' in df_processed.columns: final_return_cols.append('item_id')
        
        for expected_te_col in self.get_generated_feature_names():
            if expected_te_col not in df_processed.columns:
                print(f"警告: 预期的TE特征 '{expected_te_col}' 在df_processed中不存在，将用默认值创建。")
                df_processed[expected_te_col] = 0.01
            
            if expected_te_col not in final_return_cols:
                 final_return_cols.append(expected_te_col)
        
        final_return_cols_unique = []
        if 'user_id' in final_return_cols:
            final_return_cols_unique.append('user_id')
        if 'item_id' in final_return_cols:
            final_return_cols_unique.append('item_id')
        
        for col in final_return_cols:
            if col not in final_return_cols_unique and col in df_processed.columns:
                final_return_cols_unique.append(col)
        
        for col_from_init in self.get_generated_feature_names():
            if col_from_init not in final_return_cols_unique and col_from_init in df_processed.columns:
                final_return_cols_unique.append(col_from_init)
            elif col_from_init not in df_processed.columns:
                print(f"警告: 最终检查，特征 '{col_from_init}' 在df_processed中缺失，将用默认值创建并包含在返回结果中。")
                df_processed[col_from_init] = 0.01
                if col_from_init not in final_return_cols_unique:
                    final_return_cols_unique.append(col_from_init)

        missing_in_df = [col for col in final_return_cols_unique if col not in df_processed.columns]
        if missing_in_df:
            raise KeyError(f"在尝试返回TE特征时，以下列在df_processed中缺失: {missing_in_df}. df_processed列: {df_processed.columns.tolist()}")

        return df_processed[final_return_cols_unique].copy()