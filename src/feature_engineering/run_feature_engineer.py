# src/feature_engineering/run_feature_engineer.py
import pandas as pd
import os
import time
import importlib
from datetime import timedelta

# 确保可以从正确的相对路径导入
try:
    from ..utils.config_loader import load_config
except ImportError: # 如果直接运行此脚本用于测试，可能需要调整路径
    from utils.config_loader import load_config # type: ignore
try:
    from .feature_utils import parse_behavior_end_date
except ImportError:
    from feature_utils import parse_behavior_end_date # type: ignore


def dynamically_load_generator_class_from_config(gen_config_entry: dict, generator_key_name: str):
    """
    从配置条目中动态加载特征生成器类。
    """
    module_path = gen_config_entry.get("module")
    class_name = gen_config_entry.get("class")

    if not module_path or not class_name:
        raise ValueError(f"生成器 '{generator_key_name}' 的配置中缺少 'module' 或 'class' 字段。配置: {gen_config_entry}")

    try:
        # 使用 package 参数进行相对导入, 'src.feature_engineering' 是包含这些模块的包
        module = importlib.import_module(module_path, package='src.feature_engineering')
        GeneratorClass = getattr(module, class_name)
        return GeneratorClass
    except ImportError as e:
        if isinstance(e, ModuleNotFoundError) and e.name == module_path.lstrip('.'):
             raise ImportError(f"无法导入模块 '{module_path}' (在包 'src.feature_engineering' 中) 用于生成器 '{generator_key_name}'. "
                               f"请检查文件名和路径是否正确，以及 '{module_path.lstrip('.')}.py' 文件是否存在于 'src/feature_engineering/' 目录下。原始错误: {e}")
        raise ImportError(f"无法导入模块 '{module_path}' (在包 'src.feature_engineering' 中) 用于生成器 '{generator_key_name}'. 原始错误: {e}")
    except AttributeError:
        # 尝试给出模块实际加载路径以帮助调试
        loaded_module_path = "未知"
        try:
            # 再次尝试加载模块以获取其路径 (如果第一次失败了)
            temp_module = importlib.import_module(module_path, package='src.feature_engineering')
            loaded_module_path = temp_module.__file__
        except:
            pass
        raise ImportError(f"在模块 '{module_path}' (尝试加载路径: {loaded_module_path}) 中未找到类 '{class_name}' 用于生成器 '{generator_key_name}'. 请检查类名是否正确。")
    except Exception as e:
        raise ImportError(f"加载生成器 '{generator_key_name}' 时发生未知错误 (module: {module_path}, class: {class_name}): {e}")


def generate_all_features(
    project_root_path: str,
    feature_engineering_config_path: str,
    recall_candidates_file_path: str,
    output_filename_key: str,
    behavior_end_date_str: str,
    is_training_run: bool = False
):
    """
    主编排函数，用于生成所有启用的手动特征。
    """
    print(f"\n开始为数据集 {output_filename_key} 进行特征工程 (project_root: {project_root_path})")
    print(f"行为数据截止日期 (包含此日): {behavior_end_date_str}")
    if is_training_run:
        print("模式: 训练运行 (将为 Target Encoding 启用交叉验证)")

    overall_start_time = time.time()

    # 1. 加载总的特征工程配置
    try:
        fe_config_total = load_config(feature_engineering_config_path)
    except FileNotFoundError:
        print(f"错误: 特征工程配置文件未找到于 {feature_engineering_config_path}")
        return
    except Exception as e:
        print(f"错误: 加载特征工程配置文件失败: {e}")
        return

    global_fe_settings = fe_config_total.get("global_settings", {})
    # 将 project_root_path 添加到全局配置中，FeatureGeneratorBase 构造函数会使用它
    # (注意：FeatureGeneratorBase 的 __init__ 需要能接收 project_root_path)
    # global_fe_settings["project_root_path"] = project_root_path # 直接添加，而不是用占位符

    all_generator_configs = fe_config_total.get("feature_generators", {})
    processed_data_dir = os.path.join(project_root_path, global_fe_settings.get("processed_data_path", "data/1_interim/"))
    feature_output_dir = os.path.join(project_root_path, global_fe_settings.get("feature_output_path", "data/2_processed/"))
    os.makedirs(feature_output_dir, exist_ok=True)    # 2. 加载基础数据
    print("  正在加载基础 DataFrame...")
    try:
        # 尝试加载全量用户行为数据（包含所有商品集I上的行为）
        user_log_all_path = os.path.join(processed_data_dir, "processed_user_log_all.parquet")
        p_related_log_path = os.path.join(processed_data_dir, "processed_user_log_p_related.parquet")
        
        # 优先使用全量数据，如果不存在则回退到仅P相关的数据
        if os.path.exists(user_log_all_path):
            print("  加载全量用户行为数据（包含商品全集I上的行为）...")
            user_log_df = pd.read_parquet(user_log_all_path)
            print(f"  成功加载全量用户行为数据，利用更广泛的用户行为构建特征！")
        else:
            print("  未找到全量用户行为数据，回退到仅使用P相关数据...")
            user_log_df = pd.read_parquet(p_related_log_path)
            print(f"  注意: 仅使用P子集相关数据可能会限制模型表现。考虑重新运行数据处理模块生成全量数据。")
        
        if 'datetime' not in user_log_df.columns and 'time' in user_log_df.columns:
             user_log_df['datetime'] = pd.to_datetime(user_log_df['time'])
        elif 'datetime' in user_log_df.columns and not pd.api.types.is_datetime64_any_dtype(user_log_df['datetime']):
             user_log_df['datetime'] = pd.to_datetime(user_log_df['datetime'])
        elif 'datetime' not in user_log_df.columns:
             raise ValueError("用户日志 DataFrame 中缺少 'datetime' 列，并且未找到 'time' 列用于转换。")

        items_df = pd.read_parquet(os.path.join(processed_data_dir, "processed_items.parquet"))
        candidates_df = pd.read_parquet(recall_candidates_file_path)
        print(f"    用户日志: {user_log_df.shape}, 物品信息: {items_df.shape}, 候选集: {candidates_df.shape}")
    except FileNotFoundError as e:
        print(f"错误: 基础数据文件未找到。 {e}")
        return
    except Exception as e:
        print(f"加载基础数据时出错: {e}")
        return

    behavior_end_date_ts = parse_behavior_end_date(behavior_end_date_str)
    master_features_df = candidates_df.copy()
    all_generated_feature_names_set = set()
    label_col_name = 'label'

    # --- 新增：确保基础的 item_category 列存在于 master_features_df ---
    if not items_df.empty and 'item_id' in items_df.columns and 'item_category' in items_df.columns:
        if 'item_id' in master_features_df.columns:
            if 'item_category' not in master_features_df.columns:
                print("  合并 item_category 到 master_features_df...")
                item_id_to_category = items_df[['item_id', 'item_category']].drop_duplicates(subset=['item_id'], keep='first')
                master_features_df = pd.merge(master_features_df, item_id_to_category, on='item_id', how='left')
                print(f"    合并 item_category 后 master_features_df 形状: {master_features_df.shape}")
                if master_features_df['item_category'].isnull().any():
                    num_null_categories = master_features_df['item_category'].isnull().sum()
                    print(f"    警告: item_category 合并后存在 {num_null_categories} 个缺失值。")
                    master_features_df['item_category'] = master_features_df['item_category'].fillna(-1) # 假设用-1填充
                    print(f"    item_category 缺失值已填充。")
            else: # item_category 已存在，仍检查并填充 NaN
                print("  master_features_df 中已存在 item_category 列，检查并填充NaN...")
                if master_features_df['item_category'].isnull().any():
                    master_features_df['item_category'] = master_features_df['item_category'].fillna(-1)
        else:
            print("警告: master_features_df 中缺少 'item_id' 列，无法合并 item_category。")
    else:
        print("警告: items_df 为空或缺少 'item_id'/'item_category' 列，无法合并 item_category。")
    # --- 合并 item_category 结束 ---


    # 3. 为 Target Encoding 准备标签 (如果启用)
    te_generator_config_entry = all_generator_configs.get("target_encode", {})
    if te_generator_config_entry and te_generator_config_entry.get("enabled", False):
        print("  正在为 Target Encoding 准备标签...")
        label_date = (behavior_end_date_ts + timedelta(days=1)).strftime('%Y-%m-%d')
        true_purchases_on_label_day = user_log_df[
            (user_log_df['datetime'].dt.strftime('%Y-%m-%d') == label_date) &
            (user_log_df['behavior_type'] == 4)
        ][['user_id', 'item_id']].drop_duplicates()

        if not true_purchases_on_label_day.empty:
            true_purchases_on_label_day[label_col_name] = 1
            master_features_df = pd.merge(master_features_df, true_purchases_on_label_day,
                                          on=['user_id', 'item_id'], how='left')
            master_features_df[label_col_name] = master_features_df[label_col_name].fillna(0).astype(int)
            print(f"    日期 {label_date} 的标签已合并。正样本数量: {master_features_df[label_col_name].sum()}")
        else:
            master_features_df[label_col_name] = 0
            print(f"    未找到标签日 {label_date} 的购买数据。所有标签已设置为0。")


    # 4. 遍历并运行特征生成器 (Target Encoding 最后运行)
    # 先运行所有非 Target Encoding 的生成器
    non_te_generators = {k: v for k, v in all_generator_configs.items() if k != "target_encode"}

    for gen_name_key, specific_gen_config_entry in non_te_generators.items():
        if not isinstance(specific_gen_config_entry, dict):
            print(f"警告: 生成器 '{gen_name_key}' 的配置不是一个字典，跳过。配置: {specific_gen_config_entry}")
            continue
        if specific_gen_config_entry.get("enabled", False):
            print(f"\n  运行特征生成器: {gen_name_key.upper()}")
            start_gen_time = time.time()
            try:
                GeneratorClass = dynamically_load_generator_class_from_config(specific_gen_config_entry, gen_name_key)
                init_args = {
                    "generator_name": gen_name_key,
                    "generator_specific_config": specific_gen_config_entry,
                    "global_feature_engineering_config": global_fe_settings,
                    "user_log_df": user_log_df,
                    "items_df": items_df,
                    "project_root_path": project_root_path # 传递 project_root 给所有生成器
                }
                # 注意：如果某个非TE生成器也需要特殊参数，需要在此处添加
                generator_instance = GeneratorClass(**init_args)

                # 确定传递给 generate_features 的输入 DataFrame
                input_candidates_for_generator = master_features_df[['user_id', 'item_id']].copy()
                if gen_name_key == "user_behavior":
                    input_candidates_for_generator = master_features_df[['user_id']].drop_duplicates().copy()
                elif gen_name_key == "item_dynamic": # 假设有这个生成器
                    input_candidates_for_generator = master_features_df[['item_id']].drop_duplicates().copy()
                # TemporalFeatures 也应该作用于 user_id, item_id 对，尽管其值对所有行相同
                
                features_from_generator_df = generator_instance.generate_features(
                    input_candidates_for_generator, behavior_end_date_ts
                )

                current_gen_feat_names = generator_instance.get_generated_feature_names()
                if not features_from_generator_df.empty:
                    merge_keys = []
                    # 检查返回的DataFrame是否包含预期的键
                    if 'user_id' in features_from_generator_df.columns: merge_keys.append('user_id')
                    if 'item_id' in features_from_generator_df.columns:
                        # 只有当它不是纯用户级或纯物品级特征时，才考虑item_id作为合并键
                        if gen_name_key not in ["user_behavior", "item_dynamic"]:
                            if 'item_id' in master_features_df.columns : # 确保主DF也有此键
                                merge_keys.append('item_id')
                        elif not merge_keys and 'user_id' not in features_from_generator_df.columns : # 如果是纯item级特征
                             if 'item_id' in master_features_df.columns: merge_keys = ['item_id']


                    if not merge_keys and current_gen_feat_names and len(features_from_generator_df) == 1:
                        print(f"    应用全局特征来自 {gen_name_key}...")
                        for col_name in current_gen_feat_names:
                            if col_name in features_from_generator_df.columns and col_name not in ['user_id', 'item_id']:
                                master_features_df[col_name] = features_from_generator_df[col_name].iloc[0]
                                all_generated_feature_names_set.add(col_name)
                    elif merge_keys and current_gen_feat_names:
                        cols_to_merge = merge_keys + [col for col in current_gen_feat_names if col in features_from_generator_df.columns and col not in merge_keys]
                        features_to_merge_df = features_from_generator_df[list(set(cols_to_merge))].copy() # 使用set确保列名唯一

                        if not all(key in master_features_df.columns for key in merge_keys):
                            print(f"    警告: 主 DataFrame 缺少合并键 {merge_keys} 中的部分或全部，无法合并来自 {gen_name_key} 的特征。")
                        else:
                            master_features_df = pd.merge(master_features_df, features_to_merge_df, on=merge_keys, how='left', suffixes=('', '_ عامر_dup'))
                            master_features_df = master_features_df[[col for col in master_features_df.columns if not col.endswith('_ عامر_dup')]]
                            all_generated_feature_names_set.update(current_gen_feat_names)
                            print(f"    成功合并来自 {gen_name_key} 的特征。主 DataFrame 形状: {master_features_df.shape}")
                    elif merge_keys and not current_gen_feat_names:
                         print(f"    生成器 {gen_name_key} 未声明任何新特征名，但返回了带键的DataFrame。")
                    else: # 无合并键也无特征名，或不符合全局特征模式
                        print(f"    警告: 生成器 {gen_name_key} 返回的数据不符合预期的合并模式或未生成特征。")

                elif current_gen_feat_names:
                     print(f"    警告: 生成器 {gen_name_key} 返回了空 DataFrame，但声明了特征 {current_gen_feat_names}。")
                else:
                    print(f"    生成器 {gen_name_key} 未产生任何特征或数据。")

            except ImportError as e:
                print(f"    无法导入或实例化生成器 {gen_name_key}: {e}")
            except Exception as e:
                import traceback
                print(f"    运行生成器 {gen_name_key} 时出错: {e}")
                traceback.print_exc()
            end_gen_time = time.time()
            print(f"    {gen_name_key} 耗时: {end_gen_time - start_gen_time:.2f}秒")
        else:
            print(f"  跳过已禁用的生成器: {gen_name_key}")

    # 在所有其他特征生成完毕后，运行 Target Encoding (如果启用)
    if te_generator_config_entry and te_generator_config_entry.get("enabled", False):
        gen_name_key_te = "target_encode"
        print(f"\n  运行特征生成器: {gen_name_key_te.upper()} (在其他特征之后)")
        start_gen_time_te = time.time()
        try:
            GeneratorClassTE = dynamically_load_generator_class_from_config(te_generator_config_entry, gen_name_key_te)
            init_args_te = {
                "generator_name": gen_name_key_te,
                "generator_specific_config": te_generator_config_entry,
                "global_feature_engineering_config": global_fe_settings,
                "user_log_df": user_log_df,
                "items_df": items_df,
                "project_root_path": project_root_path,
                "is_training_run": is_training_run,
                "target_col_name": label_col_name
            }
            generator_instance_te = GeneratorClassTE(**init_args_te)
            
            # TargetEncodeFeatures.generate_features 接收包含所有已生成特征和标签的 master_features_df
            # 并返回一个包含 user_id, item_id 和新TE特征的 DataFrame
            te_features_df = generator_instance_te.generate_features(
                master_features_df.copy(), # 传递副本以防意外修改原始df
                behavior_end_date_ts
            )

            current_te_feat_names = generator_instance_te.get_generated_feature_names()
            if not te_features_df.empty and current_te_feat_names:
                # TE特征应该只包含 user_id, item_id 和新的TE列
                cols_to_merge_te = ['user_id', 'item_id'] + [
                    col for col in current_te_feat_names
                    if col in te_features_df.columns and col not in ['user_id', 'item_id']
                ]
                if all(key in te_features_df.columns for key in ['user_id', 'item_id']):
                    # 先从 master_features_df 中移除可能已存在的旧的 TE 列 (如果重复运行或列名冲突)
                    for te_col_to_remove in current_te_feat_names:
                        if te_col_to_remove in master_features_df.columns:
                            print(f"    移除已存在的列 '{te_col_to_remove}' 以便合并新的TE特征。")
                            master_features_df = master_features_df.drop(columns=[te_col_to_remove])

                    master_features_df = pd.merge(
                        master_features_df,
                        te_features_df[list(set(cols_to_merge_te))], # 使用set去重选择列
                        on=['user_id', 'item_id'],
                        how='left',
                        suffixes=('', '_te_dup')
                    )
                    master_features_df = master_features_df[[col for col in master_features_df.columns if not col.endswith('_te_dup')]]
                    all_generated_feature_names_set.update(current_te_feat_names)
                    print(f"    成功合并来自 {gen_name_key_te} 的特征。主 DataFrame 形状: {master_features_df.shape}")
                else:
                    print(f"    警告: TargetEncodeFeatures 返回的 DataFrame 缺少 user_id 或 item_id，无法合并。")
            elif current_te_feat_names :
                 print(f"    警告: TargetEncodeFeatures 返回了空 DataFrame，但声明了特征 {current_te_feat_names}。")
            else:
                 print(f"    TargetEncodeFeatures 未产生任何特征或数据。")

        except Exception as e:
            import traceback
            print(f"    运行生成器 {gen_name_key_te} 时出错: {e}")
            traceback.print_exc()
        end_gen_time_te = time.time()
        print(f"    {gen_name_key_te} 耗时: {end_gen_time_te - start_gen_time_te:.2f}秒")


    # 5. 最终处理和保存
    print("\n  正在对特征进行最终处理...")
    unique_feature_names_to_fill = list(all_generated_feature_names_set)
    for col in unique_feature_names_to_fill:
        if col in master_features_df.columns:
             if master_features_df[col].isnull().any():
                  master_features_df[col] = master_features_df[col].fillna(0) # 简单的 NaN 填充

    # 再次确保移除重复列 (以防万一)
    master_features_df = master_features_df.loc[:,~master_features_df.columns.duplicated()]

    output_file_name = f"ranking_features_manual_{output_filename_key}.parquet"
    output_file_full_path = os.path.join(feature_output_dir, output_file_name)
    try:
        master_features_df.to_parquet(output_file_full_path, index=False)
        print(f"\n成功将所有手动特征保存至: {output_file_full_path}")
        print(f"  最终 DataFrame 形状: {master_features_df.shape}")
        print(f"  生成的唯一特征列数量: {len(unique_feature_names_to_fill)}") # 这里是集合的大小
    except Exception as e:
        print(f"保存特征 DataFrame 时出错: {e}")

    overall_end_time = time.time()
    print(f"特征工程总耗时: {overall_end_time - overall_start_time:.2f}秒")


if __name__ == "__main__":
    current_script_path = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_script_path, '..', '..'))

    fe_config_path_main = os.path.join(project_root, "conf", "features", "feature_config.yaml")

    # --- 定义执行参数 ---
    # 你可以修改这些参数来为不同的数据集生成特征
    # PREDICTION_DAY_STR = "2014-12-19" # 例如，为测试集预测12月19日
    PREDICTION_DAY_STR = "2014-12-18" # 例如，为验证集(valid_s1)或最终训练集(train_full)预测12月18日
    # PREDICTION_DAY_STR = "2014-12-17" # 例如，为训练集(train_s1)预测12月17日


    # BEHAVIOR_END_DAY_STR = (pd.to_datetime(PREDICTION_DAY_STR) - timedelta(days=1)).strftime('%Y-%m-%d')
    
    #PREDICTION_DAY_STR = "2014-12-17"
    BEHAVIOR_END_DAY_STR = "2014-12-17"
    RECALL_CANDIDATES_FILENAME = f"recall_candidates_for_{PREDICTION_DAY_STR.replace('-', '')}.parquet"
    RECALL_CANDIDATES_FILE_FULL_PATH = os.path.join(project_root, "data", "1_interim", RECALL_CANDIDATES_FILENAME)

    # 根据 PREDICTION_DAY_STR 和你的项目计划来确定 OUTPUT_KEY 和 IS_TRAINING_MODE
    if PREDICTION_DAY_STR == "2014-12-19":
        OUTPUT_KEY = f"test_target_{PREDICTION_DAY_STR.replace('-', '')}"
        IS_TRAINING_MODE = False
    elif PREDICTION_DAY_STR == "2014-12-18": # 假设12-18是验证集或最终模型的训练目标日
        # 如果这是验证集 Valid_S1
        OUTPUT_KEY = f"valid_s1_{PREDICTION_DAY_STR.replace('-', '')}"
        IS_TRAINING_MODE = False # 为验证集生成特征时，TE使用已学习的映射
        # 如果这是最终训练集 Train_Full (用于预测12-18，则 is_training_run=True)
        # OUTPUT_KEY = f"train_full_{PREDICTION_DAY_STR.replace('-', '')}"
        # IS_TRAINING_MODE = True
    elif PREDICTION_DAY_STR == "2014-12-17": # 假设12-17是训练集 Train_S1 的目标日
        OUTPUT_KEY = f"train_s1_{PREDICTION_DAY_STR.replace('-', '')}"
        IS_TRAINING_MODE = True
    else:
        print(f"错误: 未知的 PREDICTION_DAY_STR: {PREDICTION_DAY_STR}，无法确定 OUTPUT_KEY 和 IS_TRAINING_MODE。")
        exit()


    if os.path.exists(RECALL_CANDIDATES_FILE_FULL_PATH):
        generate_all_features(
            project_root_path=project_root,
            feature_engineering_config_path=fe_config_path_main,
            recall_candidates_file_path=RECALL_CANDIDATES_FILE_FULL_PATH,
            output_filename_key=OUTPUT_KEY,
            behavior_end_date_str=BEHAVIOR_END_DAY_STR,
            is_training_run=IS_TRAINING_MODE
        )
    else:
        print(f"召回候选集文件未找到，跳过特征生成: {RECALL_CANDIDATES_FILE_FULL_PATH}")
        print("请确保已成功运行召回脚本，并检查文件名和路径是否匹配。")