# src/recall/factory.py
import yaml
import importlib
import os
import pandas as pd
from src.recall.recall_strategy import RecallStrategy

class RecallStrategyFactory:
    @staticmethod
    def get_active_strategies(
        config_file_path: str,
        user_log_grouped_global: dict, 
        items_df_global: pd.DataFrame,
        precomputed_popular_items: pd.Series = None,
        # 这个参数是为ItemCF特殊准备的，包含 {user_id: [item_id_hist1, item_id_hist2,...]}
        user_preprocessed_itemcf_histories: dict = None 
    ) -> list[RecallStrategy]:
        if not os.path.exists(config_file_path):
            raise FileNotFoundError(f"Recall configuration file not found: {config_file_path}")

        with open(config_file_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)

        active_strategies = []
        processed_data_path_cfg = config_data.get("global_settings", {}).get("processed_data_path", "data/1_interim/")
        project_root_temp = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        processed_data_path_abs = os.path.join(project_root_temp, processed_data_path_cfg)

        for strategy_name, strategy_config in config_data.get("strategies", {}).items():
            if strategy_config.get("enabled", False):
                try:
                    module_path = strategy_config["module"]
                    class_name = strategy_config["class"]
                    strategy_params = strategy_config.get("params", {})

                    module = importlib.import_module(module_path)
                    StrategyClass = getattr(module, class_name)

                    init_kwargs = {
                        "strategy_name": strategy_name,
                        "processed_data_path": processed_data_path_abs,
                        "strategy_specific_config": strategy_params,
                        "user_log_grouped": user_log_grouped_global, # 给通用策略
                        "items_df_global": items_df_global.copy() if items_df_global is not None else pd.DataFrame(),
                    }

                    if strategy_name == "global_popular_items" and precomputed_popular_items is not None:
                        init_kwargs["precomputed_popular_items_series"] = precomputed_popular_items
                    
                    # 特殊处理ItemCF，传递专属的预处理历史
                    if strategy_name == "item_cf" and user_preprocessed_itemcf_histories is not None:
                        init_kwargs["user_specific_preprocessed_history"] = user_preprocessed_itemcf_histories
                    
                    instance = StrategyClass(**init_kwargs)
                    active_strategies.append(instance)
                    print(f"Successfully loaded and instantiated strategy: '{strategy_name}'")
                except ImportError as e:
                    print(f"Error importing module/class for strategy '{strategy_name}': {e}")
                except AttributeError as e:
                    print(f"Error: Class '{class_name}' not found in module '{module_path}' for strategy '{strategy_name}': {e}")
                except Exception as e:
                    import traceback
                    print(f"Error instantiating strategy '{strategy_name}': {e}")
                    traceback.print_exc()
            else:
                print(f"Strategy '{strategy_name}' is disabled.")
        return active_strategies