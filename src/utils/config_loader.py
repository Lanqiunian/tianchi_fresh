# src/utils/config_loader.py
import yaml
import os

def load_config(config_file_path: str) -> dict:
    """
    加载 YAML 配置文件。

    Args:
        config_file_path (str): YAML 配置文件的路径。

    Returns:
        dict: 从 YAML 文件加载的配置数据。

    Raises:
        FileNotFoundError: 如果配置文件未找到。
        yaml.YAMLError: 如果解析 YAML 文件时出错。
    """
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"配置文件未找到: {config_file_path}")

    try:
        with open(config_file_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        if config_data is None: # 如果文件为空或只包含注释
            return {}
        return config_data
    except yaml.YAMLError as e:
        # 可以加入更详细的错误日志或处理
        print(f"解析 YAML 文件时出错 '{config_file_path}': {e}")
        raise # 重新抛出异常，让调用者处理
    except Exception as e:
        print(f"加载配置文件 '{config_file_path}' 时发生未知错误: {e}")
        raise

if __name__ == '__main__':
    # 模块的简单测试 (可选)
    # 假设项目根目录下有 conf/recall/recall_config.yaml
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    test_config_path = os.path.join(project_root, 'conf', 'recall', 'recall_config.yaml')
    print(f"正在尝试加载测试配置文件: {test_config_path}")
    try:
        config = load_config(test_config_path)
        print("配置文件加载成功:")
        # print(config) # 取消注释以打印整个配置
        if 'global_settings' in config:
            print("\n全局设置:")
            print(config['global_settings'])
        if 'strategies' in config:
            print("\n策略 (部分):")
            for strategy_name, strategy_conf in list(config['strategies'].items())[:2]: #只打印前两个策略
                 print(f"  {strategy_name}: {strategy_conf.get('enabled')}")
    except Exception as e:
        print(f"测试加载失败: {e}")