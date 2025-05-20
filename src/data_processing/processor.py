import pandas as pd
import numpy as np
# from datetime import datetime # datetime 通常由 pandas 内部处理 pd.to_datetime
import gc # Garbage Collector
import os
import time # 用于计时

# --- 脚本配置 ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', '0_raw')
INTERIM_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', '1_interim')

# 原始文件名
USER_BEHAVIOR_FILES_CONFIG = [
    'tianchi_fresh_comp_train_user_online_partA.txt',
    'tianchi_fresh_comp_train_user_online_partB.txt'
]
ITEM_INFO_FILE_CONFIG = 'tianchi_fresh_comp_train_item_online.txt'

# 输出文件名
PROCESSED_USER_LOG_FILENAME = 'processed_user_log_p_related.parquet'
PROCESSED_ITEMS_FILENAME = 'processed_items.parquet'

# 分块大小 (根据你的内存调整，例如 500万或1000万行)
# 对于11.6亿行，32GB内存，1000万行一块，每块处理后约100MB-300MB (P相关筛选后)
# 预计P相关行为约1亿行，最终合并时可能需要 5-10GB 内存。
CHUNK_SIZE_CONFIG = 600_000_000

# 预定义数据类型 (基于之前的EDA和数据范围优化)
USER_DTYPES_CONFIG = {
    'user_id': 'int32',
    'item_id': 'int32',
    'behavior_type': 'int8',   # 1, 2, 3, 4
    'user_geohash': 'object',  # 稍后填充 'UNK'
    'item_category': 'int16',  # 约1.2万个类别，int16 (max 32767) 足够
    'time': 'object'           # 原始时间戳字符串
}
ITEM_DTYPES_CONFIG = {
    'item_id': 'int32',
    'item_geohash': 'object',  # 稍后填充 'UNK'
    'item_category': 'int16'
}

# 列名 (因为原始文件无表头)
USER_COLUMN_NAMES_CONFIG = ['user_id', 'item_id', 'behavior_type', 'user_geohash', 'item_category', 'time']
ITEM_COLUMN_NAMES_CONFIG = ['item_id', 'item_geohash', 'item_category']

# --- 辅助函数 ---
def print_memory_usage(df, df_name="DataFrame"):
    """打印DataFrame的内存占用情况"""
    mem_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    print(f"  {df_name} 内存占用: {mem_mb:.2f} MB, 形状: {df.shape}")

# --- 主处理函数 ---
def process_items(raw_path, interim_path, item_file, item_dtypes, item_columns):
    """
    加载、清洗和保存商品子集 P 的数据。
    返回商品子集 P 中的唯一 item_id 集合。
    """
    print(f"\n--- 步骤 1: 开始处理商品数据 ({item_file}) ---")
    start_time = time.time()
    file_path = os.path.join(raw_path, item_file)

    if not os.path.exists(file_path):
        print(f"  错误: 商品数据文件未找到: {file_path}")
        return set()

    try:
        df_items = pd.read_csv(
            file_path,
            sep='\t',  #
            dtype=item_dtypes,
            header=None,
            names=item_columns,
            na_filter=True, # Pandas 默认会将空字符串识别为 NaN
            keep_default_na=True
        )
        print(f"  原始商品数据加载完毕.")
        print_memory_usage(df_items, "原始 df_items")

        # 清洗1: 处理 item_geohash 缺失值 (原始为空白，被Pandas读为NaN)
        if 'item_geohash' in df_items.columns:
            df_items['item_geohash'].fillna('UNK', inplace=True)
            print(f"  商品 'item_geohash' 中的 NaN 已填充为 'UNK'.")
        else:
            print("  警告: 'item_geohash' 列在商品数据中未找到 (根据列名配置).")


        # 清洗2: (可选) 去除完全重复的行 (item_id, item_geohash, item_category都相同)
        # 通常不需要，因为 item_id 对不同 geohash 是有效的多条记录
        # initial_rows = len(df_items)
        # df_items.drop_duplicates(subset=item_columns, keep='first', inplace=True)
        # if len(df_items) < initial_rows:
        #     print(f"  商品数据去重: {initial_rows - len(df_items)} 行被移除.")

        # 保存处理后的商品数据
        output_file_path = os.path.join(interim_path, PROCESSED_ITEMS_FILENAME)
        df_items.to_parquet(output_file_path, index=False, engine='pyarrow')
        print(f"  处理后的商品数据已保存到: {output_file_path}")
        print_memory_usage(df_items, "处理后 df_items")

        unique_p_item_ids = set(df_items['item_id'].unique())
        print(f"  商品子集 P 中共有 {len(unique_p_item_ids)} 个独立商品 ID.")
        duration = time.time() - start_time
        print(f"--- 商品数据处理完成, 耗时: {duration:.2f} 秒 ---")
        return unique_p_item_ids

    except Exception as e:
        print(f"  处理商品数据时发生严重错误: {e}")
        return set()

def process_user_behavior(raw_path, interim_path, user_files, user_dtypes, user_columns,
                          p_item_ids_set, chunk_size):
    """
    分块加载、清洗、筛选并保存用户行为数据。
    只保留与商品子集 P 相关的行为。
    """
    print(f"\n--- 步骤 2: 开始处理用户行为数据 (分块大小: {chunk_size}) ---")
    start_time = time.time()

    if not p_item_ids_set:
        print("  错误: 商品子集 P 为空，无法进行用户行为筛选。跳过用户行为数据处理。")
        return

    all_processed_chunks = []
    total_original_rows = 0
    total_p_related_rows = 0
    processed_files_count = 0

    for file_idx, user_file in enumerate(user_files):
        print(f"  正在处理文件: {user_file} ({file_idx + 1}/{len(user_files)})")
        file_path = os.path.join(raw_path, user_file)

        if not os.path.exists(file_path):
            print(f"    警告: 用户行为文件未找到: {file_path}. 跳过此文件.")
            continue
        processed_files_count +=1

        try:
            chunk_iterator = pd.read_csv(
                file_path,
                sep='\t', 
                chunksize=chunk_size,
                dtype=user_dtypes,
                header=None,
                names=user_columns,
                na_filter=True,
                keep_default_na=True
            )

            for chunk_idx, df_chunk_raw in enumerate(chunk_iterator):
                chunk_start_time = time.time()
                print(f"    处理文件 '{user_file}', 块 {chunk_idx + 1}...")
                current_chunk_original_rows = len(df_chunk_raw)
                total_original_rows += current_chunk_original_rows
                print_memory_usage(df_chunk_raw, "原始 df_chunk")

                # 预处理1: 转换时间列 (必须！)
                df_chunk_raw['datetime'] = pd.to_datetime(df_chunk_raw['time'], format='%Y-%m-%d %H')
                # 可以考虑删除原始 'time' 列以节省内存，但datetime对象包含原始信息
                # df_chunk_raw.drop(columns=['time'], inplace=True, errors='ignore')

                # 核心步骤: 筛选与商品子集 P 相关的行为
                df_chunk_p_related = df_chunk_raw[df_chunk_raw['item_id'].isin(p_item_ids_set)].copy()
                # 使用 .copy() 避免 SettingWithCopyWarning，并确保后续操作在副本上进行
                del df_chunk_raw # 及时释放原始块的内存
                gc.collect()

                current_chunk_p_related_rows = len(df_chunk_p_related)
                total_p_related_rows += current_chunk_p_related_rows

                if current_chunk_p_related_rows > 0:
                    # 清洗1: 处理 user_geohash 缺失值 (在筛选后的数据上)
                    df_chunk_p_related.loc[:, 'user_geohash'] = df_chunk_p_related['user_geohash'].fillna('UNK')

                    # 清洗2: (可选，且需谨慎) 处理重复的用户行为记录
                    # 比赛说明购买行为的重复是不同订单，浏览行为重复常见
                    # 暂时不执行严格去重，除非后续分析表明有必要
                    # df_chunk_p_related.drop_duplicates(subset=[c for c in user_columns if c != 'time'] + ['datetime'], keep='first', inplace=True)

                    # 添加到列表以备合并
                    all_processed_chunks.append(df_chunk_p_related)
                    print_memory_usage(df_chunk_p_related, "处理后 df_chunk_p_related")
                else:
                    del df_chunk_p_related # 如果筛选后为空，也释放内存
                    gc.collect()


                chunk_duration = time.time() - chunk_start_time
                print(f"      块 {chunk_idx + 1}: 原始行={current_chunk_original_rows}, P相关行={current_chunk_p_related_rows}. 耗时: {chunk_duration:.2f} 秒.")

        except Exception as e:
            print(f"    处理文件 '{user_file}' 时发生严重错误: {e}")
            continue # 继续处理下一个文件或块

    if not processed_files_count:
        print("  未处理任何用户行为文件（可能文件路径错误）。")
        return
        
    if not all_processed_chunks:
        print("  未收集到任何与 P 相关的用户行为数据。请检查筛选逻辑或原始数据。")
        duration = time.time() - start_time
        print(f"--- 用户行为数据处理完成 (无数据输出), 耗时: {duration:.2f} 秒 ---")
        return

    # 合并所有处理过的、与P相关的块
    print("\n  开始合并所有处理后的用户行为数据块...")
    merge_start_time = time.time()
    df_user_final = pd.concat(all_processed_chunks, ignore_index=True)
    del all_processed_chunks # 释放列表内存
    gc.collect()
    merge_duration = time.time() - merge_start_time
    print(f"  数据块合并完成. 耗时: {merge_duration:.2f} 秒.")
    print_memory_usage(df_user_final, "最终合并的 df_user_final (P相关)")

    # 保存最终的用户行为数据 (P相关)
    # 在保存前可以确认删除原始 time 列，只保留 datetime
    if 'time' in df_user_final.columns:
         df_user_final.drop(columns=['time'], inplace=True)
         print("  已从最终用户行为数据中移除原始 'time' 列.")

    output_file_path = os.path.join(interim_path, PROCESSED_USER_LOG_FILENAME)
    df_user_final.to_parquet(output_file_path, index=False, engine='pyarrow')
    print(f"  处理后的用户行为数据 (P相关) 已保存到: {output_file_path}")

    print(f"\n  用户行为数据处理总结:")
    print(f"    总原始行数处理: {total_original_rows}")
    print(f"    总P相关行为行数保留: {total_p_related_rows}")
    if total_original_rows > 0:
        print(f"    P相关行为占比: {total_p_related_rows / total_original_rows:.2%}")

    duration = time.time() - start_time
    print(f"--- 用户行为数据处理完成, 耗时: {duration:.2f} 秒 ---")


# --- 程序入口 ---
if __name__ == '__main__':
    print("====== 开始数据预处理模块 ======")
    print("1111111111111111111111111111111")
    overall_start_time = time.time()

    # 确保 interim 目录存在
    if not os.path.exists(INTERIM_DATA_PATH):
        try:
            os.makedirs(INTERIM_DATA_PATH)
            print(f"目录已创建: {INTERIM_DATA_PATH}")
        except OSError as e:
            print(f"创建目录 {INTERIM_DATA_PATH} 失败: {e}")
            exit() # 如果无法创建输出目录，则退出

    # 步骤1: 处理商品数据
    p_item_ids = process_items(
        RAW_DATA_PATH, INTERIM_DATA_PATH, ITEM_INFO_FILE_CONFIG,
        ITEM_DTYPES_CONFIG, ITEM_COLUMN_NAMES_CONFIG
    )

    # 步骤2: 处理用户行为数据
    process_user_behavior(
        RAW_DATA_PATH, INTERIM_DATA_PATH, USER_BEHAVIOR_FILES_CONFIG,
        USER_DTYPES_CONFIG, USER_COLUMN_NAMES_CONFIG,
        p_item_ids, CHUNK_SIZE_CONFIG
    )

    overall_duration = time.time() - overall_start_time
    print(f"\n====== 数据预处理模块全部完成, 总耗时: {overall_duration:.2f} 秒 ======")
