# src/rank/utils_rank.py
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

def calculate_f1_metrics(y_true, y_pred_proba, threshold):
    """
    根据给定的阈值计算精确率、召回率和F1分数。
    """
    y_pred_binary = (y_pred_proba >= threshold).astype(int)

    precision = precision_score(y_true, y_pred_binary, zero_division=0)
    recall = recall_score(y_true, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)

    return precision, recall, f1

def find_best_f1_threshold(y_true, y_pred_proba, start=0.01, end=0.5, step=0.005):
    """
    在指定范围内搜索最佳F1阈值。
    """
    best_f1 = -1.0
    best_threshold = -1.0
    best_precision = -1.0
    best_recall = -1.0

    print(f"  开始搜索最佳F1阈值 (范围: {start}-{end}, 步长: {step})...")
    thresholds_to_try = []
    current_t = start
    while current_t <= end:
        thresholds_to_try.append(current_t)
        current_t += step

    for threshold in thresholds_to_try:
        precision, recall, f1 = calculate_f1_metrics(y_true, y_pred_proba, threshold)
        # print(f"    尝试阈值: {threshold:.4f} -> P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_precision = precision
            best_recall = recall

    print(f"  搜索完成。最佳F1: {best_f1:.4f} (在阈值 {best_threshold:.4f} 时取得)")
    print(f"    对应的 Precision: {best_precision:.4f}, Recall: {best_recall:.4f}")
    return best_threshold, best_f1, best_precision, best_recall


def get_label(user_log_df: pd.DataFrame, candidates_df: pd.DataFrame, prediction_date_str: str) -> pd.DataFrame:
    """
    为候选集 (user_id, item_id) 生成标签。
    标签为1表示用户在 prediction_date_str 当天购买了该商品，否则为0。
    """
    if 'datetime' not in user_log_df.columns:
        raise ValueError("user_log_df 必须包含 'datetime' 列。")
    if not pd.api.types.is_datetime64_any_dtype(user_log_df['datetime']):
        user_log_df['datetime'] = pd.to_datetime(user_log_df['datetime'])

    # 创建 candidates_df 的副本进行操作，以避免修改原始传入的 DataFrame
    labeled_candidates_df = candidates_df.copy()

    # 如果 'label' 列已存在于输入的 candidates_df 中（例如来自特征工程的TE步骤），
    # 我们将用新生成的标签覆盖它。为避免列名冲突，先将其移除。
    if 'label' in labeled_candidates_df.columns:
        print(f"  信息: 输入的 candidates_df (特征文件) 中已存在 'label' 列。它将被基于 '{prediction_date_str}' 的新标签替换。")
        labeled_candidates_df = labeled_candidates_df.drop(columns=['label'])

    # 筛选出预测日当天的购买行为
    purchases_on_pred_date = user_log_df[
        (user_log_df['datetime'].dt.strftime('%Y-%m-%d') == prediction_date_str) &
        (user_log_df['behavior_type'] == 4) # 4 代表购买
    ][['user_id', 'item_id']].drop_duplicates()

    if purchases_on_pred_date.empty:
        print(f"警告: 在日期 {prediction_date_str} 没有找到任何购买记录。所有标签将为0。")
        labeled_candidates_df['label'] = 0 # 直接为副本创建新的 'label' 列
    else:
        # 为购买记录创建一个临时且唯一的标签列名，以避免与 candidates_df 中可能存在的列冲突
        purchases_on_pred_date['__temp_label_from_purchase__'] = 1
        
        labeled_candidates_df = pd.merge(labeled_candidates_df, 
                                         purchases_on_pred_date[['user_id', 'item_id', '__temp_label_from_purchase__']],
                                         on=['user_id', 'item_id'],
                                         how='left')
        
        # 将临时标签列的值赋给最终的 'label' 列，并将 NaN（未购买）填充为0
        labeled_candidates_df['label'] = labeled_candidates_df['__temp_label_from_purchase__'].fillna(0).astype(int)
        # 移除临时列
        labeled_candidates_df = labeled_candidates_df.drop(columns=['__temp_label_from_purchase__'])

    print(f"  为日期 {prediction_date_str} 生成标签完成。正样本数: {labeled_candidates_df['label'].sum()} / 总候选数: {len(labeled_candidates_df)}")
    return labeled_candidates_df