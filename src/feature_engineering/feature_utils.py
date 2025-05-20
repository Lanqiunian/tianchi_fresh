# src/feature_engineering/feature_utils.py
import numpy as np
import pandas as pd

def time_decay_weight(delta_hours: pd.Series, lambda_decay: float = 0.01) -> pd.Series:
    """计算指数时间衰减权重 (向量化版本)。"""
    weights = pd.Series(np.nan, index=delta_hours.index, dtype=float)
    valid_mask = pd.notna(delta_hours) & (delta_hours >= 0)
    if valid_mask.any():
        weights.loc[valid_mask] = np.exp(-lambda_decay * delta_hours[valid_mask])
    weights.fillna(0.0, inplace=True)
    return weights

def safe_division(numerator: pd.Series, denominator: pd.Series, default_val: float = 0.0) -> pd.Series:
    """
    安全除法，支持 Pandas Series 输入。
    如果分母为零或任意操作数为NaN，则对应位置返回 default_val。
    """
    # 创建一个布尔 Series，标记哪些位置可以进行安全除法
    # 条件：分母不为0，且分子和分母都不是NaN
    # np.isclose is better for float comparisons than == 0.0
    valid_mask = ~np.isclose(denominator, 0.0) & pd.notna(numerator) & pd.notna(denominator)

    # 初始化结果 Series，默认为 default_val
    result = pd.Series(default_val, index=numerator.index, dtype=float)

    # 对有效的部分进行除法
    if valid_mask.any():
        result.loc[valid_mask] = numerator.loc[valid_mask] / denominator.loc[valid_mask]
    
    # 对于除法结果可能再次产生NaN的情况 (例如 0/0 经过了isclose但后续pandas除法仍给NaN)
    # 或其他意外的NaN，再次填充
    result.fillna(default_val, inplace=True)
    return result

def get_time_since_last(series_datetime: pd.Series, current_time: pd.Timestamp, default_hours_if_empty: float = 30*24) -> float:
    """计算距离序列中最后一次事件的时间（小时）。"""
    if series_datetime.empty or series_datetime.isna().all():
        return default_hours_if_empty
    last_event_time = series_datetime.max()
    if pd.isna(last_event_time):
        return default_hours_if_empty
    return (current_time - last_event_time).total_seconds() / 3600

def parse_behavior_end_date(date_str: str) -> pd.Timestamp:
    """解析 behavior_end_date_str 并将时间设置为当天结束。"""
    return pd.to_datetime(date_str) + pd.Timedelta(hours=23, minutes=59, seconds=59)

def split_into_sessions(user_log_df: pd.DataFrame, session_timeout_minutes: int = 30,
                        user_col='user_id', time_col='datetime') -> pd.DataFrame:
    """
    将用户行为日志按时间切分成 session。
    返回带有 'session_id' 列的 DataFrame。
    session_id 是用户内的 session 编号 (user_id, session_rank)。
    """
    if user_log_df.empty:
        user_log_df['session_id'] = pd.Series(dtype='object') # or int, consistent with cumsum
        return user_log_df

    # Ensure a copy is made if user_log_df is a slice
    df = user_log_df.copy()
    df = df.sort_values(by=[user_col, time_col])
    
    # Calculate time difference within each group
    df['time_diff_to_prev_minutes'] = df.groupby(user_col)[time_col].diff().dt.total_seconds() / 60
    
    df['new_session_flag'] = (df['time_diff_to_prev_minutes'].isnull()) | \
                             (df['time_diff_to_prev_minutes'] > session_timeout_minutes)
    
    # Create session_id by cumsumming the new_session_flag within each user group
    df['session_id'] = df.groupby(user_col)['new_session_flag'].cumsum()
    
    df.drop(columns=['time_diff_to_prev_minutes', 'new_session_flag'], inplace=True)
    return df