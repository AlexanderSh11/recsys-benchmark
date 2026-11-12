import pandas as pd


def numeric_col_stats(df: pd.DataFrame, col: str) -> pd.DataFrame:
    col_data = df[col]

    return pd.DataFrame({
        f"{col}_mean": [col_data.mean()],
        f"{col}_std": [col_data.std()],
        f"{col}_min": [col_data.min()],
        f"{col}_max": [col_data.max()],
        f"{col}_median": [col_data.median()],
    })


def dataframe_stats(df: pd.DataFrame, user_id_col: str = "userId", item_id_col: str = "movieId", rating_col: str = "rating", cold_start_threshold: int = 5, threshold: float = 3.5) -> pd.DataFrame:
    n_users = df[user_id_col].nunique()
    n_items = df[item_id_col].nunique()
    n_ratings = len(df)

    density = n_ratings / (n_users * n_items)

    user_counts = df[user_id_col].value_counts()
    item_counts = df[item_id_col].value_counts()

    results = pd.DataFrame({
        "n_users": [n_users],
        "n_items": [n_items],
        "n_ratings": [n_ratings],
        "density": [density],
        "sparsity": [1 - density],
        "user_ratings_median": [user_counts.median()],
        "user_ratings_p95": [user_counts.quantile(0.95)],
        "user_ratings_max": [user_counts.max()],
        "item_ratings_median": [item_counts.median()],
        "item_ratings_p95": [item_counts.quantile(0.95)],
        "item_ratings_max": [item_counts.max()],
        f"frac_users_lt_{cold_start_threshold}": [(user_counts < cold_start_threshold).mean()],
        f"frac_items_lt_{cold_start_threshold}": [(item_counts < cold_start_threshold).mean()],
        f"frac_ratings_ge_{threshold}": [(df[rating_col] >= threshold).mean()],
        "df_memory_mb": [df.memory_usage(deep=True).sum() / 1024**2],
    })

    results = pd.concat([results, numeric_col_stats(df, col=rating_col)], axis=1)

    return results