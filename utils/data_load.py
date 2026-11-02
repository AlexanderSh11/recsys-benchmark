import numpy as np
import pandas as pd


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    return df


def make_subset(df, n_ratings, user_id_col="userId", min_ratings=20, random_state=42):
    if n_ratings >= len(df):
        return df.reset_index(drop=True)

    counts = df.groupby(user_id_col).size()
    active = counts[counts >= min_ratings].index.to_numpy()

    rng = np.random.default_rng(random_state)
    active = rng.permutation(active)

    user_counts = counts.loc[active].to_numpy()
    cumsum = user_counts.cumsum()
    n_users = int(np.searchsorted(cumsum, n_ratings)) + 1
    chosen_users = active[:n_users]

    subset = df[df[user_id_col].isin(chosen_users)]
    return subset.reset_index(drop=True)


def split_train_test_by_time(
    df: pd.DataFrame,
    test_frac: float = 0.2,
    user_id_col: str = "userId",
    timestamp_col: str = "timestamp",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values([user_id_col, timestamp_col]).reset_index(drop=True)
    
    df["rank"] = df.groupby(user_id_col).cumcount(ascending=False)
    df["n_user"] = df.groupby(user_id_col)[user_id_col].transform("size")
    df["n_test"] = (df["n_user"] * test_frac).round().astype(int).clip(lower=1)
    
    df.loc[df["n_user"] < 2, "n_test"] = 0
    
    test = df[df["rank"] < df["n_test"]].drop(columns=["rank", "n_user", "n_test"])
    train = df[df["rank"] >= df["n_test"]].drop(columns=["rank", "n_user", "n_test"])
    return train.reset_index(drop=True), test.reset_index(drop=True)


def save_df(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)
