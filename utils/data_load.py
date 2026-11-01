import pandas as pd


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    return df


def make_subset(
    df: pd.DataFrame,
    n_ratings: int,
    user_id_col: str = "userId",
    random_state: int = 42,
) -> pd.DataFrame:
    frac = n_ratings / len(df)
    parts = []
    for _, group in df.groupby(user_id_col, sort=False):
        n = max(1, int(len(group) * frac))
        sampled = group.sample(n=n, random_state=random_state)
        parts.append(sampled)
    subset = pd.concat(parts, ignore_index=True)
    return subset


def split_train_test_by_time(
    df: pd.DataFrame,
    test_frac: float = 0.2,
    user_id_col: str = "userId",
    timestamp_col: str = "timestamp",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values([user_id_col, timestamp_col]).reset_index(drop=True)

    df["rank"] = df.groupby(user_id_col).cumcount(ascending=False)
    df["n_user"] = df.groupby(user_id_col)[user_id_col].transform("size")
    df["n_test"] = (df["n_user"] * test_frac).astype(int).clip(lower=1)

    cols_to_drop = ["rank", "n_user", "n_test"]

    test = df[df["rank"] < df["n_test"]].drop(columns=cols_to_drop)
    train = df[df["rank"] >= df["n_test"]].drop(columns=cols_to_drop)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def save_df(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)
