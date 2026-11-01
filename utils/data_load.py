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
    subset = df.groupby(user_id_col, group_keys=False).apply(
        lambda group: group.sample(frac=frac, random_state=random_state)
    )
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
    df["n_test"] = (df["n_user"] * test_frac).astype(int).clip(lower=1)

    test = df[df["rank"] < df["n_test"]]
    train = df[df["rank"] >= df["n_test"]]
    return train, test
