from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseMetrics(ABC):
    @abstractmethod
    def recall_at_k(self, k: int, threshold: float) -> float: ...

    @abstractmethod
    def ndcg_at_k(self, k: int) -> float: ...


class RecsMetrics(BaseMetrics):
    def __init__(
        self,
        recs: dict,
        test_df: pd.DataFrame,
        user_id_col: str = "userId",
        movie_id_col: str = "movieId",
        rating_col: str = "rating",
    ):
        self.recs = recs
        self.test_df = test_df
        self.user_id_col = user_id_col
        self.movie_id_col = movie_id_col
        self.rating_col = rating_col

    def recall_at_k(self, k: int = 10, threshold: float = 3.5) -> float:
        recalls = []

        recs = self.recs
        test_df = self.test_df

        for uid, items in recs.items():
            user_test = test_df[test_df[self.user_id_col] == uid]
            relevant = set(
                user_test[user_test[self.rating_col] >= threshold][self.movie_id_col]
            )
            if not relevant:
                continue
            top_k = set(items[:k])
            recalls.append(len(relevant & top_k) / len(relevant))
        return float(np.mean(recalls)) if recalls else 0.0

    def ndcg_at_k(self, k: int = 10) -> float:
        ndcgs = []
        discounts = np.log2(np.arange(2, k + 2))

        recs = self.recs
        test_df = self.test_df

        for uid, items in recs.items():
            user_test = test_df[test_df[self.user_id_col] == uid]

            if user_test.empty:
                continue

            ratings = dict(
                zip(user_test[self.movie_id_col], user_test[self.rating_col])
            )
            gains = np.array([ratings.get(m, 0) for m in items[:k]])
            ideal = np.sort(np.array(list(ratings.values())))[::-1][:k]
            ideal = np.pad(ideal, (0, k - len(ideal)), constant_values=0)

            dcg = np.sum(gains / discounts)
            idcg = np.sum(ideal / discounts)
            if idcg > 0:
                ndcgs.append(dcg / idcg)

        return float(np.mean(ndcgs)) if ndcgs else 0.0
