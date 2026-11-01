import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances


class UserBasedCF:
    def __init__(
        self,
        top_k: int = 100,
        user_id_col: str = "userId",
        movie_id_col: str = "movieId",
    ):
        self.top_k = top_k
        self.user_similarity = None
        self.train_matrix = None
        self.n_users = 0
        self.n_items = 0
        self.user_to_index = None
        self.item_to_index = None
        self.index_to_item = None
        self.user_id_col = user_id_col
        self.movie_id_col = movie_id_col

    def fit(self, train_df: pd.DataFrame):
        self.user_to_index = {
            uid: i for i, uid in enumerate(train_df[self.user_id_col].unique())
        }
        self.item_to_index = {
            iid: i for i, iid in enumerate(train_df[self.movie_id_col].unique())
        }
        self.index_to_item = {v: k for k, v in self.item_to_index.items()}
        self.n_users = len(self.user_to_index)
        self.n_items = len(self.item_to_index)

        self.train_matrix = np.zeros((self.n_users, self.n_items), dtype=np.float32)
        for row in train_df.itertuples(index=False):
            u = self.user_to_index[row.userId]
            i = self.item_to_index[row.movieId]
            self.train_matrix[u, i] = row.rating

        self.user_similarity = 1 - pairwise_distances(
            self.train_matrix, metric="cosine"
        )
        return self

    def predict(self, top_k=None) -> np.ndarray:
        """Взвешенное среднее с нормализацией (k_fract_mean из films_app)"""
        top_k = top_k or self.top_k
        user_means = np.zeros(self.n_users, dtype=np.float32)
        for u in range(self.n_users):
            rated = self.train_matrix[u] > 0
            if rated.any():
                user_means[u] = self.train_matrix[u][rated].mean()

        pred = np.zeros((self.n_users, self.n_items), dtype=np.float32)
        for u in range(self.n_users):
            sim_users = self.user_similarity[u].argsort()[::-1][1 : top_k + 1]
            sims = self.user_similarity[u][sim_users]
            weighted = np.zeros(self.n_items, dtype=np.float32)
            sim_sum = np.zeros(self.n_items, dtype=np.float32)
            for j, v in enumerate(sim_users):
                r = self.train_matrix[v]
                mask = r > 0
                weighted[mask] += sims[j] * (r[mask] - user_means[v])
                sim_sum[mask] += abs(sims[j])
            pred[u] = np.where(
                sim_sum > 0,
                user_means[u] + weighted / np.maximum(sim_sum, 1e-9),
                user_means[u],
            )
            pred[u] = np.clip(pred[u], 0, 10)
        return pred
