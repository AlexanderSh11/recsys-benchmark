import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

from algorithms.base import BaseRecommender


class UserBasedCF(BaseRecommender):
    name = "UserBasedCF"

    def __init__(
        self,
        top_k: int = 100,
    ):
        self.top_k = top_k
        self.user_similarity = None
        self.train_matrix = None
        self.n_users = 0
        self.n_items = 0
        self.user_to_index = None
        self.item_to_index = None
        self.index_to_item = None
        self._pred = None
        self.rating_for_train_items = -1e9

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
        self._pred = None
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
            pred[u][self.train_matrix[u] > 0] = self.rating_for_train_items
        return pred

    def get_pred_matrix(self):
        if self._pred is None:
            self._pred = self.predict()
        return self._pred

    def recommend(self, user_ids: list, n: int = 10) -> dict[int, list[int]]:
        if self.train_matrix is None:
            raise RuntimeError("Модель не была обучена")

        if not hasattr(self, "_pred") or self._pred is None:
            self._pred = self.predict()

        result = {}
        for uid in user_ids:
            if uid not in self.user_to_index:
                continue
            u = self.user_to_index[uid]
            scores = self._pred[u]

            k = min(n, int((scores > self.rating_for_train_items).sum()))
            if k == 0:
                result[uid] = []
                continue

            top_k = np.argpartition(-scores, k - 1)[:k]
            top_k = top_k[np.argsort(-scores[top_k])]

            result[uid] = [self.index_to_item[i] for i in top_k]
        return result

    def info(self):
        print(self.name)
        n_users = self.n_users
        n_items = self.n_items
        n_ratings = int((self.train_matrix > 0).sum())
        sparsity = n_ratings / (n_users * n_items) * 100 if n_users and n_items else 0

        print(f"Количество пользователей: {n_users:,}")
        print(f"Количество объектов: {n_items:,}")
        print(f"Количество оценок в train: {n_ratings:,}")
        print(f"Разреженность матрицы оценок: {sparsity:.4f}%")
        print(f"Размер матрицы схожестей: {n_users**2 * 4 / 1024**3:.2f} ГБ")
