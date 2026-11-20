import pandas as pd
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix

from algorithms.base import BaseRecommender


class ImplicitALS(BaseRecommender):
    name = "implicit ALS"

    def __init__(
        self,
        factors: int = 64,
        regularization: float = 0.1,
        iterations: int = 15,
        alpha: float = 40.0,
        random_state: int = 42,
    ):
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self.random_state = random_state

        self.params = {
            "factors": self.factors,
            "iterations": self.iterations,
            "regularization": self.regularization,
            "alpha": self.alpha,
        }

        self.model = None
        self.user_items_sparse = None

        self.user_to_index = None
        self.item_to_index = None
        self.index_to_item = None

        self.n_users = None
        self.n_items = None
        self.n_ratings = None

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
        self.n_ratings = len(train_df)

        user_idxs = train_df[self.user_id_col].map(self.user_to_index).values
        item_idxs = train_df[self.movie_id_col].map(self.item_to_index).values

        ratings = train_df[self.rating_col].values

        self.user_items_sparse = csr_matrix(
            (ratings * self.alpha, (user_idxs, item_idxs)),
            shape=(self.n_users, self.n_items),
        )
        item_users_sparse = self.user_items_sparse.tocsr()

        self.model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=self.random_state,
        )
        self.model.fit(item_users_sparse, show_progress=False)

    def recommend(self, user_ids: list, n=10) -> dict[int, list[int]]:
        if self.model is None:
            raise RuntimeError("Модель не была обучена")

        result = {}
        for uid in user_ids:
            if uid not in self.user_to_index:
                result[uid] = []
                continue

            u = self.user_to_index[uid]

            ids, _ = self.model.recommend(
                userid=u,
                user_items=self.user_items_sparse[u],
                N=n,
                filter_already_liked_items=True,
            )

            result[uid] = [self.index_to_item[i] for i in ids]

        return result

    def get_pred_matrix(self):
        return None

    def info(self):
        print(
            f"Модель: {self.name} (factors={self.factors}, iterations={self.iterations}, regularization={self.regularization}, alpha={self.alpha})"
        )
        sparsity = (
            self.n_ratings / (self.n_users * self.n_items) * 100
            if self.n_users and self.n_items
            else 0
        )
        print(f"Количество пользователей: {self.n_users}")
        print(f"Количество объектов: {self.n_items}")
        print(f"Количество оценок в train: {self.n_ratings}")
        print(f"Разреженность матрицы взаимодействий: {sparsity:.4f}%")
