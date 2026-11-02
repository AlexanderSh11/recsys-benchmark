from abc import ABC, abstractmethod

import pandas as pd


class BaseRecommender(ABC):
    name: str = "base"
    user_id_col: str = "userId"
    movie_id_col: str = "movieId"
    rating_col: str = "rating"

    @abstractmethod
    def fit(self, train_df: pd.DataFrame) -> "BaseRecommender":
        """Обучает модель на train_df с колонками userId, movieId, rating"""
        ...

    @abstractmethod
    def recommend(self, user_ids: list, n: int = 10) -> dict[int, list[int]]:
        """Возвращает top-N рекомендаций в виде {user_id: [item_id, ...]}"""
        ...

    @abstractmethod
    def info(self):
        """Возвращает информацию о модели"""
        ...
