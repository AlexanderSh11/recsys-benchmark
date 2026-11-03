import numpy as np
import pandas as pd

from metrics.metrics import MatrixMetrics, RecsMetrics
from utils.timer import timer


def build_test_matrix(model, test):
    if not hasattr(model, "user_to_index") or not hasattr(model, "item_to_index"):
        return None
    if not hasattr(model, "train_matrix"):
        return None

    test_matrix = np.zeros_like(model.train_matrix)
    for row in test.itertuples(index=False):
        if row.userId in model.user_to_index and row.movieId in model.item_to_index:
            u = model.user_to_index[row.userId]
            i = model.item_to_index[row.movieId]
            test_matrix[u, i] = row.rating
    return test_matrix


def run_benchmark(
    model,
    train: pd.DataFrame,
    test: pd.DataFrame,
    k: int = 10,
    threshold: float = 3.5,
    n_runs: int = 3,
):

    results = []
    for _ in range(n_runs):
        
        with timer(f"Обучение {model.name}") as train_time:
            model.fit(train)

        test_users = test[model.user_id_col].unique().tolist()
        with timer(f"Выдача рекомендаций {model.name}") as pred_time:
            recs = model.recommend(test_users, n=k)

        pred_matrix = model.get_pred_matrix()
        if pred_matrix is not None:
            test_matrix = build_test_matrix(model, test)
            metrics = MatrixMetrics(pred_matrix, test_matrix)
        else:
            test_matrix = None
            metrics = RecsMetrics(recs, test)

        recall = metrics.recall_at_k(k=k, threshold=threshold)
        ndcg = metrics.ndcg_at_k(k=k)

        results.append(
            {
                "model": model,
                "metrics": metrics,
                "train_size": len(train),
                "test_size": len(test),
                "train_time": train_time["elapsed_time"],
                "pred_time": pred_time["elapsed_time"],
                "recall": recall,
                "ndcg": ndcg,
                "test_matrix": test_matrix,
            }
        )

    return results


def print_summary(results: dict):
    model = results[0]["model"]
    model.info()
    print("Среднее train_time", np.mean([r["train_time"] for r in results]))
    print("Среднее pred_time", np.mean([r["pred_time"] for r in results]))
    print("Средний recall", np.mean([r["recall"] for r in results]))
    print("Средний ndcg", np.mean([r["ndcg"] for r in results]))
    test_matrix = results[0]["test_matrix"]
    if test_matrix is not None:
        MatrixMetrics.test_matrix_info(test_matrix)
