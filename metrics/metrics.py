import numpy as np


def recall_at_k(
    pred: np.ndarray, test_matrix: np.ndarray, k: int = 10, threshold: float = 3.5
) -> float:
    recalls = []

    for u in range(pred.shape[0]):
        relevant = set(np.where(test_matrix[u] >= threshold)[0])
        if not relevant:
            continue
        top_k = set(np.argsort(pred[u])[::-1][:k])
        recalls.append(len(relevant & top_k)) / len(relevant)
    return float(np.mean(recalls)) if recalls else 0.0


def ndcg_at_k(pred: np.ndarray, test_matrix: np.ndarray, k: int = 10) -> float:
    ndcgs = []
    for u in range(pred.shape[0]):
        top_k = np.argsort(pred[u])[::-1][:k]
        gains = test_matrix[u][top_k]
        discounts = np.log2(np.arange(2, k + 2))
        dcg = np.sum(gains / discounts)
        ideal = np.sort(test_matrix[u][test_matrix[u] > 0])[::-1][:k]
        idcg = np.sum(ideal / discounts)
        if idcg > 0:
            ndcgs.append(dcg / idcg)
    return float(np.mean(ndcgs)) if ndcgs else 0.0
