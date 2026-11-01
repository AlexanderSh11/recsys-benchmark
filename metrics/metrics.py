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
