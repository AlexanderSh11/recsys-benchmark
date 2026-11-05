import gc
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from metrics.metrics import MatrixMetrics, RecsMetrics
from utils.memory_usage import memory_usage
from utils.timer import timer

FIGSIZE = (8, 6)


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
    model_factory,
    train: pd.DataFrame,
    test: pd.DataFrame,
    k: int = 10,
    threshold: float = 3.5,
    n_runs: int = 3,
):

    results = []
    for _ in range(n_runs):
        model = model_factory()

        with (
            memory_usage(f"Обучение {model.label()}") as train_mem,
            timer(f"Обучение {model.label()}") as train_time,
        ):
            model.fit(train)

        test_users = test[model.user_id_col].unique().tolist()

        with (
            memory_usage(f"Выдача рекомендаций {model.label()}") as pred_mem,
            timer(f"Выдача рекомендаций {model.label()}") as pred_time,
        ):
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
                "model": model.name,
                "train_size": len(train),
                "test_size": len(test),
                "train_time": train_time["elapsed_time"],
                "pred_time": pred_time["elapsed_time"],
                "train_mem_delta": train_mem["delta_memory_usage"],
                "pred_mem_delta": pred_mem["delta_memory_usage"],
                "recall": recall,
                "ndcg": ndcg,
            }
        )
        del pred_matrix, metrics, recs, test_matrix, model
        gc.collect()

    return results


def mean_result(results: list[dict], param: str):
    return np.mean([r[param] for r in results])


def print_summary(results: dict):
    print("Модель", results[0]["model"])
    print("Среднее train_time", mean_result(results=results, param="train_time"))
    print("Среднее pred_time", mean_result(results=results, param="pred_time"))
    print(f"Средний прирост RSS процесса при обучении: {mean_result(results=results, param='train_mem_delta'):.3f} MB")
    print(f"Средний прирост RSS процесса при прогнозе: {mean_result(results=results, param='pred_mem_delta'):.3f} MB")
    print("Средний recall", mean_result(results=results, param="recall"))
    print("Средний ndcg", mean_result(results=results, param="ndcg"))


def params_search(
    model_factory,
    train: pd.DataFrame,
    test: pd.DataFrame,
    params: dict,
    n_runs: int = 3,
    k: int = 10,
    threshold: float = 3.5,
) -> pd.DataFrame:
    """
    Перебирает гиперпараметры и возвращает DataFrame с усредненными метриками.

    model_factory - функция, принимающая **kwargs и возвращающая новую модель.
    params - {имя_параметра: [значения]}.
    """
    keys = list(params.keys())
    combinations = list(itertools.product(*params.values()))

    results = []

    for comb in combinations:
        kwargs = dict(zip(keys, comb))

        factory = lambda kw=kwargs: model_factory(**kw)
        results_param = run_benchmark(
            model_factory=factory,
            train=train,
            test=test,
            k=k,
            threshold=threshold,
            n_runs=n_runs,
        )
        results.append(
            {
                **kwargs,
                "train_time": mean_result(results_param, "train_time"),
                "pred_time": mean_result(results_param, "pred_time"),
                "train_mem_delta": mean_result(results_param, "train_mem_delta"),
                "pred_mem_delta": mean_result(results_param, "pred_mem_delta"),
                f"recall@{k}": mean_result(results_param, "recall"),
                f"ndcg@{k}": mean_result(results_param, "ndcg"),
            }
        )

    return pd.DataFrame(results)


def show_graph(
    results_df: pd.DataFrame,
    params: list[str],
    metrics: list[str],
    save_dir: str = "graphs",
    max_lines: int | None = 10,
    group_by_params=None,
):
    for param in params:
        if group_by_params is not None and param in group_by_params:
            other_params = [p for p in group_by_params if p != param]
        else:
            other_params = [p for p in params if p != param]
        for metric in metrics:
            _, ax = plt.subplots(figsize=FIGSIZE)

            if other_params:
                groups = list(results_df.groupby(other_params))
                cmap = plt.get_cmap("turbo")
                n = len(groups)
                if max_lines is not None and n > max_lines:
                    # усредняем - одна линия
                    agg = results_df.groupby(param)[metric].agg(["mean", "std"])
                    ax.errorbar(
                        agg.index, agg["mean"], yerr=agg["std"],
                        marker="o", capsize=3,
                        label=f"mean +- std по {len(other_params)} параметрам.",
                    )
                    file_name = f"{save_dir}/{param}_{metric.replace('@', '_')}_mean_of_{n}_configs.png"
                else:
                    for i, (comb, df_sub) in enumerate(groups):
                        df_sub = df_sub.sort_values(param)
                        label = ", ".join(f"{k}={v}" for k, v in zip(other_params, comb))
                        color = cmap(i / max(n - 1, 1))
                        ax.plot(
                            df_sub[param], df_sub[metric],
                            marker="o", markersize=4,
                            color=color, label=label,
                        )
                    file_name = f"{save_dir}/{param}_{metric.replace('@', '_')}_all_{n}_lines.png"
                ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0)
            else:
                df = results_df.sort_values(param)
                ax.plot(df[param], df[metric], marker="o")
                file_name = f"{save_dir}/{param}_{metric.replace('@', '_')}_single_line.png"

            ax.set_xlabel(param)
            ax.set_ylabel(metric)
            ax.set_title(f"Зависимость {metric} от {param}")
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            if save_dir:
                plt.savefig(
                    file_name,
                    dpi=150,
                    bbox_inches="tight",
                )
            plt.show()
