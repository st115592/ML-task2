"""Графики и краткие выводы, полученные из результатов расчёта."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from numpy.typing import NDArray
from sklearn.metrics import precision_recall_curve, roc_curve

from cnc_data import FEATURE_NAMES, SAMPLE_RATE, Recording, acceleration
from experiment import (
    BOOST,
    BOOST_RATES,
    BOOST_STEPS,
    DUMMY,
    FOREST,
    RF_COUNTS,
    SVM,
    SVM_PREFIX,
    threshold_metrics,
    Evaluation,
    Selection,
    candidates,
)

BLUE = "#255c85"
RED = "#ac4937"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


class Figures:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.items: dict[str, tuple[Path, str, str]] = {}

    def add(self, key: str, fig: Figure, title: str, finding: str) -> None:
        path = self.root / f"{key}.png"
        fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        self.items[key] = path, title, finding


def raw_figures(
    out: Figures, records: list[Recording], y: NDArray[np.int_], train: NDArray[np.int_]
) -> None:
    fig, ax = plt.subplots(figsize=(6.7, 2.8), layout="constrained")
    counts = np.bincount(y)
    bars = ax.barh(["good", "bad"], counts, color=[BLUE, RED])
    ax.bar_label(bars, labels=[f"{n} ({n / len(y):.1%})" for n in counts], padding=5)
    ax.set(xlabel="Число файлов", xlim=(0, counts.max() * 1.25))
    out.add(
        "classes",
        fig,
        "Распределение классов в наборе",
        f"Аномальные записи составляют {counts[1] / len(y):.1%} всех файлов.",
    )

    groups: dict[tuple[str, str, str], list[Recording]] = {}
    for i in sorted(train):
        record = records[i]
        groups.setdefault((record.machine, record.operation, record.period), []).append(record)
    key, group = max(
        ((key, rows) for key, rows in groups.items() if {row.label for row in rows} == {0, 1}),
        key=lambda pair: len(pair[1]),
    )
    fig, axes = plt.subplots(3, 1, figsize=(6.7, 5.4), sharex=True, layout="constrained")
    rms = []
    for label, color in [(0, BLUE), (1, RED)]:
        rows = [row for row in group if row.label == label]
        fragments = []
        for row in rows:
            samples = acceleration(row.path)
            size = min(len(samples), SAMPLE_RATE // 2)
            start = (len(samples) - size) // 2
            segment = samples[start : start + size]
            fragments.append((float(np.sqrt(np.mean(segment**2))), segment, row.path.name))
        score, segment, filename = sorted(fragments, key=lambda item: item[0])[len(rows) // 2]
        rms.append((score, filename))
        t = np.arange(len(segment)) / SAMPLE_RATE
        for j, ax in enumerate(axes):
            ax.plot(
                t, segment[:, j], lw=0.65, alpha=0.8, color=color, label=["good", "bad"][label]
            )
            ax.set_ylabel(f"a{'XYZ'[j]}, ед. данных")
    axes[0].legend(ncols=2)
    axes[-1].set_xlabel("Время от начала фрагмента, с")
    out.add(
        "signal",
        fig,
        f"Фрагменты ускорения {key[0]}, {key[1]}, {key[2]}",
        f"СКЗ трёх компонент в выбранных фрагментах равно {rms[0][0]:.3g} для good "
        f"({rms[0][1]}) и {rms[1][0]:.3g} для bad ({rms[1][1]}).",
    )


def threshold_step(
    ax: Axes,
    thresholds: NDArray[np.float64],
    values: NDArray[np.float64],
    label: str,
) -> Line2D:
    # Для >= каждая ступень включает свою правую границу.
    return ax.step(
        np.r_[0.0, thresholds, 1.0], np.r_[values[0], values], where="pre", label=label
    )[0]


def selection_figures(
    out: Figures, choices: dict[str, Selection], winner: str, y: NDArray[np.int_]
) -> None:
    names = list(choices)
    means = [s.scores.mean() for s in choices.values()]
    stds = [s.scores.std() for s in choices.values()]
    fig, ax = plt.subplots(figsize=(6.7, 3.3), layout="constrained")
    ax.bar(names, means, yerr=stds, capsize=3, color=BLUE)
    ax.set(ylabel="Средняя точность (AP)", ylim=(0, min(1.07, max(np.add(means, stds)) + 0.1)))
    out.add(
        "cv",
        fig,
        "Сравнение моделей по пяти разбиениям",
        f"Наибольшая средняя AP у модели «{winner}» ({max(means):.3f}). "
        "Отрезки показывают стандартное отклонение между разбиениями.",
    )

    choice = choices[SVM]
    grid = candidates()[SVM].grid
    matrix = np.empty((len(grid["C"]), len(grid["gamma"])))
    for params, mean in zip(choice.grid_results["params"], choice.grid_results["mean_test_score"]):
        matrix[
            grid["C"].index(params[f"{SVM_PREFIX}C"]),
            grid["gamma"].index(params[f"{SVM_PREFIX}gamma"]),
        ] = mean
    fig, ax = plt.subplots(figsize=(5.5, 3.4), layout="constrained")
    image = ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    ax.set(
        xticks=range(3),
        xticklabels=grid["gamma"],
        yticks=range(3),
        yticklabels=grid["C"],
        xlabel="gamma",
        ylabel="C",
    )
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(
            j,
            i,
            f"{value:.3f}",
            ha="center",
            va="center",
            color="white" if value > 0.57 else "black",
        )
    fig.colorbar(image, ax=ax, label="AP")
    out.add(
        "grid",
        fig,
        "Сеточный поиск SVM",
        f"Лучший вариант дал AP {matrix.max():.3f} при C={choice.params['C']} "
        f"и gamma={choice.params['gamma']}.",
    )

    choice = choices[winner]
    thresholds, precision, recall, f1 = threshold_metrics(y, choice.oof)
    fig, ax = plt.subplots(figsize=(6.7, 3.2), layout="constrained")
    for values, label in [(precision, "Точность"), (recall, "Полнота"), (f1, "F1")]:
        threshold_step(ax, thresholds, values, label)
    ax.axvline(choice.threshold, ls="--", color=".3", label=f"Порог {choice.threshold:.4g}")
    ax.set(xlabel="Порог оценки класса bad", ylabel="Значение метрики", ylim=(0, 1.06))
    ax.legend(ncols=2, fontsize=9)
    out.add(
        "threshold",
        fig,
        f"Рабочий порог модели «{winner}»",
        f"Порог {choice.threshold:.4g} даёт F1={f1[:-1].max():.3f} на вневыборочных "
        "оценках обучающей части. При равенстве F1 выбран больший порог.",
    )


def diagnostics(
    out: Figures,
    choices: dict[str, Selection],
    tree: dict[str, list[tuple[str, float, float]]],
    forest: NDArray[np.float64],
    boost: NDArray[np.float64],
    importance: NDArray[np.float64],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 3), sharey=True, layout="constrained")
    gaps = []
    for ax, (param, points) in zip(axes, tree.items()):
        labels, fit, val = zip(*points)
        ax.plot(labels, fit, "o-", label="Обучение")
        ax.plot(labels, val, "s-", label="Проверка")
        ax.set_xlabel(param)
        gap = np.array(val) - fit
        index = int(np.argmax(gap))
        gaps.append(f"{param}={labels[index]} ({gap[index]:.3f})")
    axes[0].set_ylabel("1 − сбалансированная доля")
    axes[1].legend(fontsize=9)
    out.add(
        "tree",
        fig,
        "Переобучение решающего дерева",
        f"Наибольшие разрывы между ошибками проверки и обучения — {' и '.join(gaps)}.",
    )

    mean, std = forest.mean(axis=1), forest.std(axis=1)
    fig, ax = plt.subplots(figsize=(6.7, 3.1), layout="constrained")
    ticks = ["Дерево", *map(str, RF_COUNTS)]
    ax.errorbar(ticks, mean, yerr=std, fmt="o-", capsize=4)
    ax.set(xlabel="Число деревьев в лесу", ylabel="Ошибка 1 − AP")
    out.add(
        "forest_count",
        fig,
        "Число деревьев и разброс по пяти случайным состояниям",
        f"У одиночного дерева ошибка {mean[0]:.3f} ± {std[0]:.3f}, у леса из одного "
        f"дерева {mean[1]:.3f} ± {std[1]:.3f}, из {RF_COUNTS[-1]} — "
        f"{mean[-1]:.3f} ± {std[-1]:.3f}.",
    )

    fig, ax = plt.subplots(figsize=(6.7, 3.1), layout="constrained")
    findings = []
    for rate, values in zip(BOOST_RATES, boost):
        ax.plot(BOOST_STEPS, values, "o-", label=f"learning_rate={rate}")
        i = int(values.argmin())
        findings.append(f"{rate:g} — {values[i]:.3f} на {BOOST_STEPS[i]} итерациях")
    ax.set(xlabel="Число итераций", ylabel="Ошибка 1 − AP")
    ax.legend()
    out.add(
        "boost_iterations",
        fig,
        "Проверочная ошибка бустинга по итерациям",
        "Минимальная ошибка среди показанных точек при скорости " + ", при ".join(findings) + ".",
    )

    for key, name in [("forest_parameters", FOREST), ("boost_parameters", BOOST)]:
        grid = candidates()[name].grid
        result = choices[name].grid_results
        fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.75), sharey=True, layout="constrained")
        differences = {}
        for ax, (param, values) in zip(axes, grid.items()):
            means = []
            for value in values:
                mask = [row[f"model__{param}"] == value for row in result["params"]]
                means.append(result["mean_test_score"][mask].mean())
            ax.plot([str(val) for val in values], means, "o-")
            ax.set_xlabel(param, fontsize=9)
            differences[param] = max(means) - min(means)
        axes[0].set_ylabel("Средняя AP")
        largest = max(differences, key=differences.__getitem__)
        out.add(
            key,
            fig,
            f"Влияние параметров модели «{name}»",
            f"Наибольший сдвиг средней AP по сетке связан с {largest} "
            f"({differences[largest]:.3f}). Остальные параметры усреднены.",
        )

    mean = importance.mean(axis=1)
    top = np.argsort(mean)[-8:]
    fig, ax = plt.subplots(figsize=(6.7, 3.5), layout="constrained")
    ax.barh(
        [FEATURE_NAMES[i] for i in top],
        mean[top],
        xerr=importance.std(axis=1)[top],
        color=BLUE,
        capsize=3,
    )
    ax.axvline(0, color=".3", lw=1)
    ax.set_xlabel("Снижение AP после перестановки")
    out.add(
        "importance",
        fig,
        "Перестановочная важность на проверочных частях",
        f"Наибольшее среднее снижение AP ({mean[top[-1]]:.3f}) вызывает перестановка "
        f"признака «{FEATURE_NAMES[top[-1]]}».",
    )


def test_figures(
    out: Figures,
    standard: dict[str, Evaluation],
    transfer: dict[str, Evaluation],
    winner: str,
    y: NDArray[np.int_],
    later: dict[str, Evaluation],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.1), layout="constrained")
    for name, result in standard.items():
        fpr, tpr, _ = roc_curve(y, result.scores)
        precision, recall, _ = precision_recall_curve(y, result.scores)
        width = 2.4 if name == winner else 1.1
        axes[0].plot(fpr, tpr, lw=width, label=f"{name} {result.metrics['ROC-AUC']:.3f}")
        axes[1].plot(recall, precision, lw=width, label=f"{name} {result.metrics['PR-AUC']:.3f}")
    axes[1].axhline(y.mean(), color=".4", ls=":", label="Доля bad")
    axes[0].set(xlabel="Доля ложных тревог", ylabel="Полнота", title="ROC")
    axes[1].set(xlabel="Полнота", ylabel="Точность", title="Точность–полнота")
    for ax in axes:
        ax.legend(fontsize=7, loc="lower right" if ax is axes[0] else "upper right")
    metrics = standard[winner].metrics
    out.add(
        "roc_pr",
        fig,
        "ROC и кривые точности–полноты на обычном тесте",
        f"Для модели «{winner}» ROC-AUC={metrics['ROC-AUC']:.3f}, "
        f"PR-AUC={metrics['PR-AUC']:.3f}. Доля bad в тесте {y.mean():.1%}.",
    )

    fig, axes = plt.subplots(1, 2, figsize=(6.7, 3.15), layout="constrained")
    for ax, results, title in zip(axes, (standard, transfer), ("Обычный тест", "Перенос на M03")):
        matrix = results[winner].matrix
        fraction = matrix / matrix.sum(axis=1, keepdims=True)
        ax.imshow(fraction, vmin=0, vmax=1, cmap="Blues")
        for (i, j), value in np.ndenumerate(fraction):
            ax.text(
                j,
                i,
                f"{value:.1%}",
                ha="center",
                va="center",
                color="white" if value > 0.6 else "black",
            )
        ax.set(
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["good", "bad"],
            yticklabels=["good", "bad"],
            xlabel="Решение",
            ylabel="Истинный класс",
            title=f"{title}\nF1={results[winner].metrics['F1']:.3f}",
        )
    out.add(
        "matrix",
        fig,
        f"Нормированные матрицы модели «{winner}»",
        f"Полнота класса bad составила {metrics['Recall']:.1%} в обычном тесте "
        f"и {transfer[winner].metrics['Recall']:.1%} на M03.",
    )

    fig, ax = plt.subplots(figsize=(6.7, 3.2), layout="constrained")
    names = list(standard)
    positions = np.arange(len(names))
    normal = [standard[name].metrics["AP"] for name in names]
    moved = [transfer[name].metrics["AP"] for name in names]
    ax.bar(positions - 0.18, normal, 0.36, color=BLUE, label="Обычный тест")
    ax.bar(positions + 0.18, moved, 0.36, color=RED, label="Перенос на M03")
    ax.set(xticks=positions, xticklabels=names, ylabel="AP", ylim=(0, 1.09))
    ax.legend(fontsize=9)
    best = max((n for n in names if n != DUMMY), key=lambda name: transfer[name].metrics["AP"])
    delta = transfer[winner].metrics["AP"] - metrics["AP"]
    out.add(
        "transfer",
        fig,
        "Обычная проверка и перенос на другой станок",
        f"На M03 наибольшая AP у модели «{best}» ({transfer[best].metrics['AP']:.3f}). "
        f"Для выбранной модели AP изменилась на {delta:+.3f}.",
    )

    fig, ax = plt.subplots(figsize=(6.7, 3.2), layout="constrained")
    late = [later[name].metrics["AP"] for name in names]
    ax.bar(positions - 0.18, normal, 0.36, color=BLUE, label="Обычный тест")
    ax.bar(positions + 0.18, late, 0.36, color=RED, label="Август 2021")
    ax.set(xticks=positions, xticklabels=names, ylabel="AP", ylim=(0, 1.09))
    ax.legend(fontsize=9)
    best_late = max(names, key=lambda name: later[name].metrics["AP"])
    out.add(
        "time_transfer",
        fig,
        "Обычная проверка и перенос на более поздний период",
        f"В августе 2021 наибольшая AP у модели «{best_late}» "
        f"({later[best_late].metrics['AP']:.3f}). Для каждой модели настройки "
        "временной проверки выбраны заново на более ранних файлах.",
    )
