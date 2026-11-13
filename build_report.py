"""Воспроизводимая сборка отчёта на исходных данных Bosch."""

import argparse
import csv
import os
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from cnc_data import inventory
from download_data import download_dataset
from experiment import (
    BOOST,
    FOREST,
    TREE,
    FOLDS,
    SEED,
    boosting_error,
    candidates,
    evaluate,
    evaluate_time_transfer,
    feature_importance,
    forest_error,
    scale_check,
    select,
    split,
    time_split,
    tree_error,
)

ROOT = Path(__file__).resolve().parent
BUILD = ROOT / "build"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="Путь к уже загруженным данным Bosch")
    args = parser.parse_args()
    data = ROOT / ".bosch" / "data" if args.data is None else args.data.resolve()
    download_dataset(data, repair=args.data is None)
    records = inventory(data)
    paths = np.array([str(record.path) for record in records])
    y = np.array([record.label for record in records])
    train, test, transfer = indices = split(records)
    earlier, _, _ = time_indices = time_split(records)
    for part in (*indices, *time_indices):
        if len(np.unique(y[part])) != 2:
            raise ValueError("В каждом разбиении должны присутствовать оба класса")
    if any(min(np.bincount(y[part])) < FOLDS for part in (train, earlier)):
        raise ValueError("Слишком мало файлов редкого класса для пяти разбиений")

    BUILD.mkdir(exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(BUILD / "matplotlib"))
    from figures import Figures, diagnostics, raw_figures, selection_figures, test_figures
    from report import build_report

    figures = Figures(BUILD)
    raw_figures(figures, records, y, train)
    cv = StratifiedKFold(FOLDS, shuffle=True, random_state=SEED)
    x_train, y_train = paths[train], y[train]
    with threadpool_limits(limits=1):
        choices = select(x_train, y_train, cv, candidates())
        winner = max(choices, key=lambda name: choices[name].scores.mean())
        selection_figures(figures, choices, winner, y_train)
        print(f"Диагностика и интерпретация: {winner}", flush=True)
        scaling = scale_check(x_train, y_train, cv, choices)
        diagnostics(
            figures,
            choices,
            tree_error(x_train, y_train, cv),
            forest_error(x_train, y_train, cv, choices[FOREST], choices[TREE]),
            boosting_error(x_train, y_train, cv, choices[BOOST]),
            feature_importance(x_train, y_train, cv, choices[winner]),
        )
        print("Подбор для переноса во времени", flush=True)
        time_choices, early_test, later = evaluate_time_transfer(paths, y, time_indices, cv)
        print("Итоговая оценка на отложенных выборках", flush=True)
        ordinary = {
            name: evaluate(choice, paths[test], y[test]) for name, choice in choices.items()
        }
        moved = {
            name: evaluate(choice, paths[transfer], y[transfer])
            for name, choice in choices.items()
        }

    test_figures(figures, ordinary, moved, winner, y[test], early_test, later)
    with (BUILD / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        metrics = list(ordinary[winner].metrics)
        writer.writerow(["sample", "model", "cv_ap", "cv_ap_std", "threshold", *metrics])
        for sample, values, settings in (
            ("ordinary", ordinary, choices),
            ("transfer", moved, choices),
            ("time_early", early_test, time_choices),
            ("time_august", later, time_choices),
        ):
            for name, value in values.items():
                choice = settings[name]
                writer.writerow(
                    [
                        sample,
                        name,
                        choice.scores.mean(),
                        choice.scores.std(),
                        choice.threshold,
                        *value.metrics.values(),
                    ]
                )
    target = ROOT / "report.pdf"
    build_report(
        target,
        figures,
        records,
        indices,
        choices,
        winner,
        ordinary,
        moved,
        scaling,
        time_indices,
        time_choices,
        early_test,
        later,
    )
    print(target, flush=True)


if __name__ == "__main__":
    main()
