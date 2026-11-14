"""Разбиение, выбор моделей и окончательная проверка."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    LeaveOneGroupOut,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from cnc_data import Recording, SignalFeatures

SEED = 42
FOLDS = 5
MAIN = "average_precision"
TRANSFER_REVIEW_FRACTION = 0.05
FOREST = "Лес"
BOOST = "Бустинг"
TREE = "Дерево"
DUMMY = "Dummy"
SVM = "SVM"
SVM_PREFIX = "estimator__model__"
LATE_PERIOD = "Aug 2021"
MONTHS = {"Feb": 2, "Aug": 8}
RF_SEEDS = (11, 23, 42, 67, 91)
RF_COUNTS = (1, 5, 15, 40, 100)
BOOST_STEPS = (1, 5, 10, 20, 40, 80, 120)
BOOST_RATES = (0.04, 0.16)
FloatArray = NDArray[np.float64]
Labels = NDArray[np.int_]
Paths = NDArray[np.str_]


@dataclass
class Candidate:
    estimator: BaseEstimator
    grid: dict[str, list[Any]]


@dataclass
class Selection:
    model: BaseEstimator
    params: dict[str, Any]
    scores: FloatArray
    grid_results: dict[str, Any]
    threshold: float
    oof: FloatArray


@dataclass
class Evaluation:
    scores: FloatArray
    prediction: Labels
    metrics: dict[str, float]
    matrix: Labels


@dataclass
class MachineTransfer:
    winner: str
    choices: dict[str, Selection]
    validation: dict[str, dict[str, Evaluation]]


def pipeline(model: BaseEstimator, scale: bool = False) -> Pipeline:
    return Pipeline(
        [
            ("features", SignalFeatures()),
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler() if scale else "passthrough"),
            ("model", model),
        ]
    )


def candidates() -> dict[str, Candidate]:
    return {
        DUMMY: Candidate(pipeline(DummyClassifier(strategy="most_frequent")), {}),
        "Лог. регр.": Candidate(
            pipeline(LogisticRegression(max_iter=2000), True), {"C": [0.1, 1, 10]}
        ),
        "kNN": Candidate(
            pipeline(KNeighborsClassifier(weights="distance"), True), {"n_neighbors": [3, 7, 15]}
        ),
        SVM: Candidate(
            CalibratedClassifierCV(
                pipeline(SVC(), scale=True),
                method="sigmoid",
                cv=StratifiedKFold(3, shuffle=True, random_state=SEED),
                ensemble=False,
            ),
            {"C": [0.3, 3, 30], "gamma": [0.01, 0.05, 0.2]},
        ),
        TREE: Candidate(
            pipeline(DecisionTreeClassifier(random_state=SEED)),
            {"max_depth": [2, 4, 8, None], "min_samples_leaf": [1, 5, 15]},
        ),
        FOREST: Candidate(
            pipeline(RandomForestClassifier(random_state=SEED)),
            {"n_estimators": [40, 100], "max_features": ["sqrt", 1.0], "min_samples_leaf": [1, 5]},
        ),
        BOOST: Candidate(
            pipeline(
                HistGradientBoostingClassifier(
                    random_state=SEED,
                    early_stopping=False,
                    max_iter=120,
                )
            ),
            {
                "learning_rate": list(BOOST_RATES),
                "max_leaf_nodes": [7, 15],
                "min_samples_leaf": [8, 20],
            },
        ),
    }


def split(records: list[Recording]) -> tuple[Labels, Labels, Labels]:
    source = np.array([i for i, row in enumerate(records) if row.machine != "M03"])
    transfer = np.array([i for i, row in enumerate(records) if row.machine == "M03"])
    train, test = train_test_split(
        source,
        test_size=0.25,
        random_state=SEED,
        stratify=[records[i].label for i in source],
    )
    return train, test, transfer


def period_order(period: str) -> tuple[int, int]:
    month, year = period.split()
    return int(year), MONTHS[month]


def time_split(records: list[Recording]) -> tuple[Labels, Labels, Labels]:
    source = [i for i, row in enumerate(records) if row.machine in {"M01", "M02"}]
    cutoff = period_order(LATE_PERIOD)
    earlier = [i for i in source if period_order(records[i].period) < cutoff]
    late = [i for i in source if records[i].period == LATE_PERIOD]
    train, control = train_test_split(
        np.array(earlier, dtype=int),
        test_size=0.25,
        random_state=SEED,
        stratify=[records[i].label for i in earlier],
    )
    return train, control, np.array(late, dtype=int)


def threshold_metrics(
    y: Labels,
    scores: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    precision, recall, thresholds = precision_recall_curve(y, scores)
    # При пороге выше максимальной оценки модель не подаёт тревог.
    precision[-1] = 0.0
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    return thresholds, precision, recall, f1


def operating_threshold(y: Labels, scores: FloatArray) -> float:
    thresholds, _, _, f1 = threshold_metrics(y, scores)
    best = np.flatnonzero(np.isclose(f1[:-1], f1[:-1].max(), rtol=0, atol=1e-12))
    return float(thresholds[best[-1]])


def select(
    x: Paths,
    y: Labels,
    cv: StratifiedKFold | list[tuple[Labels, Labels]],
    models: dict[str, Candidate],
) -> dict[str, Selection]:
    selected = {}
    n_splits = len(cv) if isinstance(cv, list) else cv.get_n_splits()
    for name, candidate in models.items():
        print(f"Подбор: {name}", flush=True)
        prefix = SVM_PREFIX if name == SVM else "model__"
        search = GridSearchCV(
            candidate.estimator,
            {f"{prefix}{key}": values for key, values in candidate.grid.items()},
            scoring=MAIN,
            cv=cv,
            error_score="raise",
        )
        search.fit(x, y)
        index = search.best_index_
        fold_scores = np.array(
            [search.cv_results_[f"split{i}_test_score"][index] for i in range(n_splits)]
        )
        oof = cross_val_predict(search.best_estimator_, x, y, cv=cv, method="predict_proba")[:, 1]
        params = {key.removeprefix(prefix): val for key, val in search.best_params_.items()}
        selected[name] = Selection(
            search.best_estimator_,
            params,
            fold_scores,
            search.cv_results_,
            0.5 if name == DUMMY else operating_threshold(y, oof),
            oof,
        )
    return selected


def evaluate(selection: Selection, x: Paths, y: Labels) -> Evaluation:
    score = selection.model.predict_proba(x)[:, 1]
    return evaluate_scores(y, score, selection.threshold)


def evaluate_scores(y: Labels, score: FloatArray, threshold: float) -> Evaluation:
    pred = (score >= threshold).astype(int)
    precision, recall, _ = precision_recall_curve(y, score)
    metrics = {
        "Accuracy": accuracy_score(y, pred),
        "Balanced acc.": balanced_accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y, score),
        "PR-AUC": auc(recall, precision),
        "AP": average_precision_score(y, score),
    }
    return Evaluation(score, pred, metrics, confusion_matrix(y, pred, labels=[0, 1]))


def review_threshold(scores: FloatArray) -> float:
    """Порог для ручного просмотра не более 5 % записей одной партии."""
    budget = int(len(scores) * TRANSFER_REVIEW_FRACTION)
    boundary = len(scores) - budget - 1
    # Если несколько записей совпали на границе, вся группа остаётся ниже порога.
    return float(np.nextafter(np.partition(scores, boundary)[boundary], np.inf))


def evaluate_review(
    selection: Selection,
    x: Paths,
    y: Labels,
) -> tuple[Evaluation, float]:
    scores = selection.model.predict_proba(x)[:, 1]
    threshold = review_threshold(scores)
    return evaluate_scores(y, scores, threshold), threshold


def select_machine_transfer(x: Paths, y: Labels, machines: Paths) -> MachineTransfer:
    if set(machines) != {"M01", "M02"}:
        raise ValueError("Для подбора переноса нужны только обучающие файлы M01 и M02")
    folds = list(LeaveOneGroupOut().split(x, y, machines))
    models = candidates()
    del models[DUMMY]
    choices = select(x, y, folds, models)
    validation = {}
    for name, choice in choices.items():
        validation[name] = {
            str(machines[test[0]]): evaluate_scores(
                y[test], choice.oof[test], review_threshold(choice.oof[test])
            )
            for _, test in folds
        }
    winner = max(
        choices,
        key=lambda name: (
            np.mean([result.metrics["Recall"] for result in validation[name].values()]),
            choices[name].scores.mean(),
        ),
    )
    return MachineTransfer(winner, choices, validation)


def evaluate_time_transfer(
    x: Paths,
    y: Labels,
    indices: tuple[Labels, Labels, Labels],
    cv: StratifiedKFold,
) -> tuple[dict[str, Selection], dict[str, Evaluation], dict[str, Evaluation]]:
    train, control, late = indices
    choices = select(x[train], y[train], cv, candidates())
    baseline = {name: evaluate(choice, x[control], y[control]) for name, choice in choices.items()}
    august = {name: evaluate(choice, x[late], y[late]) for name, choice in choices.items()}
    return choices, baseline, august


def scale_check(
    x: Paths, y: Labels, cv: StratifiedKFold, selected: dict[str, Selection]
) -> dict[str, tuple[FloatArray, FloatArray]]:
    result = {}
    for name in (TREE, FOREST):
        choice = selected[name]
        scaled = pipeline(choice.model.named_steps["model"], True)
        result[name] = (choice.scores, cross_val_score(scaled, x, y, cv=cv, scoring=MAIN))
    return result


def tree_error(
    x: Paths, y: Labels, cv: StratifiedKFold
) -> dict[str, list[tuple[str, float, float]]]:
    curves = {}
    for key, values in candidates()[TREE].grid.items():
        points = []
        for value in values:
            # Обе кривые включают дерево без ограничений роста.
            params = {"max_depth": None, "min_samples_leaf": 1, key: value}
            train_error, val_error = [], []
            for train, val in cv.split(x, y):
                model = pipeline(DecisionTreeClassifier(random_state=SEED, **params))
                model.fit(x[train], y[train])
                train_error.append(1 - balanced_accuracy_score(y[train], model.predict(x[train])))
                val_error.append(1 - balanced_accuracy_score(y[val], model.predict(x[val])))
            points.append(
                (
                    str(value) if value is not None else "Без огр.",
                    float(np.mean(train_error)),
                    float(np.mean(val_error)),
                )
            )
        curves[key] = points
    return curves


def forest_error(
    x: Paths, y: Labels, cv: StratifiedKFold, chosen: Selection, tree: Selection
) -> FloatArray:
    errors = []
    for count in (0, *RF_COUNTS):
        by_seed = []
        for seed in RF_SEEDS:
            if count == 0:
                estimator = DecisionTreeClassifier(random_state=seed, **tree.params)
            else:
                params = {**chosen.params, "n_estimators": count}
                estimator = RandomForestClassifier(random_state=seed, **params)
            scores = cross_val_score(pipeline(estimator), x, y, cv=cv, scoring=MAIN)
            by_seed.append(1 - scores.mean())
        errors.append(by_seed)
    return np.array(errors)


def boosting_error(x: Paths, y: Labels, cv: StratifiedKFold, chosen: Selection) -> FloatArray:
    result = []
    for rate in BOOST_RATES:
        by_fold = []
        for train, val in cv.split(x, y):
            params = {**chosen.params, "learning_rate": rate}
            model = pipeline(
                HistGradientBoostingClassifier(
                    random_state=SEED,
                    early_stopping=False,
                    max_iter=max(BOOST_STEPS),
                    **params,
                )
            )
            model.fit(x[train], y[train])
            transformed = model[:-1].transform(x[val])
            iterations = model[-1].staged_predict_proba(transformed)
            by_fold.append(
                [
                    1 - average_precision_score(y[val], prob[:, 1])
                    for step, prob in enumerate(iterations, 1)
                    if step in BOOST_STEPS
                ]
            )
        result.append(np.mean(by_fold, axis=0))
    return np.array(result)


def feature_importance(x: Paths, y: Labels, cv: StratifiedKFold, chosen: Selection) -> FloatArray:
    # Извлечение признаков не обучается. Остальная обработка остаётся в модели.
    model = clone(chosen.model)
    prefix = "estimator__" if isinstance(model, CalibratedClassifierCV) else ""
    model.set_params(**{f"{prefix}features": "passthrough"})
    features = SignalFeatures().transform(x)
    by_fold = []
    for train, val in cv.split(features, y):
        model.fit(features[train], y[train])
        result = permutation_importance(
            model, features[val], y[val], scoring=MAIN, n_repeats=12, random_state=SEED
        )
        by_fold.append(result.importances)
    return np.concatenate(by_fold, axis=1)
