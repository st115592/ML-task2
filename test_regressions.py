"""Регрессии порогового решения, разбиений, калибровки и кэша Bosch."""

from pathlib import Path, PurePosixPath

import h5py
import numpy as np
import pytest
from matplotlib.figure import Figure
from numpy.typing import NDArray
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

import download_data
import experiment
from cnc_data import Recording
from experiment import SVM, candidates, operating_threshold, split, threshold_metrics, time_split
from figures import threshold_step


def test_equal_scores() -> None:
    assert operating_threshold(np.array([0, 1]), np.array([0.2, 0.2])) == 0.2


def test_equal_f1_chooses_larger_threshold() -> None:
    labels = np.array([1, 0, 0, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    assert f1_score(labels, scores >= 0.1) == f1_score(labels, scores >= 0.4)
    assert operating_threshold(labels, scores) == 0.4


def test_threshold_plot_between_scores() -> None:
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.02, 0.05, 0.1, 0.2])
    thresholds, _, _, f1 = threshold_metrics(labels, scores)
    line = threshold_step(Figure().subplots(), thresholds, f1, "F1")
    points = line.get_path().vertices
    for left, right in zip(points[:-1], points[1:]):
        if left[0] < 0.15 < right[0]:
            assert left[1] == right[1] == pytest.approx(f1_score(labels, scores >= 0.15))
            break
    else:
        pytest.fail("График не покрывает порог 0.15")


def test_file_and_period_splits() -> None:
    records = [
        Recording(Path(f"{machine}-{period}-{i}.h5"), machine, "OP00", period, i % 2)
        for machine in ("M01", "M02", "M03")
        for period in ("Aug 2019", "Feb 2021", "Aug 2021", "Feb 2022")
        for i in range(8)
    ]
    train, test, moved = split(records)
    paths = [{records[i].path for i in indices} for indices in (train, test, moved)]
    assert not (paths[0] & paths[1] or paths[0] & paths[2] or paths[1] & paths[2])
    assert {records[i].machine for i in [*train, *test]} == {"M01", "M02"}
    assert set(moved) == {i for i, row in enumerate(records) if row.machine == "M03"}
    earlier, control, late = time_split(records)
    train_paths, control_paths, late_paths = (
        {records[i].path for i in indices} for indices in (earlier, control, late)
    )
    assert not (
        train_paths & control_paths or train_paths & late_paths or control_paths & late_paths
    )
    assert len(control) == 8
    assert set(earlier) | set(control) == {
        i
        for i, row in enumerate(records)
        if row.machine != "M03" and row.period in {"Aug 2019", "Feb 2021"}
    }
    assert set(late) == {
        i for i, row in enumerate(records) if row.machine != "M03" and row.period == "Aug 2021"
    }


def test_time_transfer_reuses_model_and_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    paths = np.array(
        ["train-good", "train-bad", "early-good", "early-bad", "august-good", "august-bad"]
    )
    labels = np.array([0, 1, 0, 1, 0, 1])
    indices = (np.array([0, 1]), np.array([2, 3]), np.array([4, 5]))
    cv = experiment.StratifiedKFold(2)
    train_calls: list[tuple[str, ...]] = []
    predict_calls: list[tuple[str, ...]] = []

    class FittedModel(experiment.BaseEstimator):
        def predict_proba(self, x: NDArray[np.str_]) -> NDArray[np.float64]:
            predict_calls.append(tuple(x))
            return np.array([[0.4, 0.6], [0.2, 0.8]])

    selection = experiment.Selection(
        FittedModel(), {}, np.array([0.9]), {}, 0.7, np.array([0.6, 0.8])
    )

    def select(
        x: NDArray[np.str_],
        y: NDArray[np.int_],
        cv: experiment.StratifiedKFold,
        models: dict[str, experiment.Candidate],
    ) -> dict[str, experiment.Selection]:
        train_calls.append(tuple(x))
        np.testing.assert_array_equal(y, [0, 1])
        return {"model": selection}

    monkeypatch.setattr(experiment, "select", select)
    selected, early, august = experiment.evaluate_time_transfer(paths, labels, indices, cv)
    assert selected["model"] is selection
    assert train_calls == [("train-good", "train-bad")]
    assert predict_calls == [("early-good", "early-bad"), ("august-good", "august-bad")]
    for result in (early, august):
        np.testing.assert_array_equal(result["model"].prediction, [0, 1])


def test_svm_fits_preprocessing_inside_calibration(monkeypatch: pytest.MonkeyPatch) -> None:
    model = candidates()[SVM].estimator.set_params(estimator__features="passthrough")
    values = np.column_stack((np.arange(18, dtype=float), np.arange(18, dtype=float) ** 2))
    values[::4, 1] = np.nan
    labels = np.arange(len(values)) % 2
    fit_rows: dict[str, list[tuple[int, ...]]] = {"imputer": [], "scale": []}
    imputer_fit = SimpleImputer.fit
    scaler_fit = StandardScaler.fit

    def record_imputer(self: SimpleImputer, x: NDArray, y: NDArray | None = None) -> SimpleImputer:
        fit_rows["imputer"].append(tuple(x[:, 0].astype(int)))
        return imputer_fit(self, x, y)

    def record_scale(
        self: StandardScaler,
        x: NDArray,
        y: NDArray | None = None,
        sample_weight: NDArray | None = None,
    ) -> StandardScaler:
        fit_rows["scale"].append(tuple(x[:, 0].astype(int)))
        return scaler_fit(self, x, y, sample_weight=sample_weight)

    monkeypatch.setattr(SimpleImputer, "fit", record_imputer)
    monkeypatch.setattr(StandardScaler, "fit", record_scale)
    model.fit(values, labels)
    expected = [tuple(train) for train, _ in model.cv.split(values, labels)]
    expected.append(tuple(range(len(values))))
    assert sorted(fit_rows["imputer"]) == sorted(expected)
    assert sorted(fit_rows["scale"]) == sorted(expected)


def test_dataset_resumes_and_repairs_same_size_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    names = ["data/M01/OP00/good/a.h5", "data/M01/OP00/good/b.h5"]
    originals = []
    for i in range(3):
        source = tmp_path / f"original-{i}.h5"
        with h5py.File(source, "w") as handle:
            handle["vibration_data"] = np.full((8, 3), i, dtype=np.float32)
        originals.append(source.read_bytes())
    assert len({len(data) for data in originals}) == 1
    payloads = dict(zip(names, originals[:2]))
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        "".join(
            f"{name}\t{len(data)}\t{download_data.blob_sha(data)}\n"
            for name, data in payloads.items()
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(download_data, "MANIFEST", manifest)
    monkeypatch.setattr(download_data, "EXPECTED_FILES", 2)
    requests = []
    fail = {names[1]}

    def fetch(url: str) -> bytes:
        name = url.split(f"/{download_data.REVISION}/")[1]
        requests.append(name)
        return b"x" if name in fail else payloads[name]

    monkeypatch.setattr(download_data, "fetch", fetch)
    paths = [tmp_path.joinpath(*PurePosixPath(name).parts[1:]) for name in names]
    with pytest.raises(ValueError, match="Контрольная сумма"):
        download_data.download_dataset(tmp_path)
    assert paths[0].read_bytes() == payloads[names[0]]
    assert not paths[1].exists()
    fail.clear()
    requests.clear()
    download_data.download_dataset(tmp_path)
    assert requests == [names[1]]
    requests.clear()
    paths[0].write_bytes(originals[2])
    with h5py.File(paths[0], "r") as handle:
        assert np.all(handle["vibration_data"][:] == 2)
    with pytest.raises(ValueError, match="не прошёл проверку"):
        download_data.download_dataset(tmp_path, repair=False)
    assert not requests
    download_data.download_dataset(tmp_path)
    assert requests == [names[0]]
    assert paths[0].read_bytes() == payloads[names[0]]
