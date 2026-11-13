"""Чтение записей Bosch и признаки одного измерения."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
from numpy.typing import NDArray
from scipy.signal import welch
from sklearn.base import BaseEstimator, TransformerMixin

SAMPLE_RATE = 2000
WELCH_SIZE = 1024
EXPECTED_FILES = 1702
BANDS = ((0, 100), (100, 300), (300, 600), (600, 1000))
FEATURE_NAMES = tuple(
    f"{stat} {axis}"
    for stat in ("СКЗ", "Ст. откл.", "Размах", "Эксцесс", *(f"{lo}–{hi} Гц" for lo, hi in BANDS))
    for axis in "XYZ"
)


@dataclass(frozen=True)
class Recording:
    path: Path
    machine: str
    operation: str
    period: str
    label: int


def inventory(root: Path) -> list[Recording]:
    records = []
    for path in sorted(root.glob("M*/OP*/*/*.h5")):
        machine, operation, label, _ = path.relative_to(root).parts
        if label not in {"good", "bad"}:
            continue
        fields = path.stem.split("_")
        period = " ".join(fields[1 : fields.index(operation)])
        records.append(Recording(path, machine, operation, period, int(label == "bad")))
    if {row.machine for row in records} != {"M01", "M02", "M03"}:
        raise ValueError(f"Ожидаются данные трёх станков в {root}")
    if len(records) != EXPECTED_FILES:
        raise ValueError(f"В {root} найдено {len(records)} файлов, ожидалось {EXPECTED_FILES}")
    return records


def acceleration(path: Path) -> NDArray[np.float64]:
    with h5py.File(path, "r") as source:
        samples = np.asarray(source["vibration_data"], dtype=np.float64)
    if samples.ndim != 2 or samples.shape[1] != 3 or len(samples) < 4:
        raise ValueError(f"Ожидался массив N×3 с N ≥ 4: {path}")
    return samples


def signal_features(samples: NDArray[np.float64]) -> NDArray[np.float64]:
    if samples.ndim != 2 or samples.shape[1] != 3 or len(samples) < 4:
        raise ValueError("Ожидался массив N×3 с N ≥ 4")
    # Если канал повреждён, его признаки заполняются медианами внутри Pipeline.
    valid = np.isfinite(samples).all(axis=0)
    clean = np.where(valid, samples, 0)
    centered = clean - clean.mean(axis=0)
    std = centered.std(axis=0)
    normalized = np.divide(centered, std, out=np.zeros_like(clean), where=std > 0)
    excess = np.where(std > 0, (normalized**4).mean(axis=0) - 3, np.nan)
    freq, power = welch(
        clean,
        fs=SAMPLE_RATE,
        nperseg=min(WELCH_SIZE, len(clean)),
        axis=0,
        detrend="constant",
        scaling="density",
    )
    # Интеграл односторонней спектральной плотности по полосе.
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html
    step = freq[1] - freq[0]
    energy = [
        power[(freq >= lo) & ((freq < hi) | ((hi == SAMPLE_RATE / 2) & (freq == hi)))].sum(axis=0)
        * step
        for lo, hi in BANDS
    ]
    result = np.vstack(
        (np.sqrt((clean**2).mean(axis=0)), std, np.ptp(clean, axis=0), excess, *energy)
    )
    result[:, ~valid] = np.nan
    return result.ravel()


@lru_cache(maxsize=None)
def file_features(path: str) -> NDArray[np.float64]:
    # Кэш содержит только независимые статистики одного файла.
    return signal_features(acceleration(Path(path)))


class SignalFeatures(TransformerMixin, BaseEstimator):
    def fit(
        self, paths: NDArray[np.str_], labels: NDArray[np.int_] | None = None
    ) -> "SignalFeatures":
        return self

    def transform(self, paths: NDArray[np.str_]) -> NDArray[np.float64]:
        return np.stack([file_features(path) for path in paths])
