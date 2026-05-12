"""Загрузка и подготовка датасетов BasicMotions и FordA."""

from __future__ import annotations

import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from pyts.datasets import load_basic_motions

PathLike = str | Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_ARCHIVE_URL = "https://timeseriesclassification.com/aeon-toolkit/"
FORD_A_RAW_URLS = {
    "train": "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/FordA_TRAIN.tsv",
    "test": "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/FordA_TEST.tsv",
}

__all__ = [
    "TSLoader",
    "TimeSeriesDatasetSplit",
    "load_dataset",
    "prepare_dataset",
    "load_basic_motions_dataset",
    "load_ford_a_dataset",
]


@dataclass(frozen=True)
class TimeSeriesDatasetSplit:
    """Разбиение train/test: ``X_*`` имеют форму ``(n_samples, n_channels, n_timesteps)``."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


def _normalize_name(dataset_name: str) -> str:
    aliases = {
        "basicmotions": "BasicMotions",
        "basic_motions": "BasicMotions",
        "forda": "FordA",
        "ford_a": "FordA",
    }
    key = dataset_name.replace("-", "_").replace(" ", "_").lower()
    if key not in aliases:
        raise ValueError(f"Неизвестный датасет {dataset_name!r}. Доступны: BasicMotions, FordA")
    return aliases[key]


def _encode_labels(y_train: np.ndarray, y_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    classes = np.unique(y_train)
    mapping = {label: idx for idx, label in enumerate(classes)}

    unknown = set(np.unique(y_test)) - set(classes)
    if unknown:
        raise ValueError(f"В test найдены классы, которых нет в train: {sorted(unknown)}")

    encode = np.vectorize(mapping.__getitem__)
    return encode(y_train).astype(np.int64), encode(y_test).astype(np.int64)


def _as_channel_first_time_series(x: np.ndarray) -> np.ndarray:
    """Приводит к ``(n_samples, n_channels, n_timesteps)``."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 2:
        return x[:, np.newaxis, :]
    if x.ndim == 3:
        return x
    raise ValueError(f"Ожидались 2D или 3D данные временных рядов, получена форма {x.shape}")


def prepare_dataset(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> TimeSeriesDatasetSplit:
    """Подготовка под PyTorch/CrossEntropyLoss: channel-first X и метки ``0..C-1``."""
    y_train_encoded, y_test_encoded = _encode_labels(y_train, y_test)
    return TimeSeriesDatasetSplit(
        X_train=_as_channel_first_time_series(X_train),
        X_test=_as_channel_first_time_series(X_test),
        y_train=y_train_encoded,
        y_test=y_test_encoded,
    )


class TSLoader:
    """Loader UCR/UEA датасетов с кэшем в ``data/<DatasetName>``."""

    archive_url = DEFAULT_ARCHIVE_URL

    def __init__(self, data_dir: PathLike = DEFAULT_DATA_DIR) -> None:
        self.data_dir = Path(data_dir)
        self.download_dir = self.data_dir / "_downloads"

    @staticmethod
    def read_tsv_or_csv(
        dataset_name: str,
        data_path: PathLike,
        mode: str = "tsv",
    ) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
        if mode not in {"tsv", "csv"}:
            raise ValueError('mode должен быть "tsv" или "csv"')

        separator = "\t" if mode == "tsv" else ","
        train_path, test_path = TSLoader._find_train_test_files(dataset_name, data_path, [mode])
        return TSLoader._read_table(train_path, separator), TSLoader._read_table(test_path, separator)

    @staticmethod
    def read_txt_files(
        dataset_name: str,
        data_path: PathLike,
    ) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
        train_path, test_path = TSLoader._find_train_test_files(dataset_name, data_path, ["txt"])
        return TSLoader._read_table(train_path, None), TSLoader._read_table(test_path, None)

    @staticmethod
    def _read_table(
        path: PathLike,
        delimiter: Optional[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        data = np.genfromtxt(path, delimiter=delimiter)
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError(f"Файл {path} должен содержать label в первой колонке и признаки далее")
        return data[:, 1:], data[:, 0]

    @staticmethod
    def _find_train_test_files(
        dataset_name: str,
        data_path: PathLike,
        extensions: Iterable[str],
    ) -> tuple[Path, Path]:
        root = Path(data_path) / dataset_name
        for ext in extensions:
            train = next(root.rglob(f"{dataset_name}_TRAIN.{ext}"), None)
            test = next(root.rglob(f"{dataset_name}_TEST.{ext}"), None)
            if train is not None and test is not None:
                return train, test
        raise FileNotFoundError(f"Не найдены TRAIN/TEST файлы для {dataset_name} в {root}")

    def read_train_test_files(
        self,
        dataset_name: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        dataset_dir = self.data_dir / dataset_name

        for mode in ("tsv", "csv"):
            try:
                (X_train, y_train), (X_test, y_test) = self.read_tsv_or_csv(
                    dataset_name, self.data_dir, mode
                )
                return X_train, y_train, X_test, y_test
            except FileNotFoundError:
                pass

        try:
            (X_train, y_train), (X_test, y_test) = self.read_txt_files(dataset_name, self.data_dir)
            return X_train, y_train, X_test, y_test
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Данные не найдены в {dataset_dir}") from exc

    def download_by_url(
        self,
        dataset_name: str,
        *,
        url: str = DEFAULT_ARCHIVE_URL,
        use_cache: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        dataset_name = _normalize_name(dataset_name)
        dataset_dir = self.data_dir / dataset_name
        if use_cache and self._has_train_test_files(dataset_name):
            return self.read_train_test_files(dataset_name)

        self.data_dir.mkdir(parents=True, exist_ok=True)
        dataset_dir.mkdir(parents=True, exist_ok=True)

        if dataset_name == "FordA":
            self._download_ford_a_files(dataset_dir, use_cache=use_cache)
        else:
            self._download_archive(dataset_name, url)

        return self.read_train_test_files(dataset_name)

    def _has_train_test_files(self, dataset_name: str) -> bool:
        for ext in ("tsv", "csv", "txt"):
            try:
                self._find_train_test_files(dataset_name, self.data_dir, [ext])
                return True
            except FileNotFoundError:
                pass
        return False

    def _download_archive(self, dataset_name: str, url: str) -> None:
        archive_url = f"{url.rstrip('/')}/{dataset_name}.zip"
        self.download_dir.mkdir(parents=True, exist_ok=True)
        archive_path = self.download_dir / f"{dataset_name}.zip"
        self._download_file(archive_url, archive_path, expect_zip=True)
        with zipfile.ZipFile(archive_path) as zip_file:
            zip_file.extractall(self.data_dir / dataset_name)

    def _download_ford_a_files(self, dataset_dir: Path, *, use_cache: bool) -> None:
        for split, url in FORD_A_RAW_URLS.items():
            target = dataset_dir / f"FordA_{split.upper()}.tsv"
            if use_cache and target.is_file():
                continue
            self._download_file(url, target, expect_zip=False)

    @staticmethod
    def _download_file(url: str, target: Path, *, expect_zip: bool) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target.with_suffix(target.suffix + ".tmp")

        with urllib.request.urlopen(url, timeout=60) as response:
            with tmp_path.open("wb") as file:
                shutil.copyfileobj(response, file)

        prefix = tmp_path.read_bytes()[:256]
        if expect_zip and not prefix.startswith(b"PK"):
            tmp_path.unlink(missing_ok=True)
            preview = prefix.decode("utf-8", errors="replace").strip()
            raise FileNotFoundError(f"По URL {url} пришёл не zip-файл: {preview[:120]}")
        if not expect_zip and prefix.lstrip().startswith((b"<!", b"<html", b"<HTML")):
            tmp_path.unlink(missing_ok=True)
            raise FileNotFoundError(f"По URL {url} пришёл HTML вместо данных")

        tmp_path.replace(target)


def load_dataset(
    dataset_name: str,
    *,
    data_dir: PathLike = DEFAULT_DATA_DIR,
    use_cache: bool = True,
) -> TimeSeriesDatasetSplit:
    """Единая точка загрузки датасетов для экспериментов."""
    dataset_name = _normalize_name(dataset_name)

    if dataset_name == "BasicMotions":
        X_train, X_test, y_train, y_test = load_basic_motions(return_X_y=True)
        return prepare_dataset(X_train, y_train, X_test, y_test)

    loader = TSLoader(data_dir)
    X_train, y_train, X_test, y_test = loader.download_by_url(dataset_name, use_cache=use_cache)
    return prepare_dataset(X_train, y_train, X_test, y_test)


def load_basic_motions_dataset() -> TimeSeriesDatasetSplit:
    """BasicMotions: 6 каналов, 100 отсчётов, 4 класса."""
    return load_dataset("BasicMotions")


def load_ford_a_dataset(
    *,
    data_home: Optional[PathLike] = None,
    use_cache: bool = True,
) -> TimeSeriesDatasetSplit:
    """FordA: одномерный ряд; сохраняется в ``data/FordA``."""
    return load_dataset("FordA", data_dir=data_home or DEFAULT_DATA_DIR, use_cache=use_cache)
