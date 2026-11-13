"""Загрузка зафиксированной версии Bosch CNC Machining."""

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile
from urllib.request import Request, urlopen

from cnc_data import EXPECTED_FILES

REVISION = "d60581d6a3ab6015dcc5488c3d76112bb8e1bcb1"
REPO = "boschresearch/CNC_Machining"
MANIFEST = Path(__file__).with_name("bosch_manifest.tsv")


def fetch(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "ML-task2"})
    for attempt in range(4):
        try:
            with urlopen(request, timeout=60) as response:
                return response.read()
        except OSError:
            if attempt == 3:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("Загрузка не завершена")


def blob_sha(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("ascii")
    digest = hashlib.sha1(header, usedforsecurity=False)
    digest.update(data)
    return digest.hexdigest()


def fetch_record(
    root: Path,
    relative: PurePosixPath,
    size: int,
    sha: str,
    *,
    repair: bool = True,
) -> None:
    path = root.joinpath(*relative.parts[1:])
    if path.is_file() and path.stat().st_size == size and blob_sha(path.read_bytes()) == sha:
        return
    if not repair:
        raise ValueError(f"Файл отсутствует или не прошёл проверку Bosch: {path}")
    url = f"https://raw.githubusercontent.com/{REPO}/{REVISION}/{relative}"
    contents = fetch(url)
    if len(contents) != size or blob_sha(contents) != sha:
        raise ValueError(f"Контрольная сумма не совпала для {relative}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".part", delete=False) as stream:
        partial = Path(stream.name)
        try:
            stream.write(contents)
            stream.close()
            partial.replace(path)
        finally:
            stream.close()
            partial.unlink(missing_ok=True)


def download_dataset(root: Path, *, repair: bool = True) -> None:
    files = []
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        name, size, sha = line.split("\t")
        relative = PurePosixPath(name)
        if (
            len(relative.parts) != 5
            or relative.parts[0] != "data"
            or ".." in relative.parts
            or relative.suffix != ".h5"
        ):
            raise ValueError(f"Неожиданный путь в манифесте Bosch: {relative}")
        files.append((relative, int(size), sha))
    if len(files) != EXPECTED_FILES or len({row[0] for row in files}) != EXPECTED_FILES:
        raise ValueError("Неполный манифест файлов Bosch")
    print(f"Проверка набора Bosch ({EXPECTED_FILES} файлов)", flush=True)

    def fetch_one(row: tuple[PurePosixPath, int, str]) -> None:
        fetch_record(root, *row, repair=repair)

    with ThreadPoolExecutor(max_workers=16) as workers:
        for index, _ in enumerate(workers.map(fetch_one, files), 1):
            if index % 100 == 0 or index == len(files):
                print(f"Проверено {index} из {len(files)} файлов", flush=True)
