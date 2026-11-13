"""Загрузка зафиксированной версии Bosch CNC Machining."""

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from urllib.request import Request, urlopen

from cnc_data import EXPECTED_FILES

REVISION = "d60581d6a3ab6015dcc5488c3d76112bb8e1bcb1"
REPO = "boschresearch/CNC_Machining"


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


def fetch_record(root: Path, relative: PurePosixPath, size: int, sha: str) -> None:
    path = root.joinpath(*relative.parts[1:])
    if path.is_file() and path.stat().st_size == size and blob_sha(path.read_bytes()) == sha:
        return
    url = f"https://raw.githubusercontent.com/{REPO}/{REVISION}/{relative}"
    contents = fetch(url)
    if len(contents) != size or blob_sha(contents) != sha:
        raise ValueError(f"Контрольная сумма не совпала для {relative}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".part")
    partial.write_bytes(contents)
    partial.replace(path)


def download_dataset(root: Path) -> None:
    url = f"https://api.github.com/repos/{REPO}/git/trees/{REVISION}?recursive=1"
    tree = json.loads(fetch(url))
    files = []
    for row in tree["tree"]:
        relative = PurePosixPath(row["path"])
        if relative.parts[0] == "data" and relative.suffix == ".h5":
            if len(relative.parts) != 5 or ".." in relative.parts or row["type"] != "blob":
                raise ValueError(f"Неожиданный путь в источнике Bosch: {relative}")
            files.append((relative, row["size"], row["sha"]))
    if tree["truncated"] or len(files) != EXPECTED_FILES:
        raise ValueError("GitHub вернул неполный список файлов Bosch")
    print(f"Загрузка набора Bosch ({EXPECTED_FILES} файлов)", flush=True)

    def fetch_one(row: tuple[PurePosixPath, int, str]) -> None:
        fetch_record(root, *row)

    with ThreadPoolExecutor(max_workers=16) as workers:
        for index, _ in enumerate(workers.map(fetch_one, files), 1):
            if index % 100 == 0 or index == len(files):
                print(f"Проверено {index} из {len(files)} файлов", flush=True)
