from __future__ import annotations

import re
from pathlib import Path

import numpy as np


def _natural_key(path: Path) -> list[object]:
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", path.name)]


def list_pcd_files(folder: str | Path) -> list[Path]:
    files = sorted(Path(folder).glob("*.pcd"), key=_natural_key)
    if not files:
        raise FileNotFoundError(f"No .pcd files found in {folder}")
    return files


def consecutive_pairs(folder: str | Path, max_pairs: int | None = None) -> list[tuple[Path, Path]]:
    files = list_pcd_files(folder)
    pairs = list(zip(files[:-1], files[1:]))
    return pairs[:max_pairs] if max_pairs else pairs


def read_pcd(path: str | Path) -> np.ndarray:
    """Read ASCII or binary PCD files containing at least x/y/z float fields."""
    path = Path(path)
    with path.open("rb") as f:
        header_lines: list[str] = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"PCD header in {path} ended before DATA line")
            text = line.decode("utf-8", errors="replace").strip()
            header_lines.append(text)
            if text.lower().startswith("data"):
                data_mode = text.split()[1].lower()
                break
        header = _parse_header(header_lines)
        fields = header["FIELDS"]
        if not {"x", "y", "z"}.issubset(fields):
            raise ValueError(f"{path} does not contain x/y/z fields")
        points = int(header.get("POINTS", header.get("WIDTH", 0)))
        if data_mode == "ascii":
            arr = np.loadtxt(f, dtype=np.float32)
            return arr[:, [fields.index("x"), fields.index("y"), fields.index("z")]]
        if data_mode != "binary":
            raise NotImplementedError(f"Unsupported PCD DATA mode: {data_mode}")
        dtype = _pcd_dtype(header)
        raw = np.frombuffer(f.read(), dtype=dtype, count=points)
        xyz = np.column_stack([raw["x"], raw["y"], raw["z"]]).astype(np.float64, copy=False)
        return xyz[np.isfinite(xyz).all(axis=1)]


def _parse_header(lines: list[str]) -> dict[str, object]:
    header: dict[str, object] = {}
    for line in lines:
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        key = parts[0].upper()
        if key in {"FIELDS", "TYPE"}:
            header[key] = parts[1:]
        elif key in {"SIZE", "COUNT"}:
            header[key] = [int(v) for v in parts[1:]]
        elif key in {"WIDTH", "HEIGHT", "POINTS"}:
            header[key] = int(parts[1])
    header.setdefault("COUNT", [1] * len(header["FIELDS"]))
    return header


def _pcd_dtype(header: dict[str, object]) -> np.dtype:
    names = header["FIELDS"]
    sizes = header["SIZE"]
    types = header["TYPE"]
    counts = header["COUNT"]
    dtype_fields = []
    for name, size, typ, count in zip(names, sizes, types, counts):
        if typ == "F" and size == 4:
            base = "<f4"
        elif typ == "F" and size == 8:
            base = "<f8"
        elif typ == "U" and size == 4:
            base = "<u4"
        elif typ == "I" and size == 4:
            base = "<i4"
        else:
            raise NotImplementedError(f"Unsupported PCD field type {typ}{size}")
        dtype_fields.append((name, base) if count == 1 else (name, base, (count,)))
    return np.dtype(dtype_fields)
