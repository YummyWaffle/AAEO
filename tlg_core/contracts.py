"""Serializable records. No network, GIS or ML imports occur in this module."""
from dataclasses import dataclass, field, asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any


def stable_key(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class GridSpec:
    crs: str
    transform: tuple  # GDAL order: origin x, pixel x, rotation x, origin y, rotation y, pixel y
    width: int
    height: int

    def __post_init__(self):
        if self.width < 1 or self.height < 1 or len(self.transform) != 6:
            raise ValueError("Grid needs positive dimensions and a six-value GDAL transform")

    def same_grid(self, other, atol=1e-7):
        return (self.width == other.width and self.height == other.height
                and self.crs == other.crs
                and all(abs(a-b) <= atol for a, b in zip(self.transform, other.transform)))


@dataclass(frozen=True)
class BandSpec:
    name: str
    asset_key: str
    index: int = 1
    unit: str = "native"
    scale: float = 1.0
    offset: float = 0.0
    kind: str = "continuous"

    def __post_init__(self):
        if not self.name or self.index < 1 or self.kind not in ("continuous", "categorical", "direction"):
            raise ValueError("Invalid band specification")


@dataclass(frozen=True)
class FeatureSchema:
    bands: tuple[BandSpec, ...]
    schema_version: int = 1

    def __post_init__(self):
        names = [b.name for b in self.bands]
        if not names or len(set(names)) != len(names):
            raise ValueError("Feature names must be nonempty and unique")

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class QuerySpec:
    collections: tuple[str, ...] = ()
    bbox: tuple[float, float, float, float] | None = None
    datetime: str | None = None
    filters: dict = field(default_factory=dict)
    ids: tuple[str, ...] = ()

    def __post_init__(self):
        if self.bbox is not None:
            if len(self.bbox) != 4 or self.bbox[0] >= self.bbox[2] or self.bbox[1] >= self.bbox[3]:
                raise ValueError("bbox must be (west, south, east, north), with positive extent")
        if self.datetime:
            parsed = []
            for part in self.datetime.split("/"):
                if part not in ("..", ""):
                    parsed.append(datetime.fromisoformat(part.replace("Z", "+00:00")))
            if len(parsed) == 2 and parsed[0] > parsed[1]:
                raise ValueError("datetime start must precede end")


@dataclass(frozen=True)
class AssetRef:
    provider: str
    collection: str
    item_id: str
    key: str
    href: str
    revision: str = ""
    checksum: str | None = None
    media_type: str | None = None

    @property
    def identity(self):
        return stable_key([self.provider, self.collection, self.item_id, self.key, self.revision, self.checksum])


@dataclass
class SceneRecord:
    provider: str
    collection: str
    item_id: str
    datetime: str | None
    assets: dict[str, AssetRef]
    properties: dict = field(default_factory=dict)

    @property
    def identity(self):
        return stable_key([self.provider, self.collection, self.item_id])

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data["assets"] = {k: AssetRef(**v) for k, v in data["assets"].items()}
        return cls(**data)
