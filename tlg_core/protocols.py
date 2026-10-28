"""Public structural interfaces for external extensions."""
from typing import Protocol, Iterable, Any
from .contracts import AssetRef, QuerySpec, SceneRecord, GridSpec


class Provider(Protocol):
    def search(self, query: QuerySpec) -> Iterable[SceneRecord]: ...
    def prepare_access(self, asset: AssetRef) -> dict: ...


class Transport(Protocol):
    def fetch(self, access: dict, destination: str, *, timeout: float = 300,
              progress=None, cancelled=None) -> str: ...


class RasterSource(Protocol):
    @property
    def grid(self) -> GridSpec: ...
    def read(self, window=None, bands=None) -> Any: ...


class Scorer(Protocol):
    def fit(self, target, labels, **params): ...
    def score(self, target, labels, **params): ...
