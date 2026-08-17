from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

from .observation import Observation


class SearchStatus(Enum):
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


@dataclass(frozen=True)
class SearchResult:
    status: SearchStatus
    best_parameter: Any | None
    best_value: Any | None
    observations: tuple[Observation, ...]
