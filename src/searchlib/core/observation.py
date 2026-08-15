from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Observation:
    request_id: int
    parameter: Any
    value: float
