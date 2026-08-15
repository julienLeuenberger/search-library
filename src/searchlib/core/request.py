from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SearchRequest:
    request_id: int
    parameter: Any
