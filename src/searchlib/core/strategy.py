from abc import ABC, abstractmethod

from .request import SearchRequest
from .observation import Observation
from .result import SearchResult


class SearchStrategy(ABC):

    @abstractmethod
    def start(self) -> None:
        """Initialize the search."""
        ...

    @abstractmethod
    def get_next_request(self) -> SearchRequest | None:
        """Return the next parameter to evaluate."""
        ...

    @abstractmethod
    def submit_observation(self, observation: Observation) -> None:
        """Submit the result of a previous request."""
        ...

    @property
    @abstractmethod
    def is_finished(self) -> bool:
        """Return True when the search has terminated."""
        ...

    @property
    @abstractmethod
    def result(self) -> SearchResult | None:
        """Return the final result, if available."""
        ...
