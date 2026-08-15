from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

from ..core.observation import Observation
from ..core.request import SearchRequest
from ..core.result import SearchResult, SearchStatus
from ..core.strategy import SearchStrategy


class CoarseFineState(Enum):
    IDLE = auto()
    COARSE = auto()
    FINE = auto()
    COMPLETED = auto()


@dataclass(frozen=True)
class CoarseFineConfig:
    start: float
    stop: float
    coarse_step: float
    fine_step: float
    objective: str = "maximize"


class CoarseFineSearch(SearchStrategy):

    def __init__(self, config: CoarseFineConfig):
        self._config = config

        self._state = CoarseFineState.IDLE

        self._request_id = 0
        self._current_request: SearchRequest | None = None

        self._observations: list[Observation] = []

        self._coarse_best_parameter: Any | None = None
        self._coarse_best_value: float | None = None
        self._best_value: float | None = None

        self._coarse_best_parameter: Any | None = None

    def start(self) -> None:
        if self._state != CoarseFineState.IDLE:
            raise RuntimeError("Search has already been started.")

        self._validate_config()

        self._state = CoarseFineState.COARSE

        self._next_parameter = self._config.start

    def get_next_request(self) -> SearchRequest | None:
        if self._state == CoarseFineState.IDLE:
            raise RuntimeError("Search has not been started.")

        if self._state == CoarseFineState.COMPLETED:
            return None

        if self._current_request is not None:
            raise RuntimeError(
                "Cannot request a new parameter before "
                "submitting the previous observation."
            )

        if self._state == CoarseFineState.COARSE:
            parameter = self._get_next_coarse_parameter()

            if parameter is None:
                self._start_fine_search()
                return self.get_next_request()

        elif self._state == CoarseFineState.FINE:
            parameter = self._get_next_fine_parameter()

            if parameter is None:
                self._state = CoarseFineState.COMPLETED
                return None

        else:
            return None

        request = SearchRequest(
            request_id=self._request_id,
            parameter=parameter,
        )

        self._request_id += 1
        self._current_request = request

        return request

    def submit_observation(self, observation: Observation) -> None:
        if self._current_request is None:
            raise RuntimeError("No observation is currently expected.")

        if observation.request_id != self._current_request.request_id:
            raise ValueError("Observation does not match the current request.")

        if observation.parameter != self._current_request.parameter:
            raise ValueError(
                "Observation parameter does not match the current request."
            )

        self._observations.append(observation)

        if self._state == CoarseFineState.COARSE:
            self._update_coarse_best(observation)

        self._update_best(observation)

        self._current_request = None

    @property
    def is_finished(self) -> bool:
        return self._state == CoarseFineState.COMPLETED

    @property
    def result(self) -> SearchResult | None:
        if not self.is_finished:
            return None

        return SearchResult(
            status=SearchStatus.COMPLETED,
            best_parameter=self._best_parameter,
            best_value=self._best_value,
            observations=tuple(self._observations),
        )

    def _get_next_coarse_parameter(self) -> float | None:
        parameter = self._next_parameter

        if parameter > self._config.stop:
            return None

        self._next_parameter += self._config.coarse_step

        return parameter

    def _get_next_fine_parameter(self) -> float | None:
        parameter = self._next_parameter

        if parameter > self._fine_stop:
            return None

        self._next_parameter += self._config.fine_step

        return parameter

    def _start_fine_search(self) -> None:
        if self._coarse_best_parameter is None:
            raise RuntimeError("No coarse search result available.")

        half_range = self._config.coarse_step

        self._fine_start = max(
            self._config.start,
            self._coarse_best_parameter - half_range,
        )

        self._fine_stop = min(
            self._config.stop,
            self._coarse_best_parameter + half_range,
        )

        self._next_parameter = self._fine_start

        self._state = CoarseFineState.FINE

    def _update_best(self, observation: Observation) -> None:
        if self._best_value is None:
            self._best_parameter = observation.parameter
            self._best_value = observation.value
            return

        if self._config.objective == "maximize":
            is_better = observation.value > self._best_value
        elif self._config.objective == "minimize":
            is_better = observation.value < self._best_value
        else:
            raise ValueError(f"Unknown objective: {self._config.objective}")

        if is_better:
            self._best_parameter = observation.parameter
            self._best_value = observation.value

    def _update_coarse_best(self, observation: Observation) -> None:
        if self._coarse_best_value is None:
            self._coarse_best_parameter = observation.parameter
            self._coarse_best_value = observation.value
            return

        if self._config.objective == "maximize":
            is_better = observation.value > self._coarse_best_value
        else:
            is_better = observation.value < self._coarse_best_value

        if is_better:
            self._coarse_best_parameter = observation.parameter
            self._coarse_best_value = observation.value

    def _validate_config(self) -> None:
        if self._config.start >= self._config.stop:
            raise ValueError("start must be smaller than stop.")

        if self._config.coarse_step <= 0:
            raise ValueError("coarse_step must be positive.")

        if self._config.fine_step <= 0:
            raise ValueError("fine_step must be positive.")

        if self._config.fine_step >= self._config.coarse_step:
            raise ValueError("fine_step must be smaller than coarse_step.")

        if self._config.objective not in ("maximize", "minimize"):
            raise ValueError("objective must be 'maximize' or 'minimize'.")
