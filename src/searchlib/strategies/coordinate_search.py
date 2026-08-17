from dataclasses import dataclass
from enum import Enum, auto

from searchlib.core.observation import Observation
from searchlib.core.request import SearchRequest
from searchlib.core.result import SearchResult, SearchStatus
from searchlib.core.strategy import SearchStrategy

Parameter = tuple[float, ...]
Values = tuple[float, ...]


class CoordinateSearchState(Enum):
    IDLE = auto()
    SEARCHING = auto()
    COMPLETED = auto()


class CoordinatePhase(Enum):
    MINUS = auto()
    CURRENT = auto()
    PLUS = auto()


@dataclass(frozen=True)
class CoordinateSearchConfig:
    start: Parameter
    stop: Parameter
    coarse_step: Parameter
    fine_step: Parameter
    objective: str = "maximize"


class CoordinateSearch(SearchStrategy):

    def __init__(self, config: CoordinateSearchConfig) -> None:
        self._config = config

        self._state = CoordinateSearchState.IDLE

        self._request_id = 0
        self._current_request: SearchRequest | None = None

        self._observations: list[Observation] = []

        self._position: list[float] = []
        self._steps: list[float] = []
        self._frozen: list[bool] = []

        self._coordinate = 0
        self._phase = CoordinatePhase.MINUS

        self._candidate_values: dict[CoordinatePhase, float] = {}

        self._best_parameter: Parameter | None = None
        self._best_value: Values | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._state != CoordinateSearchState.IDLE:
            raise RuntimeError("Search has already been started.")

        self._validate_config()

        self._position = list(self._config.start)
        self._steps = list(self._config.coarse_step)

        self._frozen = [False] * len(self._position)

        self._coordinate = 0
        self._phase = CoordinatePhase.MINUS

        self._request_id = 0
        self._current_request = None
        self._observations.clear()
        self._candidate_values.clear()

        self._state = CoordinateSearchState.SEARCHING

    def get_next_request(self) -> SearchRequest | None:
        if self._state == CoordinateSearchState.IDLE:
            raise RuntimeError("Search has not been started.")

        if self._state == CoordinateSearchState.COMPLETED:
            return None

        if self._current_request is not None:
            raise RuntimeError(
                "Cannot request a new parameter before "
                "submitting the previous observation."
            )

        self._select_next_coordinate()

        if self._state == CoordinateSearchState.COMPLETED:
            return None

        parameter = self._build_candidate_parameter()

        request = SearchRequest(
            request_id=self._request_id,
            parameter=parameter,
        )

        self._request_id += 1
        self._current_request = request

        return request

    def submit_observation(
        self,
        observation: Observation,
    ) -> None:
        if self._current_request is None:
            raise RuntimeError("No observation is currently expected.")

        if observation.request_id != self._current_request.request_id:
            raise ValueError("Observation does not match the current request.")

        if observation.parameter != self._current_request.parameter:
            raise ValueError(
                "Observation parameter does not match " "the current request."
            )

        if not isinstance(observation.value, tuple):
            raise ValueError("CoordinateSearch expects a tuple of values.")

        if len(observation.value) != len(self._position):
            raise ValueError(
                "Observation dimension does not match " "the search dimension."
            )

        self._observations.append(observation)

        value = observation.value[self._coordinate]

        self._candidate_values[self._phase] = value

        self._current_request = None

        self._advance_phase()

    @property
    def is_finished(self) -> bool:
        return self._state == CoordinateSearchState.COMPLETED

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

    # ------------------------------------------------------------------
    # Candidate generation
    # ------------------------------------------------------------------

    def _build_candidate_parameter(self) -> Parameter:
        parameter = list(self._position)

        coordinate = self._coordinate
        step = self._steps[coordinate]

        if self._phase == CoordinatePhase.MINUS:
            parameter[coordinate] -= step

        elif self._phase == CoordinatePhase.PLUS:
            parameter[coordinate] += step

        parameter[coordinate] = self._clamp(
            parameter[coordinate],
            coordinate,
        )

        return tuple(parameter)

    def _clamp(
        self,
        value: float,
        coordinate: int,
    ) -> float:
        start = self._config.start[coordinate]
        stop = self._config.stop[coordinate]

        return max(start, min(stop, value))

    # ------------------------------------------------------------------
    # Phase management
    # ------------------------------------------------------------------

    def _advance_phase(self) -> None:
        if self._phase == CoordinatePhase.MINUS:
            self._phase = CoordinatePhase.CURRENT
            return

        if self._phase == CoordinatePhase.CURRENT:
            self._phase = CoordinatePhase.PLUS
            return

        self._evaluate_coordinate()

    # ------------------------------------------------------------------
    # Coordinate evaluation
    # ------------------------------------------------------------------

    def _evaluate_coordinate(self) -> None:
        minus = self._candidate_values[CoordinatePhase.MINUS]
        current = self._candidate_values[CoordinatePhase.CURRENT]
        plus = self._candidate_values[CoordinatePhase.PLUS]

        coordinate = self._coordinate
        step = self._steps[coordinate]

        if self._is_better(current, minus) and self._is_better(current, plus):
            self._handle_local_optimum()

        elif self._is_better(minus, current):
            self._position[coordinate] = self._clamp(
                self._position[coordinate] - step,
                coordinate,
            )

        elif self._is_better(plus, current):
            self._position[coordinate] = self._clamp(
                self._position[coordinate] + step,
                coordinate,
            )

        else:
            self._handle_plateau()

        self._candidate_values.clear()
        self._phase = CoordinatePhase.MINUS

    def _handle_local_optimum(self) -> None:
        coordinate = self._coordinate

        if self._steps[coordinate] > self._config.fine_step[coordinate]:
            self._steps[coordinate] = max(
                self._config.fine_step[coordinate],
                self._steps[coordinate] / 2,
            )
        else:
            self._frozen[coordinate] = True

        self._advance_coordinate()

    def _handle_plateau(self) -> None:
        """
        If the three points have exactly the same value,
        reduce the step or freeze the coordinate.
        """
        coordinate = self._coordinate

        if self._steps[coordinate] > self._config.fine_step[coordinate]:
            self._steps[coordinate] = max(
                self._config.fine_step[coordinate],
                self._steps[coordinate] / 2,
            )
        else:
            self._frozen[coordinate] = True

        self._advance_coordinate()

    # ------------------------------------------------------------------
    # Coordinate selection
    # ------------------------------------------------------------------

    def _select_next_coordinate(self) -> None:
        if not self._frozen[self._coordinate]:
            return

        self._advance_coordinate()

    def _advance_coordinate(self) -> None:
        dimension = len(self._position)

        for _ in range(dimension):
            self._coordinate = (self._coordinate + 1) % dimension

            if not self._frozen[self._coordinate]:
                return

        self._finish()

    # ------------------------------------------------------------------
    # Result
    # ------------------------------------------------------------------

    def _finish(self) -> None:
        self._best_parameter = tuple(self._position)

        self._best_value = self._get_best_values()

        self._state = CoordinateSearchState.COMPLETED

    def _get_best_values(self) -> Values:
        dimension = len(self._position)

        best_values: list[float] = []

        for coordinate in range(dimension):
            coordinate_observations = [
                observation
                for observation in self._observations
                if observation.parameter is not None
            ]

            values = [
                observation.value[coordinate] for observation in coordinate_observations
            ]

            if not values:
                raise RuntimeError("No observations available for coordinate.")

            if self._config.objective == "maximize":
                best_values.append(max(values))
            else:
                best_values.append(min(values))

        return tuple(best_values)

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def _is_better(
        self,
        value: float,
        other: float,
    ) -> bool:
        if self._config.objective == "maximize":
            return value > other

        if self._config.objective == "minimize":
            return value < other

        raise ValueError(f"Unknown objective: {self._config.objective}")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_config(self) -> None:
        dimension = len(self._config.start)

        if dimension == 0:
            raise ValueError("Search must have at least one dimension.")

        if len(self._config.stop) != dimension:
            raise ValueError("start and stop must have the same dimension.")

        if len(self._config.coarse_step) != dimension:
            raise ValueError("coarse_step must match the search dimension.")

        if len(self._config.fine_step) != dimension:
            raise ValueError("fine_step must match the search dimension.")

        for start, stop in zip(
            self._config.start,
            self._config.stop,
        ):
            if start >= stop:
                raise ValueError("Every start value must be smaller than stop.")

        for coarse, fine in zip(
            self._config.coarse_step,
            self._config.fine_step,
        ):
            if coarse <= 0:
                raise ValueError("coarse_step values must be positive.")

            if fine <= 0:
                raise ValueError("fine_step values must be positive.")

            if fine >= coarse:
                raise ValueError("fine_step must be smaller than coarse_step.")

        if self._config.objective not in (
            "maximize",
            "minimize",
        ):
            raise ValueError("objective must be 'maximize' or 'minimize'.")
