import math

from searchlib.core.observation import Observation
from searchlib.strategies.coordinate_search import (
    CoordinateSearch,
    CoordinateSearchConfig,
)


def run_search(search, function):
    search.start()

    while True:
        request = search.get_next_request()

        if request is None:
            break

        values = function(request.parameter)

        search.submit_observation(
            Observation(
                request_id=request.request_id,
                parameter=request.parameter,
                value=values,
            )
        )

    return search.result


def test_finds_four_motor_positions():
    targets = (2750, 3320, 4010, 4520)

    def focus_measure(
        position: tuple[float, ...],
    ) -> tuple[float, ...]:
        return tuple(
            math.exp(-(((position[i] - targets[i]) / 100) ** 2)) for i in range(4)
        )

    search = CoordinateSearch(
        CoordinateSearchConfig(
            start=(2200, 2200, 2200, 2200),
            stop=(4800, 4800, 4800, 4800),
            coarse_step=(200, 200, 200, 200),
            fine_step=(10, 10, 10, 10),
        )
    )

    search.start()

    iteration = 0

    while True:
        request = search.get_next_request()

        if request is None:
            print("\nSearch finished.")
            break

        iteration += 1

        position = request.parameter
        values = focus_measure(position)

        print(
            f"[{iteration:03d}] "
            f"position={position} "
            f"focus={tuple(round(v, 3) for v in values)}"
        )

        search.submit_observation(
            Observation(
                request_id=request.request_id,
                parameter=position,
                value=values,
            )
        )

    result = search.result

    assert result is not None

    print(f"\nResult:")
    print(f"  position = {result.best_parameter}")
    print(f"  focus    = {result.best_value}")
    print(f"  target   = {targets}")

    assert result.best_parameter == targets
