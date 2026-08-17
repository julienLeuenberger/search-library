import math
import threading

from searchlib.core.observation import Observation
from searchlib.strategies.coarse_fine import (
    CoarseFineConfig,
    CoarseFineSearch,
)


def run_search(search, function):
    search.start()

    while True:
        request = search.get_next_request()

        if request is None:
            break

        value = function(request.parameter)

        search.submit_observation(
            Observation(
                request_id=request.request_id,
                parameter=request.parameter,
                value=value,
            )
        )

    return search.result


def test_finds_maximum():
    search = CoarseFineSearch(
        CoarseFineConfig(
            start=0,
            stop=100,
            coarse_step=20,
            fine_step=5,
            objective="maximize",
        )
    )

    result = run_search(
        search,
        lambda x: -((x - 63) ** 2),
    )

    assert result is not None
    assert result.best_parameter == 65
    assert result.best_value == -4


def test_finds_minimum():
    search = CoarseFineSearch(
        CoarseFineConfig(
            start=0,
            stop=100,
            coarse_step=20,
            fine_step=5,
            objective="minimize",
        )
    )

    result = run_search(
        search,
        lambda x: (x - 63) ** 2,
    )

    assert result is not None
    assert result.best_parameter == 65
    assert result.best_value == 4


def test_maximum_at_start():
    search = CoarseFineSearch(
        CoarseFineConfig(
            start=0,
            stop=100,
            coarse_step=20,
            fine_step=5,
            objective="maximize",
        )
    )

    result = run_search(
        search,
        lambda x: -((x + 10) ** 2),
    )

    assert result is not None
    assert result.best_parameter == 0


def test_maximum_at_stop():
    search = CoarseFineSearch(
        CoarseFineConfig(
            start=0,
            stop=100,
            coarse_step=20,
            fine_step=5,
            objective="maximize",
        )
    )

    result = run_search(
        search,
        lambda x: -((x - 110) ** 2),
    )

    assert result is not None
    assert result.best_parameter == 100


def test_fine_search_improves_coarse_result():
    search = CoarseFineSearch(
        CoarseFineConfig(
            start=0,
            stop=100,
            coarse_step=20,
            fine_step=5,
            objective="maximize",
        )
    )

    result = run_search(
        search,
        lambda x: -((x - 63) ** 2),
    )

    assert result is not None

    # Coarse search would find 60.
    # Fine search should improve this to 65.
    assert result.best_parameter != 60
    assert result.best_parameter == 65


# not really interesting
def test_finds_maximum_with_noise():
    values = {
        0: 0.1,
        20: 0.3,
        40: 0.6,
        60: 0.9,
        80: 0.5,
        100: 0.2,
    }

    search = CoarseFineSearch(
        CoarseFineConfig(
            start=0,
            stop=100,
            coarse_step=20,
            fine_step=5,
            objective="maximize",
        )
    )

    # this curious lambda x: values.get(x, 0.8) is equivalent to the following function:
    # def function(x):
    #     if x in values:
    #         return values[x]

    #     return 0.8

    result = run_search(
        search,
        lambda x: values.get(x, 0.8),
    )

    assert result is not None
    assert result.best_parameter == 60


def test_four_motor_position_searches():
    """
    Simulate four independent coarse/fine searches over
    the motor range 2200..4800.

    Each motor has its own focus curve and therefore
    its own optimal position.
    """

    target_positions = [3810, 3820, 3840, 3850]

    def make_focus_measure(target, peak):
        def focus_measure(position):
            return peak * math.exp(-(((position - target) / 100) ** 2))

        return focus_measure

    focus_measures = [
        make_focus_measure(3810, 1.00),
        make_focus_measure(3820, 0.85),
        make_focus_measure(3840, 1.20),
        make_focus_measure(3850, 0.95),
    ]

    results = []

    for focus_measure in focus_measures:
        search = CoarseFineSearch(
            CoarseFineConfig(
                start=2200,
                stop=4800,
                coarse_step=100,
                fine_step=10,
                objective="maximize",
            )
        )

        result = run_search(search, focus_measure)

        assert result is not None
        results.append(result)

    positions = [result.best_parameter for result in results]

    assert positions == [
        3810,
        3820,
        3840,
        3850,
    ]


def test_four_motor_position_searches_parallel():
    """
    Simulate four independent coarse/fine searches running
    concurrently, one per motor.
    """

    target_positions = [3810, 3820, 3840, 3850]

    def make_focus_measure(target: int, peak: float):
        def focus_measure(position: int) -> float:
            return peak * math.exp(-(((position - target) / 100) ** 2))

        return focus_measure

    focus_measures = [
        make_focus_measure(3810, 1.00),
        make_focus_measure(3820, 0.85),
        make_focus_measure(3840, 1.20),
        make_focus_measure(3850, 0.95),
    ]

    results = [None] * 4
    errors = [None] * 4

    def search_motor(
        motor_number: int,
        focus_measure,
    ) -> None:
        try:
            search = CoarseFineSearch(
                CoarseFineConfig(
                    start=2200,
                    stop=4800,
                    coarse_step=100,
                    fine_step=10,
                    objective="maximize",
                )
            )

            search.start()

            while True:
                request = search.get_next_request()

                if request is None:
                    break

                assert request is not None

                value = focus_measure(request.parameter)

                search.submit_observation(
                    Observation(
                        request_id=request.request_id,
                        parameter=request.parameter,
                        value=value,
                    )
                )

            results[motor_number] = search.result

        except Exception as error:
            errors[motor_number] = error

    threads = [
        threading.Thread(
            target=search_motor,
            args=(motor_number, focus_measure),
            name=f"MotorSearch-{motor_number + 1}",
        )
        for motor_number, focus_measure in enumerate(focus_measures)
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    # Make failures inside the worker threads visible to pytest.
    for error in errors:
        if error is not None:
            raise error

    assert all(result is not None for result in results)

    positions = [result.best_parameter for result in results]

    print("\nParallel search results:")

    for motor_number, position in enumerate(positions, start=1):
        print(
            f"  Motor {motor_number}: "
            f"{position} "
            f"(target={target_positions[motor_number - 1]})"
        )

    assert positions == target_positions
