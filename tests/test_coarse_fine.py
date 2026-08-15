import pytest
from searchlib.strategies.coarse_fine import (
    CoarseFineConfig,
    CoarseFineSearch,
)
from searchlib.core.observation import Observation


def test_coarse_fine_finds_maximum():
    config = CoarseFineConfig(
        start=0,
        stop=100,
        coarse_step=20,
        fine_step=5,
        objective="maximize",
    )

    search = CoarseFineSearch(config)
    search.start()

    def function(x):
        return -((x - 63) ** 2)

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

    assert search.is_finished

    result = search.result

    assert result is not None
    assert result.best_parameter == 65
    assert result.best_value == -4


def create_search():
    return CoarseFineSearch(
        CoarseFineConfig(
            start=0,
            stop=100,
            coarse_step=20,
            fine_step=5,
            objective="maximize",
        )
    )


def submit(search, request, value):
    search.submit_observation(
        Observation(
            request_id=request.request_id,
            parameter=request.parameter,
            value=value,
        )
    )


def test_coarse_search_parameters():
    search = create_search()
    search.start()

    parameters = []

    while True:
        request = search.get_next_request()

        if request is None:
            break

        parameters.append(request.parameter)

        # Arbitrary value: we only want to test the requested parameters.
        submit(search, request, 0.0)

        # Stop avant la phase fine.
        if len(parameters) == 6:
            break

    assert parameters == [0, 20, 40, 60, 80, 100]


def test_request_ids_are_unique():
    search = create_search()
    search.start()

    request_ids = []

    while True:
        request = search.get_next_request()

        if request is None:
            break

        request_ids.append(request.request_id)

        submit(search, request, 0.0)

        if len(request_ids) == 6:
            break

    assert request_ids == [0, 1, 2, 3, 4, 5]


def test_fine_search_uses_coarse_best():
    search = create_search()
    search.start()

    # Coarse:
    # 0   -> 0.0
    # 20  -> 0.1
    # 40  -> 0.5
    # 60  -> 1.0  <-- best
    # 80  -> 0.2
    # 100 -> 0.0

    coarse_values = {
        0: 0.0,
        20: 0.1,
        40: 0.5,
        60: 1.0,
        80: 0.2,
        100: 0.0,
    }

    for _ in range(6):
        request = search.get_next_request()

        assert request is not None

        submit(
            search,
            request,
            coarse_values[request.parameter],
        )

    # The coarse best is 60.
    # With coarse_step = 20, the current implementation
    # searches from 40 to 80 during the fine phase.
    fine_parameters = []

    while True:
        request = search.get_next_request()

        if request is None:
            break

        fine_parameters.append(request.parameter)

        submit(search, request, 0.0)

    assert fine_parameters == [40, 45, 50, 55, 60, 65, 70, 75, 80]


def test_search_is_finished_after_fine_search():
    search = create_search()
    search.start()

    while True:
        request = search.get_next_request()

        if request is None:
            break

        submit(search, request, 0.0)

    assert search.is_finished


def test_result_is_none_before_search_is_finished():
    search = create_search()

    assert search.result is None

    search.start()

    assert search.result is None


def test_result_contains_all_observations():
    search = create_search()
    search.start()

    observation_count = 0

    while True:
        request = search.get_next_request()

        if request is None:
            break

        submit(search, request, 0.0)
        observation_count += 1

    result = search.result

    assert result is not None
    assert len(result.observations) == observation_count


def test_cannot_request_twice_without_observation():
    search = create_search()
    search.start()

    request = search.get_next_request()

    assert request is not None

    with pytest.raises(RuntimeError):
        search.get_next_request()


def test_observation_must_match_request():
    search = create_search()
    search.start()

    request = search.get_next_request()

    assert request is not None

    observation = Observation(
        request_id=request.request_id + 1,
        parameter=request.parameter,
        value=1.0,
    )

    with pytest.raises(ValueError):
        search.submit_observation(observation)
