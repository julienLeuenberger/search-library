import pytest

from searchlib.core.observation import Observation


def test_observation_stores_values():
    observation = Observation(
        request_id=42,
        parameter=1250,
        value=0.873,
    )

    assert observation.request_id == 42
    assert observation.parameter == 1250
    assert observation.value == 0.873


def test_observation_is_immutable():
    observation = Observation(
        request_id=42,
        parameter=1250,
        value=0.873,
    )

    with pytest.raises(AttributeError):
        observation.value = 0.9


def test_equal_observations_are_equal():
    observation_1 = Observation(
        request_id=42,
        parameter=1250,
        value=0.873,
    )

    observation_2 = Observation(
        request_id=42,
        parameter=1250,
        value=0.873,
    )

    assert observation_1 == observation_2
