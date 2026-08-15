import pytest

from searchlib.core.request import SearchRequest


def test_request_stores_values():
    request = SearchRequest(
        request_id=42,
        parameter=1250,
    )

    assert request.request_id == 42
    assert request.parameter == 1250


def test_request_is_immutable():
    request = SearchRequest(
        request_id=42,
        parameter=1250,
    )

    with pytest.raises(AttributeError):
        request.parameter = 1300


def test_equal_requests_are_equal():
    request_1 = SearchRequest(
        request_id=42,
        parameter=1250,
    )

    request_2 = SearchRequest(
        request_id=42,
        parameter=1250,
    )

    assert request_1 == request_2


def test_request_accepts_generic_parameter():
    assert SearchRequest(1, 42).parameter == 42
    assert SearchRequest(2, 12.5).parameter == 12.5
    assert SearchRequest(3, (10, 20)).parameter == (10, 20)
