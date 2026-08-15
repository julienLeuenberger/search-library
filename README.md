# Search-Library
To test the library, you can run the following code snippet:
```bash
pip install e .
pytest -v
```


```python
strategy.start()

while not strategy.is_finished:
    request = strategy.get_next_request()

    # Application / hardware / simulation
    value = evaluate(request.parameter)

    observation = Observation(
        request_id=request.request_id,
        parameter=request.parameter,
        value=value,
    )

    strategy.submit_observation(observation)

result = strategy.result



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
```

```code
┌─────────────────────────────────────────────────────┐
│                   SEARCH LIBRARY                    │
│                                                     │
│  SearchStrategy                                     │
│       │                                             │
│       ▼                                             │
│  CoarseFineSearch                                   │
│       │                                             │
│       ├──────────► SearchRequest                    │
│       │                  │                          │
│       │                  ▼                          │
│       │             external world                  │
│       │                  │                          │
│       │                  ▼                          │
│       └──────────► Observation                      │
│                                                     │
│                       │                             │
│                       ▼                             │
│                  SearchResult                       │
└─────────────────────────────────────────────────────┘
```
```
                         <<abstract>>
                       SearchStrategy
                              │
             ┌────────────────┴────────────────┐
             │                                 │
     CoarseFineSearch                    HillClimbSearch
             │
             │ creates
             ▼
      ┌───────────────┐
      │ SearchRequest │
      └───────┬───────┘
              │
              │ external system
              ▼
      ┌───────────────┐
      │ Observation   │
      └───────┬───────┘
              │
              │ submitted to
              ▼
      ┌────────────────┐
      │ SearchStrategy │
      └───────┬────────┘
              │
              │ produces
              ▼
       ┌──────────────┐
       │ SearchResult │
       └──────────────┘
```
```
                 SEARCH LIBRARY
        ┌──────────────────────────┐
        │                          │
        │      SearchStrategy      │
        │                          │
        └────────────┬─────────────┘
                     │
                     │ SearchRequest
                     ▼
              ┌─────────────┐
              │ APPLICATION  │
              └──────┬──────┘
                     │
            does whatever is needed
                     │
                     ▼
              ┌─────────────┐
              │ REAL WORLD  │
              └──────┬──────┘
                     │
                     │ measured value
                     ▼
              ┌─────────────┐
              │ APPLICATION │
              └──────┬──────┘
                     │
                     │ Observation
                     ▼
        ┌──────────────────────────┐
        │      SearchStrategy      │
        └──────────────────────────┘
```
```
                 Search
                   │
                   ▼
             Exploration
                   │
                   ▼
            Best candidate
                   │
                   ▼
              Refinement
                   │
                   ▼
            Best candidate
                   │
                  ...
                   │
                   ▼
                Finish
```
```
Search           Application          External world
  │                   │                     │
  │ start()           │                     │
  │◄──────────────────│                     │
  │                   │                     │
  │ get_next_request()│                     │
  │──────────────────►│                     │
  │ SearchRequest     │                     │
  │                   │                     │
  │                   │ execute request     │
  │                   │────────────────────►│
  │                   │                     │
  │                   │    measured value   │
  │                   │◄────────────────────│
  │                   │                     │
  │ submit_observation│                     │
  │◄──────────────────│                     │
  │                   │                     │
  │ calculate next    │                     │
  │                   │                     │
  │ get_next_request()│                     │
  │──────────────────►│                     │
  │        ...        │                     │
```
