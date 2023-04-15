import datetime
from typing import Union

class BaseMetricImpl(object):
    def __init__(self, base_val: Union[int, float]=0) -> None:
        self._val: Union[int, float] = base_val

    def get_val(self) -> Union[int, float]: return self._val

class Counter(BaseMetricImpl):
    def __init__(self, start_value: int=0) -> None:
        super().__init__(start_value)

    def incr(self) -> int:
        self._val += 1
        return self._val

    def decr(self) -> int:
        self._val -= 1
        return self._val

    def add(self, x: int) -> int:
        self._val += x
        return self._val

class Observer(BaseMetricImpl):
    def __init__(self) -> None:
        super().__init__(0)
        self._start: datetime.datetime = None

    def start(self) -> None: self._start = datetime.datetime.now()

    def observe(self) -> float:
        if self._start is None:
            raise "observer wasn't started"
        self._val = (datetime.datetime.now() - self._start).total_seconds()
        return self._val

class Metrics:
    def __init__(self) -> None:
        self._total_ops_cnt: Counter = Counter()
        self._spent: Observer = Observer()

    @property
    def total_ops_cnt(self) -> Counter: return self._total_ops_cnt

    @property
    def spent(self) -> Observer: return self._spent