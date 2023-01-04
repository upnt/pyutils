from typing import Iterable, Callable
import collections
import copy
import numpy as np


def minmax(
    iterable: Iterable, *, key: Callable = lambda val: val, default: any = None
) -> any:
    """calculate min and max in iterables

    Args:
        iterable: target iterable
        key: ordinal function
        default: default value

    Returns:
        any: _description_
    """
    min_val = default
    min_item = default
    max_val = default
    max_item = default

    for item in iterable:
        val = key(item)
        if min_val is default or val < min_val:
            min_val, min_item = val, item
        if max_val is default or val > max_val:
            max_val, max_item = val, item

    return min_item, max_item


def rotate(iterable: Iterable):
    deq = collections.deque(copy.deepcopy(iterable))
    yield iter(deq)
    for _ in range(len(deq) - 1):
        deq.rotate(1)
        yield iter(deq)


def padding(iterable: Iterable, val=None, reverse=False):
    max_length = len(max(iterable, key=len))
    for item in iterable:
        if len(item) < max_length:
            if reverse:
                item = [val] * (max_length - len(item)) + item
            else:
                item = item + [val] * (max_length - len(item))
        yield item


def split(iterable: Iterable, sep=None, remove=True):
    result = []
    line = []
    for val in iterable:
        if val == sep:
            if not remove:
                line.append(val)
            result.append(line)
            line = []
        else:
            line.append(val)
    return result
