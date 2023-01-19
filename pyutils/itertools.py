from typing import Iterable, Callable
import collections
import copy


def minmax(
    iterable: Iterable, *, key: Callable = lambda val: val, default: any = None
) -> any:
    """calculate min and max in iterables

    Args:
        iterable: target iterable
        key: ordinal function
        default: default value

    Returns:
        (min_item, max_item): min and max items
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


def rotate(iterable: Iterable) -> Iterable[Iterable]:
    """return iterables while moving the starting index

    Args:
        iterable: target iterable

    Returns:
        iterables: iterables with the different starting index
    """
    deq = collections.deque(copy.deepcopy(iterable))
    yield iter(deq)
    for _ in range(len(deq) - 1):
        deq.rotate(1)
        yield iter(deq)


def padding(iterables: Iterable[Iterable], value: any = None, is_forward: bool = False):
    """make iterables of the same length by adding value

    Args:
        iterables: target iterables
        value: padding value
        is_forward: if is_forward is true, add value to front

    Returns:
        iterables: iterables of the same length
    """
    max_length = len(max(iterables, key=len))
    for iterable in iterables:
        if len(iterable) < max_length:
            if is_forward:
                iterable = [value] * (max_length - len(iterable)) + iterable
            else:
                iterable = iterable + [value] * (max_length - len(iterable))
        yield iterable


def split(iterable: Iterable, sep: any = None, remove: bool = True):
    """split iterable based on separator

    Args:
        iterable: target iterable
        sep: separator
        remove: if remove is true, removing separator from iterable

    Returns:
        iterables: splited iterables

        input:
            iterable: [0, 1, 2, 3, 2]
            sep: 2
            remove: True
        
        output:
            iterables: [0, 1], [3], []
    """
    result = []
    for val in iterable:
        if val == sep:
            if not remove:
                result.append(val)
            yield result
            result = []
        else:
            result.append(val)
    yield result

def slide(iterable: Iterable, window_size: int):
    """slide iterable

    Args:
        iterable: target iterable
        window_size: number of data retrieved at once

    Returns:
        iterables: tuple iterable

        input:
            iterable: [0, 1, 2, 3, 2]
            window_size: 2
        
        output:
            iterables: [(0, 1), (1, 2), (2, 3), (3, 2)]
    """
    vals = iter(iterable)
    result = collections.deque(
            (next(vals) for _ in range(window_size)),
            maxlen=window_size
    )
    yield result
    for val in vals:
        result.append(val)
        yield result
