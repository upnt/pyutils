from pyutils.itertools import minmax
def test_minmax():
    assert minmax(range(-1, 10)) == (-1, 9)
    assert minmax([-1.0, 4, -10.5, 3, 90.2, 0]) == (-10.5, 90.2)
