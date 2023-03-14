from pyutils.collections import nddict
import numpy as np
def test_nddict():
    dictionary = nddict()
    dictionary.add_key("category", ["cat1", "cat2", "cat3"])
    dictionary.add_key("pattern", ["pat1", "pat2", "pat3"])
    dictionary.add_key("result", ["res1", "res2", "res3"])
    assert np.all(dictionary._array == np.array([None for _ in range(3*3*3)]).reshape((3, 3, 3)))
    dictionary._array = np.array(range(3*3*3)).reshape((3, 3, 3))
    print(dictionary._array)
    assert np.all(dictionary["cat1"] == np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]))
    assert np.all(dictionary["pat1"] == np.array([[0, 1, 2], [9, 10, 11], [18, 19, 20]]))
    assert np.all(dictionary["res1"] == np.array([[0, 3, 6], [9, 12, 15], [18, 21, 24]]))
