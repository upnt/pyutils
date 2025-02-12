from pyutils.mycollections import nddict
import numpy as np
import itertools as itools

def test_create_nddict():
    keys = [["cat1", "cat2"], ["pat1", "pat2", "pat3"]]
    nd = nddict.from_keys(keys)
    assert nd._values.shape == (2, 3)

def test_nddict01():
    keys = [["cat1", "cat2"], ["pat1", "pat2", "pat3"]]
    mdict = nddict.from_keys(keys)
    mdict["cat1", :] = 10
    mdict["cat2"][1:2] = 20
    expected = np.array([
        [10, 10, 10],
        [None, 20, None],
    ])
    assert np.all(mdict._values == expected)

def test_nddict02():
    categories = ["cat1", "cat2", "cat3"]
    patterns = ["pat1", "pat2", "pat3"]
    results = ["res1", "res2", "res3"]
    mdict = nddict.from_keys([categories, patterns, results])
    mdict["cat1", "pat2", "res1"] = 0
    assert mdict["cat1"]["pat2"]["res1"] == 0
    
    for cat, pat, res in itools.product(categories, patterns, results):
        mdict[cat, pat, res] = f"{cat} {pat} {res}"
    
    assert mdict["cat1"]["pat2"]._values.tolist() == [f"cat1 pat2 {res}" for res in results]
    assert mdict["pat2"]._values.flatten().tolist() == [f"{cat} pat2 {res}" for cat, res in itools.product(categories, results)]
