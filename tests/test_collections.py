from pyutils.collections import nddict
import numpy as np
import itertools as itools
def test_nddict():
    categories = ["cat1", "cat2", "cat3"]
    patterns = ["pat1", "pat2", "pat3"]
    results = ["res1", "res2", "res3"]
    mdict = nddict()
    mdict.add_key(categories)
    mdict.add_key(patterns)
    mdict.add_key(results)
    mdict["cat1"]["pat2"]["res1"] = 0
    assert mdict["cat1"]["pat2"]["res1"] == 0
    
    for cat, pat, res in itools.product(categories, patterns, results):
        mdict[cat][pat][res] = f"{cat} {pat} {res}"
    
    assert mdict["cat1"]["pat2"]._values.tolist() == [f"cat1 pat2 {res}" for res in results]
    assert mdict["pat2"]._values.flatten().tolist() == [f"{cat} pat2 {res}" for cat, res in itools.product(categories, results)]
