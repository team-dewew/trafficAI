from evaluate import match_segments, prf, tiou


def test_tiou():
    assert tiou((0, 2), (0, 2)) == 1.0
    assert tiou((0, 2), (2, 4)) == 0.0
    assert tiou((0, 2), (1, 3)) == 1 / 3
    assert tiou((0, 4), (1, 3)) == 2 / 4

def test_prf():
    res = prf(10, 0, 0)
    assert res["precision"] == 1.0
    assert res["recall"] == 1.0
    assert res["f1"] == 1.0

    res = prf(5, 5, 5)
    assert res["precision"] == 0.5
    assert res["recall"] == 0.5
    assert res["f1"] == 0.5

def test_match_segments():
    # 1 GT, 1 Pred, exact match
    tp, fp, fn = match_segments([(0, 2)], [(0, 2)], 0.5)
    assert (tp, fp, fn) == (1, 0, 0)

    # 1 GT, 1 Pred, no overlap
    tp, fp, fn = match_segments([(0, 2)], [(2, 4)], 0.5)
    assert (tp, fp, fn) == (0, 1, 1)

    # 1 GT, 2 Pred (one match, one FP)
    tp, fp, fn = match_segments([(0, 2)], [(0, 2), (4, 6)], 0.5)
    assert (tp, fp, fn) == (1, 1, 0)

    # 2 GT, 1 Pred (one match, one FN)
    tp, fp, fn = match_segments([(0, 2), (4, 6)], [(4.1, 6.1)], 0.5)
    assert (tp, fp, fn) == (1, 0, 1)
