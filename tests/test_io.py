import io
import pytest
from pyutils.myio import default_input, candidate_input
def test_default_input01(capsys, monkeypatch):
    monkeypatch.setattr('sys.stdin', io.StringIO("\n"))
    vals = default_input("> ", [9, 10, 11, 12])
    assert vals == [9, 10, 11, 12]

def test_default_input02(capsys, monkeypatch):
    monkeypatch.setattr('sys.stdin', io.StringIO("15\n"))
    vals = default_input("> ", [9, 10, 11, 12])
    assert vals == [15]

def test_default_input03(capsys, monkeypatch):
    monkeypatch.setattr('sys.stdin', io.StringIO("11,9,15\n"))
    vals = default_input("> ", [9, 10, 11, 12])
    assert vals == [11, 9, 15]

def test_candidate_input01(capsys, monkeypatch):
    monkeypatch.setattr('sys.stdin', io.StringIO("\n"))
    vals = candidate_input("number", [9, 10, 11, 12])
    assert vals == [9, 10, 11, 12]

def test_candidate_input02(capsys, monkeypatch):
    monkeypatch.setattr('sys.stdin', io.StringIO("9,16,11,15\n"))
    try:
        vals = candidate_input("number", [9, 10, 11, 12])
    except ValueError:
        pass
    else:
        assert False

def test_candidate_input03(capsys, monkeypatch):
    monkeypatch.setattr('sys.stdin', io.StringIO("10,9,12\n"))
    vals = candidate_input("number", [9, 10, 11, 12])
    assert vals == [10, 9, 12]
