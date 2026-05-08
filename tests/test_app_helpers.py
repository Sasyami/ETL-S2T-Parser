"""Unit tests for pure helpers in app (no HTTP, minimal DB)."""
import datetime
import numpy as np
import pandas as pd
import pytest

from app import (
    allowed_file,
    build_nested_columns,
    convert_to_serializable,
    is_empty_or_irrelevant,
)


def test_allowed_file_accepted():
    assert allowed_file("report.xlsx") is True
    assert allowed_file("data.xls") is True
    assert allowed_file("macro.xlsm") is True


def test_allowed_file_rejected():
    assert allowed_file("note.txt") is False
    assert allowed_file("noextension") is False
    assert allowed_file("") is False


def test_convert_to_serializable_datetime():
    dt = datetime.datetime(2024, 6, 1, 12, 30, 0)
    assert convert_to_serializable(dt) == dt.isoformat()
    d = datetime.date(2024, 6, 1)
    assert convert_to_serializable(d) == d.isoformat()


def test_convert_to_serializable_numpy():
    assert convert_to_serializable(np.int64(42)) == 42
    assert convert_to_serializable(np.float64(3.5)) == 3.5
    assert convert_to_serializable(np.bool_(True)) is True
    arr = np.array([1, 2])
    assert convert_to_serializable(arr) == [1, 2]


def test_convert_to_serializable_nested():
    payload = {
        "nums": np.array([1.0]),
        "nested": [np.int32(7), None],
    }
    out = convert_to_serializable(payload)
    assert out == {"nums": [1.0], "nested": [7, None]}


def test_build_nested_columns_merges_forward_fill():
    df = pd.DataFrame(
        [
            ["A", "A", "B"],
            ["x", "y", "z"],
        ]
    )
    cols = build_nested_columns(df, header_rows=2)
    assert cols[0] == ["A", "x"]
    assert cols[1] == ["A", "y"]
    assert cols[2] == ["B", "z"]


def test_build_nested_columns_empty():
    assert build_nested_columns(pd.DataFrame(), header_rows=1) == []


def test_is_empty_or_irrelevant_empty_list():
    ok, reason = is_empty_or_irrelevant([], "S")
    assert ok is True
    assert "empty" in reason.lower()


def test_is_empty_or_irrelevant_whitespace_only():
    ok, reason = is_empty_or_irrelevant([[" ", ""], [None, None]], "S")
    assert ok is True
    assert "whitespace" in reason.lower()


def test_is_empty_or_irrelevant_has_data():
    ok, reason = is_empty_or_irrelevant([[None, "Name"], [1, 2]], "S")
    assert ok is False
    assert reason == ""
