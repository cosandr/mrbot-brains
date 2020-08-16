from typing import NamedTuple

from ext.response import Response


class ExpectedJSON(NamedTuple):
    comment: str
    input: dict
    output: str


def test_to_json():
    expected = [
        ExpectedJSON(
            comment="String data",
            input=dict(data="some string"),
            output='{"data": "some string"}',
        ),
        ExpectedJSON(
            comment="List data",
            input=dict(data=["item 1", "item 2"]),
            output='{"data": ["item 1", "item 2"]}',
        ),
        ExpectedJSON(
            comment="Dict data",
            input=dict(data=dict(words=10, text="output text")),
            output='{"data": {"words": 10, "text": "output text"}}',
        ),
        ExpectedJSON(
            comment="Only error",
            input=dict(error="Argument error"),
            output='{"error": "Argument error"}',
        ),
        ExpectedJSON(
            comment="Data with time",
            input=dict(data="timed data", time=12.6),
            output='{"data": "timed data", "time": 12.6}',
        ),
        ExpectedJSON(
            comment="Empty",
            input={},
            output='{}',
        ),
    ]
    for e in expected:
        actual = Response(**e.input).to_json()
        assert e.output == actual, e.comment
