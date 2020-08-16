from typing import NamedTuple

import pytest
from aiohttp import ClientSession


class Expected(NamedTuple):
    comment: str
    input: str
    output: dict


async def test_list_best():
    url = "http://localhost:7762/be/list/best"
    async with ClientSession().get(url=url) as resp:
        assert resp.status == 200, resp.reason
        data = await resp.json()
    assert data.get("data")
    assert len(data["data"]) > 0


async def test_list_all():
    url = "http://localhost:7762/be/list/all"
    async with ClientSession().get(url=url) as resp:
        assert resp.status == 200, resp.reason
        data = await resp.json()
    assert data.get("data")
    assert len(data["data"]) > 0


@pytest.mark.skip
async def test_run():
    url = "http://localhost:7762/be/run/"
    expected = [
        Expected(
            comment="jens, default args",
            input=url + "jens",
            output=dict(words=50, temp=0.5),
        ),
        Expected(
            comment="jens, 10 words",
            input=url + "jens?words=10",
            output=dict(words=10, temp=0.5),
        ),
        Expected(
            comment="jens, 10 words, 1.0 temp",
            input=url + "jens?words=10&temp=1.0",
            output=dict(words=10, temp=1.0),
        ),
    ]
    async with ClientSession() as sess:
        for e in expected:
            async with sess.get(e.input) as resp:
                assert resp.status == 200, resp.reason
                data = await resp.json()
            # Did we get any data?
            assert data.get("data")
            # Did we get any params?
            assert data.get("params")
            # Check that parameters parsed correctly
            assert data["params"]["words"] == e.output["words"]
            assert data["params"]["temp"] == e.output["temp"]
            assert data["params"]["model"].startswith("jens")
