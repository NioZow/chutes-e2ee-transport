"""Tests for the optional DiscoveryManager.instance_filter selection hook.

These are hermetic: a tiny fake httpx client serves a fixed /e2e/instances body,
so no network or credentials are required.
"""

from __future__ import annotations

import pytest

from chutes_e2ee.discovery import DiscoveryManager


class _Resp:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        return None

    def json(self):
        return self._data


def _instances():
    return [
        {"instance_id": "a", "e2e_pubkey": "pk-a", "nonces": ["a1", "a2"]},
        {"instance_id": "b", "e2e_pubkey": "pk-b", "nonces": ["b1"]},
    ]


class _Client:
    def __init__(self):
        self.calls = 0

    def get(self, url, **kwargs):
        self.calls += 1
        return _Resp({"instances": _instances(), "nonce_expires_in": 60})


class _AsyncClient:
    def __init__(self):
        self.calls = 0

    async def get(self, url, **kwargs):
        self.calls += 1
        return _Resp({"instances": _instances(), "nonce_expires_in": 60})


def test_filter_restricts_selection_and_never_falls_back():
    seen: list[str] = []

    def only_b(chute_id, instances):
        seen.append(chute_id)
        return [i for i in instances if i.instance_id == "b"]

    manager = DiscoveryManager("http://test", "key", instance_filter=only_b)
    client = _Client()
    for _ in range(3):
        instance, nonce = manager.get_nonce("chute-1", client)
        assert instance.instance_id == "b"
        assert nonce == "b1"
    assert seen and all(c == "chute-1" for c in seen)


def test_filter_refuses_when_it_rejects_everything():
    manager = DiscoveryManager("http://test", "key", instance_filter=lambda c, i: [])
    with pytest.raises(RuntimeError, match="rejected every instance"):
        manager.get_nonce("chute-1", _Client())


def test_filter_exception_propagates():
    def boom(chute_id, instances):
        raise RuntimeError("attestation gate refused")

    manager = DiscoveryManager("http://test", "key", instance_filter=boom)
    with pytest.raises(RuntimeError, match="attestation gate refused"):
        manager.get_nonce("chute-1", _Client())


def test_no_filter_is_full_passthrough():
    manager = DiscoveryManager("http://test", "key")
    instance, nonce = manager.get_nonce("chute-1", _Client())
    assert instance.instance_id == "a"
    assert nonce == "a1"


async def test_async_filter_restricts_selection():
    manager = DiscoveryManager(
        "http://test",
        "key",
        instance_filter=lambda c, i: [x for x in i if x.instance_id == "b"],
    )
    client = _AsyncClient()
    instance, nonce = await manager.get_nonce_async("chute-1", client)
    assert instance.instance_id == "b"
    assert nonce == "b1"


async def test_async_filter_refuses_empty_pool():
    manager = DiscoveryManager("http://test", "key", instance_filter=lambda c, i: [])
    with pytest.raises(RuntimeError, match="rejected every instance"):
        await manager.get_nonce_async("chute-1", _AsyncClient())
