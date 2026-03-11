"""
Instance discovery, nonce management, and model-name resolution for Chutes E2EE.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import httpx


@dataclass
class InstanceInfo:
    instance_id: str
    e2e_pubkey: str  # base64
    nonces: list[str]


@dataclass
class DiscoveryResult:
    instances: list[InstanceInfo]
    nonce_expires_at: float  # unix timestamp


@dataclass
class _CachedNonces:
    """Per-chute nonce pool with expiry tracking."""

    instances: list[InstanceInfo]
    expires_at: float
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def take_nonce(self) -> tuple[InstanceInfo, str] | None:
        """Consume one nonce, returning (instance, nonce) or None if exhausted/expired."""
        if time.time() >= self.expires_at:
            return None
        with self._lock:
            for inst in self.instances:
                if inst.nonces:
                    return inst, inst.nonces.pop(0)
        return None

    @property
    def remaining(self) -> int:
        return sum(len(i.nonces) for i in self.instances)

    @property
    def expired(self) -> bool:
        return time.time() >= self.expires_at


class DiscoveryManager:
    """Manages instance discovery, nonce caching, and model-name resolution.

    This class is thread-safe and shared across requests within a transport.
    """

    def __init__(self, api_base: str, api_key: str, models_base: str | None = None):
        self._api_base = api_base.rstrip("/")
        self._models_base = (models_base or api_base).rstrip("/")
        self._api_key = api_key
        self._auth_headers = {"Authorization": f"Bearer {api_key}"}

        # chute_id -> _CachedNonces
        self._nonce_cache: dict[str, _CachedNonces] = {}
        self._cache_lock = threading.Lock()

        # model_name -> chute_id (populated lazily from /v1/models)
        self._model_map: dict[str, str] = {}
        self._model_map_loaded_at: float = 0
        self._model_map_lock = threading.Lock()
        self._MODEL_MAP_TTL = 300  # refresh every 5 min

    # ------------------------------------------------------------------
    # Model name -> chute_id resolution
    # ------------------------------------------------------------------

    def resolve_chute_id(self, model: str, client: httpx.Client) -> str:
        """Resolve a model name to a chute_id UUID.

        If *model* looks like a UUID it is returned as-is. Otherwise we
        look it up via the /v1/models listing (cached). Raises if not found.
        """
        if self._looks_like_uuid(model):
            return model

        self._maybe_refresh_model_map(client)
        chute_id = self._model_map.get(model)
        if chute_id:
            return chute_id

        # Force a refresh in case the map was stale.
        self._model_map_loaded_at = 0
        self._maybe_refresh_model_map(client)
        chute_id = self._model_map.get(model)
        if chute_id:
            return chute_id

        raise ValueError(
            f"Could not resolve model {model!r} to a chute_id. "
            "Check that the model name is correct and available at /v1/models."
        )

    def _maybe_refresh_model_map(self, client: httpx.Client) -> None:
        now = time.time()
        if now - self._model_map_loaded_at < self._MODEL_MAP_TTL:
            return
        with self._model_map_lock:
            if now - self._model_map_loaded_at < self._MODEL_MAP_TTL:
                return
            resp = client.get(
                f"{self._models_base}/v1/models",
                headers=self._auth_headers,
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json().get("data", [])
            new_map: dict[str, str] = {}
            for entry in data:
                model_id = entry.get("id")
                chute_id = entry.get("chute_id")
                if model_id and chute_id:
                    new_map[model_id] = chute_id
            self._model_map = new_map
            self._model_map_loaded_at = time.time()

    # ------------------------------------------------------------------
    # Instance discovery + nonce management
    # ------------------------------------------------------------------

    def get_nonce(self, chute_id: str, client: httpx.Client) -> tuple[InstanceInfo, str]:
        """Get an (instance, nonce) pair, fetching new ones if needed."""
        with self._cache_lock:
            cached = self._nonce_cache.get(chute_id)

        if cached and not cached.expired:
            result = cached.take_nonce()
            if result:
                return result

        discovery = self._fetch_instances(chute_id, client)
        cached = _CachedNonces(
            instances=discovery.instances,
            expires_at=discovery.nonce_expires_at,
        )
        with self._cache_lock:
            self._nonce_cache[chute_id] = cached

        result = cached.take_nonce()
        if not result:
            raise RuntimeError(
                f"No nonces available for chute {chute_id}. "
                "The chute may have no active E2EE-capable instances."
            )
        return result

    def _fetch_instances(self, chute_id: str, client: httpx.Client) -> DiscoveryResult:
        resp = client.get(
            f"{self._api_base}/e2e/instances/{chute_id}",
            headers=self._auth_headers,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        instances = [
            InstanceInfo(
                instance_id=inst["instance_id"],
                e2e_pubkey=inst["e2e_pubkey"],
                nonces=list(inst["nonces"]),
            )
            for inst in data["instances"]
        ]
        return DiscoveryResult(
            instances=instances,
            nonce_expires_at=time.time() + data.get("nonce_expires_in", 55),
        )

    # ------------------------------------------------------------------
    # Async variants
    # ------------------------------------------------------------------

    async def resolve_chute_id_async(self, model: str, client: httpx.AsyncClient) -> str:
        if self._looks_like_uuid(model):
            return model

        await self._maybe_refresh_model_map_async(client)
        chute_id = self._model_map.get(model)
        if chute_id:
            return chute_id

        # Force refresh.
        self._model_map_loaded_at = 0
        await self._maybe_refresh_model_map_async(client)
        chute_id = self._model_map.get(model)
        if chute_id:
            return chute_id

        raise ValueError(
            f"Could not resolve model {model!r} to a chute_id. "
            "Check that the model name is correct and available at /v1/models."
        )

    async def _maybe_refresh_model_map_async(self, client: httpx.AsyncClient) -> None:
        now = time.time()
        if now - self._model_map_loaded_at < self._MODEL_MAP_TTL:
            return
        resp = await client.get(
            f"{self._models_base}/v1/models",
            headers=self._auth_headers,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        new_map: dict[str, str] = {}
        for entry in data:
            model_id = entry.get("id")
            chute_id = entry.get("chute_id")
            if model_id and chute_id:
                new_map[model_id] = chute_id
        self._model_map = new_map
        self._model_map_loaded_at = time.time()

    async def get_nonce_async(
        self, chute_id: str, client: httpx.AsyncClient
    ) -> tuple[InstanceInfo, str]:
        with self._cache_lock:
            cached = self._nonce_cache.get(chute_id)

        if cached and not cached.expired:
            result = cached.take_nonce()
            if result:
                return result

        discovery = await self._fetch_instances_async(chute_id, client)
        cached = _CachedNonces(
            instances=discovery.instances,
            expires_at=discovery.nonce_expires_at,
        )
        with self._cache_lock:
            self._nonce_cache[chute_id] = cached

        result = cached.take_nonce()
        if not result:
            raise RuntimeError(
                f"No nonces available for chute {chute_id}. "
                "The chute may have no active E2EE-capable instances."
            )
        return result

    async def _fetch_instances_async(
        self, chute_id: str, client: httpx.AsyncClient
    ) -> DiscoveryResult:
        resp = await client.get(
            f"{self._api_base}/e2e/instances/{chute_id}",
            headers=self._auth_headers,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        instances = [
            InstanceInfo(
                instance_id=inst["instance_id"],
                e2e_pubkey=inst["e2e_pubkey"],
                nonces=list(inst["nonces"]),
            )
            for inst in data["instances"]
        ]
        return DiscoveryResult(
            instances=instances,
            nonce_expires_at=time.time() + data.get("nonce_expires_in", 55),
        )

    @staticmethod
    def _looks_like_uuid(s: str) -> bool:
        parts = s.split("-")
        if len(parts) != 5:
            return False
        try:
            int(s.replace("-", ""), 16)
            return len(s) == 36
        except ValueError:
            return False
