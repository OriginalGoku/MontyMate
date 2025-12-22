from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .providers import ProviderConfig

JSON = Dict[str, Any]


@dataclass(frozen=True)
class RouteHop:
    provider: str
    model: str


@dataclass(frozen=True)
class RoleRoute:
    role: str
    chain: List[RouteHop]


@dataclass(frozen=True)
class RoutingConfig:
    providers: Dict[str, ProviderConfig]
    roles: Dict[str, RoleRoute]


def load_llm_routing_yaml(yaml_bytes: bytes) -> RoutingConfig:
    raw = yaml.safe_load(yaml_bytes.decode("utf-8"))
    lr = raw.get("llm_routing", {})

    prov_raw = lr.get("providers", {}) or {}
    providers: Dict[str, ProviderConfig] = {}
    for name, cfg in prov_raw.items():
        providers[name] = ProviderConfig(
            name=name,
            kind=str(cfg.get("kind")),
            base_url=str(cfg.get("base_url")),
            api_key_env=cfg.get("api_key_env"),
        )

    roles_raw = lr.get("roles", {}) or {}
    roles: Dict[str, RoleRoute] = {}
    for role, cfg in roles_raw.items():
        chain_raw = cfg.get("chain", []) or []
        chain = [RouteHop(provider=str(h["provider"]), model=str(h["model"])) for h in chain_raw]
        roles[role] = RoleRoute(role=role, chain=chain)

    return RoutingConfig(providers=providers, roles=roles)