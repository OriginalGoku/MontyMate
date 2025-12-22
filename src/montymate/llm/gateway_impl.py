from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from montymate.config.resolver import RunConfigResolver
from montymate.core.types import JSON
from montymate.data.paths import RepoPaths
from montymate.data.db import transaction
from montymate.data.services import DataServices
from montymate.llm.gateway import LLMGateway
from montymate.llm.models import LLMRequest, LLMResponse
from montymate.llm.providers.types import ProviderConfig
from montymate.llm.providers.openai_chat import openai_chat_complete
from montymate.llm.providers.openai_compat_chat import openai_compat_chat_complete
from montymate.llm.providers.ollama_chat import ollama_chat_complete


@dataclass
class LLMGatewayImpl(LLMGateway):
    conn: sqlite3.Connection
    paths: RepoPaths
    resolver: RunConfigResolver
    services_factory: Any  # your make_services callable from CLI

    def complete(self, *, run_id: str, step_execution_id: str, req: LLMRequest, decision_record: JSON) -> LLMResponse:
        run_row = self.conn.execute("SELECT * FROM mm_runs WHERE run_id=?", (run_id,)).fetchone()
        if not run_row:
            return LLMResponse(status="ERROR", provider="unknown", model="unknown", text="", raw={"error": "run not found"})

        repo_root = str(run_row["repo_root"])
        profile_selector = str(run_row["profile_id"])

        config = self.resolver.resolve(repo_root=repo_root, profile_selector=profile_selector)

        role_cfg = (config.llm_routing.roles or {}).get(req.role)
        if not role_cfg:
            return LLMResponse(status="ERROR", provider="unknown", model="unknown", text="", raw={"error": f"no routing for role={req.role}"})

        chain: List[JSON] = role_cfg.get("chain") or []
        if not chain:
            return LLMResponse(status="ERROR", provider="unknown", model="unknown", text="", raw={"error": f"empty chain for role={req.role}"})

        providers_cfg: Dict[str, JSON] = (config.llm_routing.raw.get("llm_routing") or {}).get("providers") or {}
        last_err: Optional[JSON] = None

        for hop in chain:
            provider_name = str(hop.get("provider"))
            model = str(hop.get("model"))

            p = providers_cfg.get(provider_name)
            if not p:
                last_err = {"error": f"unknown provider '{provider_name}'"}
                continue

            provider = ProviderConfig(
                name=provider_name,
                kind=str(p.get("kind")),
                base_url=str(p.get("base_url")),
                api_key_env=p.get("api_key_env"),
                api_key_value=p.get("api_key_value"),
            )

            # Provider call
            if provider.kind == "openai_chat":
                pr = openai_chat_complete(
                    cfg=provider,
                    model=model,
                    messages=req.messages,
                    max_output_tokens=req.max_output_tokens,
                    temperature=req.temperature,
                )
            elif provider.kind == "openai_compat_chat":
                pr = openai_compat_chat_complete(
                    cfg=provider,
                    model=model,
                    messages=req.messages,
                    max_output_tokens=req.max_output_tokens,
                    temperature=req.temperature,
                )
            elif provider.kind == "ollama_chat":
                pr = ollama_chat_complete(
                    cfg=provider,
                    model=model,
                    messages=req.messages,
                    max_output_tokens=req.max_output_tokens,
                    temperature=req.temperature,
                )
            else:
                last_err = {"error": f"unsupported provider kind '{provider.kind}'"}
                continue

            # Record call (prompt+response stored inline or as artifacts per storage_policy)
            svc: DataServices = self.services_factory(
                conn=self.conn,
                paths=self.paths,
                run_id=run_id,
                repo_root=repo_root,
                profile_id=str(run_row["profile_id"]),
                workflow_id=str(run_row["workflow_id"]),
                policy_id=str(run_row["policy_id"]),
                decision_record=decision_record,
            )

            with transaction(self.conn, write=True):
                svc.llm.record_call(
                    run_id=run_id,
                    step_execution_id=step_execution_id,
                    role=req.role,
                    provider=provider_name,
                    model=model,
                    status="OK" if pr.status == "OK" else "ERROR",
                    prompt_text=json.dumps(req.messages, ensure_ascii=False),
                    response_text=pr.text,
                    input_tokens=pr.input_tokens,
                    output_tokens=pr.output_tokens,
                    total_tokens=pr.total_tokens,
                    cost_usd=None,  # MVP: you can compute later from pricing tables; OpenAI usage tokens are recorded.
                    template_id=req.template_id,
                    pricing_id=None,
                    request_json={"messages": req.messages, "model": model},
                    response_json=pr.raw,
                )

            if pr.status == "OK":
                return LLMResponse(
                    status="OK",
                    provider=provider_name,
                    model=model,
                    text=pr.text,
                    raw=pr.raw,
                    input_tokens=pr.input_tokens,
                    output_tokens=pr.output_tokens,
                    total_tokens=pr.total_tokens,
                    cost_usd=None,
                )

            last_err = pr.raw

        return LLMResponse(status="ERROR", provider="unknown", model="unknown", text="", raw={"error": "all providers failed", "last": last_err})