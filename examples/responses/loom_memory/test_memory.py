"""Offline checks: real SDK clients, synthetic responses, no network or model."""

from __future__ import annotations

import json
from typing import Any
from collections.abc import Callable, AsyncIterator

import httpx
import httpx2
import pytest
from loom import Agent as LoomAgent
from loom.errors import LoomError

from openai import AsyncOpenAI
from examples.responses.loom_memory.memory import PreferenceMemory, respond


def call(name: str, arguments: dict[str, object], call_id: str = "call_fixture") -> dict[str, object]:
    return {
        "id": "fc_" + call_id,
        "type": "function_call",
        "name": name,
        "arguments": json.dumps(arguments),
        "call_id": call_id,
        "status": "completed",
    }


def response(output: list[dict[str, object]], *, status: str = "completed") -> dict[str, object]:
    return {
        "id": "resp_fixture",
        "created_at": 0,
        "object": "response",
        "status": status,
        "model": "fixture-model",
        "output": output,
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "metadata": {},
        "parallel_tool_calls": True,
        "temperature": 1,
        "tool_choice": "auto",
        "tools": [],
        "top_p": 1,
    }


FINAL: dict[str, object] = {
    "id": "msg_final",
    "type": "message",
    "role": "assistant",
    "status": "completed",
    "phase": "final_answer",
    "content": [{"type": "output_text", "text": "Done.", "annotations": []}],
}


@pytest.fixture
async def memory(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[tuple[PreferenceMemory, list[dict[str, Any]], list[httpx.Response]]]:
    requests: list[dict[str, Any]] = []
    replies: list[httpx.Response] = []

    def send(request: httpx.Request) -> httpx.Response:
        requests.append({"path": request.url.path, **json.loads(request.content)})
        assert replies, "Unexpected Loom request"
        return replies.pop(0)

    real_client = httpx.AsyncClient

    def create_client(**kwargs: Any) -> httpx.AsyncClient:
        return real_client(**kwargs, transport=httpx.MockTransport(send), trust_env=False)

    monkeypatch.setattr("loom.transport.httpx.AsyncClient", create_client)
    async with LoomAgent(
        org="fixture-org",
        namespace="fixture-preferences",
        agent="fixture-assistant",
        base_url="https://loom.example.invalid",
        api_key="fixture-not-a-secret",
        auto_run=False,
        mentor_brief=False,
        coalesce_sets=False,
        retry={"max_attempts": 1},
    ) as agent:
        yield PreferenceMemory(agent, user_id="fixture-user"), requests, replies


async def model_client(
    outputs: list[dict[str, object]], requests: list[dict[str, Any]], run: Callable[[AsyncOpenAI], Any]
) -> Any:
    def send(request: httpx2.Request) -> httpx2.Response:
        requests.append(json.loads(request.content))
        assert request.url.path == "/v1/responses"
        assert outputs, "Unexpected model request"
        return httpx2.Response(200, json=outputs.pop(0))

    async with AsyncOpenAI(
        api_key="fixture-not-a-secret",
        base_url="https://openai.example.invalid/v1",
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(send), trust_env=False),
        max_retries=0,
    ) as client:
        return await run(client)


async def test_complete_output_replay_and_actual_write_receipt(memory: Any) -> None:
    tools, loom_requests, replies = memory
    receipt = {"memory_key": "seat", "version": 0, "noop": True, "conflict_verdict": "duplicate"}
    replies.append(httpx.Response(200, json=receipt))
    reasoning: dict[str, object] = {
        "id": "rs_fixture",
        "type": "reasoning",
        "summary": [],
        "encrypted_content": "fixture-opaque",
    }
    commentary = {**FINAL, "id": "msg_comment", "phase": "commentary"}
    first: list[dict[str, object]] = [
        reasoning,
        commentary,
        call("remember_preference", {"memory_key": "seat", "value": "aisle"}, "call_seat"),
    ]
    requests: list[dict[str, Any]] = []
    final = await model_client(
        [response(first), response([FINAL])],
        requests,
        lambda client: respond(client, tools, model="fixture-model", prompt="Remember aisle."),
    )
    assert final.output_text == "Done."
    assert requests[1]["input"][1:4] == first
    assert requests[1]["input"][4] == {
        "type": "function_call_output",
        "call_id": "call_seat",
        "output": json.dumps(receipt),
    }
    assert all(request["store"] is False for request in requests)
    assert all(tool["strict"] is True for tool in requests[0]["tools"])
    assert len(loom_requests) == 1
    assert {
        key: loom_requests[0][key]
        for key in ["path", "org", "namespace", "agent", "user_id", "session_id", "memory_type", "source_class"]
    } == {
        "path": "/v1/memory.set",
        "org": "fixture-org",
        "namespace": "fixture-preferences",
        "agent": "fixture-assistant",
        "user_id": "fixture-user",
        "session_id": "-",
        "memory_type": "preference",
        "source_class": "derived",
    }


async def test_multiple_calls_keep_their_call_ids_and_provenance(memory: Any) -> None:
    tools, loom_requests, replies = memory
    replies.extend(
        [
            httpx.Response(200, json={"version": 1, "noop": False}),
            httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "memory_key": "seat",
                            "memory_type": "preference",
                            "content_excerpt": "aisle",
                            "distance": 0.1,
                            "similarity": 0.9,
                            "source_class": "derived",
                            "source_resource_id": "fixture-source",
                            "verified": False,
                            "user_id": "fixture-user",
                            "session_id": "-",
                        }
                    ],
                    "abstain": False,
                    "retrieval_risk": "low",
                    "pending_extraction": 2,
                    "degraded": ["fixture-fallback"],
                },
            ),
        ]
    )
    requests: list[dict[str, Any]] = []
    await model_client(
        [
            response(
                [
                    call("remember_preference", {"memory_key": "seat", "value": "aisle"}, "call_write"),
                    call("recall_preferences", {"query": "seat"}, "call_read"),
                ]
            ),
            response([FINAL]),
        ],
        requests,
        lambda client: respond(client, tools, model="fixture-model", prompt="Remember and retrieve."),
    )
    outputs = requests[1]["input"][-2:]
    assert [output["call_id"] for output in outputs] == ["call_write", "call_read"]
    recalled = json.loads(outputs[1]["output"])
    assert recalled["memories"][0]["source_class"] == "derived"
    assert recalled["memories"][0]["source_resource_id"] == "fixture-source"
    assert recalled["memories"][0]["verified"] is False
    assert recalled["pending_extraction"] == 2 and recalled["degraded"] == ["fixture-fallback"]
    assert loom_requests[1]["namespace"] == "fixture-preferences" and loom_requests[1]["agent"] == "fixture-assistant"
    assert loom_requests[1]["user_id"] == "fixture-user" and loom_requests[1]["session_id"] == "-"
    assert loom_requests[1]["namespace_scope"] == "this" and loom_requests[1]["auto_touch"] is False


async def test_abstention_is_not_a_successful_memory_answer(memory: Any) -> None:
    tools, _, replies = memory
    replies.append(
        httpx.Response(
            200,
            json={
                "results": [],
                "abstain": True,
                "abstain_code": "no_match",
                "abstain_reason": "No fixture match",
                "retrieval_risk": "abstain",
            },
        )
    )
    result = json.loads(await tools.execute("recall_preferences", '{"query":"seat"}'))
    assert result["memories"] == [] and result["abstain"] is True
    assert result["abstain_code"] == "no_match" and result["retrieval_risk"] == "abstain"


async def test_refusal_is_not_turned_into_a_write_success(memory: Any) -> None:
    tools, _, replies = memory
    replies.append(httpx.Response(403, json={"error": {"code": "FORBIDDEN", "message": "Fixture refusal"}}))
    with pytest.raises(LoomError):
        await tools.execute("remember_preference", '{"memory_key":"seat","value":"aisle"}')


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("delete_preferences", "{}"),
        ("remember_preference", '{"memory_key":"seat","value":"aisle","namespace":"other"}'),
        ("recall_preferences", '{"query":"seat","user_id":"other"}'),
        ("recall_preferences", '{"query":42}'),
        ("recall_preferences", '{"query":" "}'),
        ("recall_preferences", '{"query":"' + "x" * 501 + '"}'),
    ],
)
async def test_arguments_cannot_override_scope_or_expand_tools(memory: Any, name: str, arguments: str) -> None:
    tools, requests, _ = memory
    with pytest.raises(ValueError):
        await tools.execute(name, arguments)
    assert requests == []


async def test_incomplete_response_executes_no_pending_tool(memory: Any) -> None:
    tools, loom_requests, _ = memory
    requests: list[dict[str, Any]] = []
    with pytest.raises(RuntimeError, match="did not complete"):
        await model_client(
            [response([call("remember_preference", {"memory_key": "seat", "value": "aisle"})], status="incomplete")],
            requests,
            lambda client: respond(client, tools, model="fixture-model", prompt="Save."),
        )
    assert loom_requests == []


async def test_loop_is_bounded(memory: Any) -> None:
    tools, _, replies = memory
    replies.extend(httpx.Response(200, json={"results": [], "abstain": True}) for _ in range(4))
    requests: list[dict[str, Any]] = []
    outputs = [response([call("recall_preferences", {"query": "seat"}, f"call_{index}")]) for index in range(4)]
    with pytest.raises(RuntimeError, match="four-request limit"):
        await model_client(
            outputs, requests, lambda client: respond(client, tools, model="fixture-model", prompt="Find.")
        )
    assert len(requests) == 4
