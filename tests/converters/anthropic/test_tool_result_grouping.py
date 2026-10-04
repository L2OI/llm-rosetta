"""Preserve complete tool-result batches in Anthropic request history."""

import copy
import json

import pytest

from llm_rosetta.pipeline import ConversionPipeline


def chat_request(count=2):
    """Build one complete Chat tool turn without provider-specific fields."""
    calls = [
        {
            "id": f"call_{i}",
            "type": "function",
            "function": {"name": "read", "arguments": json.dumps({"path": f"{i}.txt"})},
        }
        for i in range(count)
    ]
    return {
        "model": "demo",
        "max_tokens": 64,
        "messages": [
            {"role": "user", "content": "Read the files."},
            {"role": "assistant", "content": None, "tool_calls": calls},
            *[
                {"role": "tool", "tool_call_id": call["id"], "content": f"result {i}"}
                for i, call in enumerate(calls)
            ],
        ],
    }


@pytest.mark.parametrize("count", [1, 2, 3])
def test_parallel_tool_results_share_the_immediately_following_user_message(count):
    request = chat_request(count)
    original = copy.deepcopy(request)
    converted = ConversionPipeline("openai_chat", "anthropic").convert_request(request)

    assert request == original
    assert [m["role"] for m in converted["messages"]] == ["user", "assistant", "user"]
    results = converted["messages"][2]["content"]
    assert [b["tool_use_id"] for b in results] == [f"call_{i}" for i in range(count)]
    assert [b["content"] for b in results] == [f"result {i}" for i in range(count)]


def test_tool_result_grouping_preserves_turns_and_normal_user_messages():
    request = chat_request()
    second = chat_request()
    for call in second["messages"][1]["tool_calls"]:
        call["id"] += "_second"
    for result in second["messages"][2:]:
        result["tool_call_id"] += "_second"
    request["messages"].extend(second["messages"])
    request["messages"].extend(
        [
            {"role": "user", "content": "First note"},
            {"role": "user", "content": "Second note"},
        ]
    )

    converted = ConversionPipeline("openai_chat", "anthropic").convert_request(request)
    messages = converted["messages"]
    assert [m["role"] for m in messages] == [
        "user",
        "assistant",
        "user",
        "user",
        "assistant",
        "user",
        "user",
        "user",
    ]
    assert [b["tool_use_id"] for b in messages[2]["content"]] == ["call_0", "call_1"]
    assert [b["tool_use_id"] for b in messages[5]["content"]] == [
        "call_0_second",
        "call_1_second",
    ]
    assert messages[6]["content"] == [{"type": "text", "text": "First note"}]
    assert messages[7]["content"] == [{"type": "text", "text": "Second note"}]


def test_parallel_tool_results_survive_native_and_cross_protocol_roundtrips():
    request = chat_request()
    native = {
        "model": "demo",
        "max_tokens": 64,
        "messages": [
            {"role": "user", "content": "Read the files."},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": f"call_{i}",
                        "name": "read",
                        "input": {"path": f"{i}.txt"},
                    }
                    for i in range(2)
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": f"call_{i}",
                        "content": f"result {i}",
                    }
                    for i in range(2)
                ],
            },
        ],
    }
    for source, body in [("anthropic", native), ("openai_chat", request)]:
        first = ConversionPipeline(source, "anthropic").convert_request(body)
        chat = ConversionPipeline("anthropic", "openai_chat").convert_request(first)
        restored = ConversionPipeline("openai_chat", "anthropic").convert_request(chat)
        assert len(restored["messages"]) == 3
        assert [b["tool_use_id"] for b in restored["messages"][2]["content"]] == [
            "call_0",
            "call_1",
        ]
