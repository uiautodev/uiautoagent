"""Tests for AI model selection and fallback."""

from importlib import import_module, reload
from unittest.mock import MagicMock

import pytest
from openai import APIConnectionError, RateLimitError


@pytest.fixture
def ai_module(monkeypatch):
    monkeypatch.delenv("UIAUTO_MODEL_NAME", raising=False)
    monkeypatch.delenv("MODEL_NAME", raising=False)
    monkeypatch.delenv("UIAUTO_MODEL_VISION", raising=False)
    monkeypatch.delenv("MODEL_VISION", raising=False)
    monkeypatch.delenv("UIAUTO_MODEL_TEXT", raising=False)
    monkeypatch.delenv("MODEL_TEXT", raising=False)

    module = import_module("uiautoagent.ai")
    module._get_ai_client.cache_clear()
    return reload(module)


def _make_chat_response(content: str = "ok") -> MagicMock:
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    return resp


def test_get_ai_model_returns_category_candidates(monkeypatch, ai_module):
    monkeypatch.setenv("UIAUTO_MODEL_NAME", "default-a,default-b")
    monkeypatch.setenv("UIAUTO_MODEL_VISION", "vision-a, vision-b")

    module = reload(ai_module)

    assert module.get_ai_model(module.Category.VISION) == ["vision-a", "vision-b"]


def test_get_ai_model_falls_back_to_default_candidates(monkeypatch, ai_module):
    monkeypatch.setenv("UIAUTO_MODEL_NAME", "default-a, default-b")

    module = reload(ai_module)

    assert module.get_ai_model(module.Category.TEXT) == ["default-a", "default-b"]
    assert module.get_ai_model() == ["default-a", "default-b"]


def test_chat_completion_falls_back_to_next_model(ai_module, monkeypatch):
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        APIConnectionError(request=None),
        _make_chat_response("done"),
    ]
    monkeypatch.setattr(ai_module, "_get_ai_client", lambda: client)
    monkeypatch.setattr(ai_module, "get_ai_model", lambda *_: ["model-a", "model-b"])

    response = ai_module.chat_completion(
        category=ai_module.Category.TEXT,
        messages=[{"role": "user", "content": "hello"}],
    )

    assert response.choices[0].message.content == "done"
    assert client.chat.completions.create.call_count == 2
    first_call = client.chat.completions.create.call_args_list[0].kwargs
    second_call = client.chat.completions.create.call_args_list[1].kwargs
    assert first_call["model"] == "model-a"
    assert second_call["model"] == "model-b"


def test_chat_completion_raises_last_error_when_all_models_fail(ai_module, monkeypatch):
    client = MagicMock()
    last_error = RateLimitError(
        message="rate limited",
        response=MagicMock(status_code=429),
        body=None,
    )
    client.chat.completions.create.side_effect = [
        APIConnectionError(request=None),
        last_error,
    ]
    monkeypatch.setattr(ai_module, "_get_ai_client", lambda: client)
    monkeypatch.setattr(ai_module, "get_ai_model", lambda *_: ["model-a", "model-b"])

    with pytest.raises(RateLimitError):
        ai_module.chat_completion(
            category=ai_module.Category.TEXT,
            messages=[{"role": "user", "content": "hello"}],
        )


def test_chat_completion_uses_explicit_model_candidates(ai_module, monkeypatch):
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        APIConnectionError(request=None),
        _make_chat_response("done"),
    ]
    monkeypatch.setattr(ai_module, "_get_ai_client", lambda: client)

    response = ai_module.chat_completion(
        category=ai_module.Category.TEXT,
        model="manual-a, manual-b",
        messages=[{"role": "user", "content": "hello"}],
    )

    assert response.choices[0].message.content == "done"
    assert (
        client.chat.completions.create.call_args_list[0].kwargs["model"] == "manual-a"
    )
    assert (
        client.chat.completions.create.call_args_list[1].kwargs["model"] == "manual-b"
    )


def test_chat_completion_does_not_retry_on_non_retryable_error(ai_module, monkeypatch):
    client = MagicMock()
    client.chat.completions.create.side_effect = ValueError("bad request")
    monkeypatch.setattr(ai_module, "_get_ai_client", lambda: client)
    monkeypatch.setattr(ai_module, "get_ai_model", lambda *_: ["model-a", "model-b"])

    with pytest.raises(ValueError, match="bad request"):
        ai_module.chat_completion(
            category=ai_module.Category.TEXT,
            messages=[{"role": "user", "content": "hello"}],
        )
    assert client.chat.completions.create.call_count == 1


def test_check_all_models_available_accepts_one_candidate_per_group(
    ai_module, monkeypatch
):
    monkeypatch.setenv("UIAUTO_MODEL_NAME", "default-a,default-b")
    monkeypatch.setenv("UIAUTO_MODEL_VISION", "vision-a,vision-b")
    monkeypatch.setenv("UIAUTO_MODEL_TEXT", "text-a")
    module = reload(ai_module)

    checked_models: list[str] = []

    def fake_check(model: str) -> bool:
        checked_models.append(model)
        return model in {"default-b", "vision-b", "text-a"}

    monkeypatch.setattr(module, "check_model_available", fake_check)

    assert module.check_all_models_available() is True
    assert checked_models == [
        "default-a",
        "default-b",
        "vision-a",
        "vision-b",
        "text-a",
    ]
