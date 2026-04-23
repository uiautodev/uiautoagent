"""Tests for executor system prompt."""

from uiautoagent.agent.executor import get_system_prompt


def test_get_system_prompt_contains_common_app_packages():
    prompt = get_system_prompt()

    assert "常用包名" in prompt
    assert "com.tencent.mm" in prompt
    assert "com.tencent.mobileqq" in prompt
