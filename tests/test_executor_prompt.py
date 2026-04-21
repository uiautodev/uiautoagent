"""Tests for executor system prompt."""

from uiautoagent.agent.executor import get_system_prompt


def test_get_system_prompt_contains_common_app_packages():
    prompt = get_system_prompt()

    assert "常用应用包名参考" in prompt
    assert "微信：Android `com.tencent.mm`，iOS `com.tencent.xin`" in prompt
    assert "QQ：Android `com.tencent.mobileqq`，iOS `com.tencent.mqq`" in prompt
