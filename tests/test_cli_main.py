"""Tests for CLI main argument wiring."""

import sys
from importlib import import_module

cli_main = import_module("uiautoagent.cli.main")


def test_demo_ai_assisted_task_passes_max_steps(monkeypatch):
    captured: dict[str, object] = {}

    def fake_run_ai_task(task: str, **kwargs):
        captured["task"] = task
        captured.update(kwargs)

    monkeypatch.setattr(cli_main, "run_ai_task", fake_run_ai_task)

    cli_main.demo_ai_assisted_task(
        "执行任务",
        platform="ios",
        serial="device-1",
        max_steps=12,
        context="上下文",
    )

    assert captured == {
        "task": "执行任务",
        "serial": "device-1",
        "max_steps": 12,
        "platform": "ios",
        "context": "上下文",
        "record_name": None,
    }


def test_main_ai_mode_passes_max_steps(monkeypatch):
    called: dict[str, object] = {}

    def fake_demo_ai_assisted_task(
        task: str,
        platform: str = "android",
        serial: str | None = None,
        max_steps: int = 30,
        context: str | None = None,
        record_name: str | None = None,
    ):
        called.update(
            {
                "task": task,
                "platform": platform,
                "serial": serial,
                "max_steps": max_steps,
                "context": context,
                "record_name": record_name,
            }
        )

    monkeypatch.setattr(cli_main, "check_all_models_available", lambda: True)
    monkeypatch.setattr(cli_main, "demo_ai_assisted_task", fake_demo_ai_assisted_task)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "uiautoagent",
            "--mode",
            "ai",
            "--task",
            "执行任务",
            "--platform",
            "ios",
            "--serial",
            "device-1",
            "--max-steps",
            "15",
        ],
    )

    cli_main.main()

    assert called == {
        "task": "执行任务",
        "platform": "ios",
        "serial": "device-1",
        "max_steps": 15,
        "context": None,
        "record_name": None,
    }
