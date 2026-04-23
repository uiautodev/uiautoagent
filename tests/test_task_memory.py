"""Tests for TaskMemory."""

from concurrent.futures import ThreadPoolExecutor

import yaml

from uiautoagent.agent.memory import TaskMemory


def test_save_task_thread_safe(tmp_path):
    """Concurrent saves should not lose task memories."""
    memory_file = tmp_path / "task_memory.yaml"
    memory = TaskMemory(memory_file)
    task_count = 50

    def save_task(index: int):
        memory.save_task(
            task=f"task-{index}",
            history=[],
            success=True,
            summary=f"summary-{index}",
            original_task=f"task-{index}",
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(save_task, range(task_count)))

    data = yaml.safe_load(memory_file.read_text(encoding="utf-8"))

    assert data["total_tasks"] == task_count
    assert len(data["tasks"]) == task_count
    assert {task["task"] for task in data["tasks"]} == {
        f"task-{index}" for index in range(task_count)
    }


def test_save_task_with_original_task(tmp_path):
    """save_task should store original_task separately from clarified task."""
    memory_file = tmp_path / "task_memory.yaml"
    memory = TaskMemory(memory_file)

    memory.save_task(
        task="打开微信应用",
        history=[],
        success=True,
        summary="test summary",
        original_task="打开微信",
    )

    data = yaml.safe_load(memory_file.read_text(encoding="utf-8"))
    task_entry = data["tasks"][0]

    assert task_entry["original_task"] == "打开微信"
    assert task_entry["task"] == "打开微信应用"


def test_find_similar_tasks_matches_original(tmp_path):
    """find_similar_tasks should match against original_task field."""
    memory_file = tmp_path / "task_memory.yaml"
    memory = TaskMemory(memory_file)

    memory.save_task(
        task="打开微信应用",
        history=[],
        success=True,
        summary="test",
        original_task="打开微信",
    )

    # Searching with the original task should match
    results = memory.find_similar_tasks("打开微信")
    assert len(results) == 1
    assert results[0]["original_task"] == "打开微信"
