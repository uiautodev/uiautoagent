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
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(save_task, range(task_count)))

    data = yaml.safe_load(memory_file.read_text(encoding="utf-8"))

    assert data["total_tasks"] == task_count
    assert len(data["tasks"]) == task_count
    assert {task["task"] for task in data["tasks"]} == {
        f"task-{index}" for index in range(task_count)
    }
