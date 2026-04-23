"""任务记忆管理系统 - 存储和复用任务执行步骤"""

from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path
from typing import List

import yaml

from uiautoagent.agent import TaskStep


def _str_presenter(dumper, data):
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


yaml.add_representer(str, _str_presenter)


class TaskMemory:
    """任务记忆管理系统 - 存储和复用任务执行步骤"""

    def __init__(self, memory_file: str | Path = "task_memory.yaml"):
        self.memory_file = Path(memory_file)
        self._lock = threading.Lock()
        self._memories: List[dict] = self._load_memories()

    def _load_memories(self) -> List[dict]:
        """从文件加载记忆"""
        if self.memory_file.exists():
            try:
                data = yaml.safe_load(self.memory_file.read_text(encoding="utf-8"))
                return data.get("tasks", []) if data else []
            except Exception as e:
                print(f"⚠️  加载任务记忆失败: {e}")
                return []
        return []

    def _save_memories(self):
        """保存记忆到文件"""
        with self._lock:
            self._write_memories_to_file_unlocked()

    def _write_memories_to_file_unlocked(self):
        """将当前记忆写入文件（注意：调用此方法前必须持有self._lock）"""
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "updated_at": datetime.now().isoformat(),
            "total_tasks": len(self._memories),
            "tasks": self._memories,
        }
        # yaml会自动将多行字符串保存为块样式标量（|- 或 |+）
        self.memory_file.write_text(
            yaml.dump(
                data,
                allow_unicode=True,
                sort_keys=False,
                default_flow_style=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def find_similar_tasks(self, task: str, limit: int = 3) -> List[dict]:
        """
        查找相似的历史任务（仅使用字符串精确匹配）

        Args:
            task: 当前任务描述
            limit: 返回数量限制

        Returns:
            相似任务列表，按时间排序
        """
        with self._lock:
            memories_snapshot = list(self._memories)

        if not memories_snapshot:
            return []

        # 字符串完全匹配（同时匹配 clarified_task 和 original_task）
        exact_matches = [
            m
            for m in memories_snapshot
            if m["success"] and (m["task"] == task or m.get("original_task") == task)
        ]
        if exact_matches:
            print(f"💡 找到完全相同的任务 ({len(exact_matches)}个)")
            return sorted(exact_matches, key=lambda x: x["timestamp"], reverse=True)[
                :limit
            ]

        return []

    def save_task(
        self,
        task: str,
        history: list[TaskStep],
        success: bool,
        original_task: str,
        summary: str | None = None,
    ):
        """
        保存任务记忆

        Args:
            task: 任务描述
            history: 执行历史
            success: 是否成功
            summary: Markdown格式的任务总结
            original_task: 用户原始输入的任务描述
        """
        memory = {
            "task": task,
            "original_task": original_task,
            "success": success,
            "timestamp": datetime.now().isoformat(),
            "total_steps": len(history),
            "summary": summary or "",
        }

        with self._lock:
            self._memories.append(memory)
            self._write_memories_to_file_unlocked()

    def find_by_original_task(self, task: str) -> dict | None:
        """
        根据原始任务名查找历史任务（精确匹配 original_task 字段）

        Args:
            task: 用户原始输入的任务描述

        Returns:
            匹配的历史任务，未找到返回 None
        """
        with self._lock:
            memories_snapshot = list(self._memories)

        for m in reversed(memories_snapshot):
            if m.get("original_task") == task:
                return m
        return None

    def format_for_ai(self, similar_tasks: list[dict]) -> str:
        """将相似任务格式化为AI可读的参考信息"""
        if not similar_tasks:
            return ""
        lines = ["## 历史任务参考"]
        for i, task_mem in enumerate(similar_tasks, 1):
            status = "成功" if task_mem["success"] else "失败"
            lines.append(f"### 历史任务 {i}")
            lines.append(f"任务: {task_mem['task']} ({status})")
            summary = task_mem.get("summary", "")
            if summary:
                lines.append(summary)
        lines.append("\n[以上经验可能随APP更新失效，请根据当前屏幕实际情况灵活调整]")
        return "\n".join(lines)


# 全局任务记忆实例
_task_memory = TaskMemory()


def get_task_memory() -> TaskMemory:
    """获取全局任务记忆实例"""
    return _task_memory
