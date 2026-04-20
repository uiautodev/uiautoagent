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
        """将当前记忆写入文件（调用方需在外部持有self._lock）"""
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
        查找相似的历史任务

        策略：
        1. 先用字符串完全匹配查找
        2. 如果找不到，再用AI查找
        3. 如果AI也找不到，返回空

        Args:
            task: 当前任务描述
            limit: 返回数量限制

        Returns:
            相似任务列表，按相似度排序
        """
        with self._lock:
            memories_snapshot = list(self._memories)

        if not memories_snapshot:
            return []

        # 步骤1：字符串完全匹配
        exact_matches = [
            m for m in memories_snapshot if m["success"] and m["task"] == task
        ]
        if exact_matches:
            print(f"💡 找到完全相同的任务 ({len(exact_matches)}个)")
            # 按时间排序，返回最新的
            return sorted(exact_matches, key=lambda x: x["timestamp"], reverse=True)[
                :limit
            ]

        # 步骤2：使用AI查找相似任务
        try:
            from uiautoagent.ai import Category, chat_completion

            # 构建历史任务列表（只返回成功任务）
            successful_tasks = [
                {"index": i, "task": m["task"], "summary": m.get("summary", "")}
                for i, m in enumerate(memories_snapshot)
                if m["success"]
            ]

            if not successful_tasks:
                return []

            # 构建AI提示
            tasks_list = "\n".join(
                [f"{i}. {t['task']}" for i, t in enumerate(successful_tasks)]
            )

            prompt = f"""你是一个任务相似度分析专家。请从以下历史任务列表中，找出与当前任务最相似的{limit}个任务。

当前任务：{task}

历史任务列表：
{tasks_list}

请以JSON格式返回最相似任务的索引号（按相似度从高到低排序）：
{{
  "similar_indices": [索引号1, 索引号2, ...],
  "reasoning": "简短说明为什么这些任务相似"
}}

只返回索引号，不要返回任务内容。"""

            response = chat_completion(
                category=Category.TEXT,
                messages=[
                    {"role": "system", "content": "你是一个任务相似度分析专家。"},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                max_tokens=512,
                temperature=0.0,
            )

            content = response.choices[0].message.content
            if not content:
                return []

            result = yaml.safe_load(content)
            indices = result.get("similar_indices", [])

            if not indices:
                print("💡 AI未找到相似任务")
                return []

            # 根据索引获取对应的记忆（需要转换回原始索引）
            similar_memories = []
            for idx in indices[:limit]:
                if 0 <= idx < len(successful_tasks):
                    original_idx = successful_tasks[idx]["index"]
                    similar_memories.append(memories_snapshot[original_idx])

            if similar_memories:
                print(f"💡 AI找到相似任务: {result.get('reasoning', '')}")

            return similar_memories

        except Exception as e:
            print(f"⚠️  AI相似度分析失败: {e}")
            return []

    def save_task(
        self,
        task: str,
        history: list[TaskStep],
        success: bool,
        summary: str | None = None,
    ):
        """
        保存任务记忆

        Args:
            task: 任务描述
            history: 执行历史
            success: 是否成功
            summary: Markdown格式的任务总结
        """
        memory = {
            "task": task,
            "success": success,
            "timestamp": datetime.now().isoformat(),
            "total_steps": len(history),
            "summary": summary or "",
        }

        with self._lock:
            self._memories.append(memory)
            self._write_memories_to_file_unlocked()

    def format_for_ai(self, similar_tasks: list[dict]) -> str:
        """将相似任务格式化为AI可读的参考信息"""
        if not similar_tasks:
            return "（无相关历史任务）"

        lines = ["## 相似历史任务经验参考"]
        for i, task_mem in enumerate(similar_tasks, 1):
            status = "✅ 成功" if task_mem["success"] else "❌ 失败"
            lines.append(f"\n### {i}. {task_mem['task']} - {status}")

            summary = task_mem.get("summary", "")
            if summary:
                # 直接使用Markdown内容
                lines.append(summary)
            else:
                lines.append("- 无总结信息")

        # 在末尾添加过期提示
        lines.append(
            "\n**⚠️ 注意: 以上经验可能随APP更新失效，请根据当前屏幕实际情况灵活调整**"
        )

        return "\n".join(lines)


# 全局任务记忆实例
_task_memory = TaskMemory()


def get_task_memory() -> TaskMemory:
    """获取全局任务记忆实例"""
    return _task_memory
