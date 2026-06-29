"""录制回放模块 - 保存AI操作路径，支持回放和AI fallback"""

from __future__ import annotations

import dictlog
import json
import time
from pathlib import Path

from pydantic import BaseModel, Field

from uiautoagent.agent.device_agent import DeviceAgent, TaskStep
from uiautoagent.agent.plan import ActionType, TaskProposal
from uiautoagent.env import env

log = dictlog.get_logger(__name__)

DEFAULT_RECORDINGS_DIR = Path("recordings")


def _get_recordings_dir() -> Path:
    if env.recordings_dir:
        return Path(env.recordings_dir)
    return DEFAULT_RECORDINGS_DIR


class Recording(BaseModel):
    """一次录制 - 包含完整的操作步骤序列"""

    name: str
    task: str
    platform: str
    device_serial: str | None = None
    resolution_w: int
    resolution_h: int
    created_at: float
    steps: list[TaskStep]


class ReplayConfig(BaseModel):
    """回放配置"""

    fallback_to_ai: bool = Field(
        default=True, description="回放失败时是否切换到AI自主模式"
    )
    skip_wait: bool = Field(default=False, description="是否跳过WAIT动作")


class ReplayStepResult(BaseModel):
    """单步回放结果"""

    step_number: int
    success: bool
    error: str | None = None


class ReplayResult(BaseModel):
    """回放最终结果"""

    success: bool
    replayed_steps: int
    failed_at_step: int | None = None
    step_results: list[ReplayStepResult] = Field(default_factory=list)
    ai_fallback_triggered: bool = False
    final_result: str | None = None
    error: str | None = None


# ---- 录制管理 ----


def save_recording(agent: DeviceAgent, name: str) -> Path:
    """保存当前 agent 的执行历史为一次录制"""
    recordings_dir = _get_recordings_dir()
    recordings_dir.mkdir(parents=True, exist_ok=True)

    info = agent.controller.get_device_info()

    recording = Recording(
        name=name,
        task=agent.task or "",
        platform=info.get("platform", "unknown"),
        device_serial=info.get("serial"),
        resolution_w=info.get("width", 0),
        resolution_h=info.get("height", 0),
        created_at=time.time(),
        steps=agent.history,
    )

    filepath = recordings_dir / f"{name}.json"
    filepath.write_text(
        recording.model_dump_json(indent=2, exclude_none=True),
        encoding="utf-8",
    )
    log.info("录制已保存", name=name, path=str(filepath), steps=len(recording.steps))
    return filepath


def list_recordings() -> list[str]:
    """列出所有可用录制名称"""
    recordings_dir = _get_recordings_dir()
    if not recordings_dir.exists():
        return []
    return sorted([p.stem for p in recordings_dir.glob("*.json") if p.is_file()])


def load_recording(name: str) -> Recording | None:
    """加载指定名称的录制"""
    recordings_dir = _get_recordings_dir()
    filepath = recordings_dir / f"{name}.json"
    if not filepath.exists():
        log.warning("录制文件不存在", name=name, path=str(filepath))
        return None

    raw = json.loads(filepath.read_text(encoding="utf-8"))
    return Recording.model_validate(raw)


# ---- 回放核心 ----


def replay_task(
    agent: DeviceAgent,
    recording: Recording,
    config: ReplayConfig | None = None,
) -> ReplayResult:
    """
    回放录制的操作步骤，失败时可自动切换到AI模式

    Args:
        agent: 设备Agent（已连接设备）
        recording: 要回放的录制数据
        config: 回放配置

    Returns:
        ReplayResult: 回放结果
    """
    config = config or ReplayConfig()

    # 分辨率校验
    actual_info = agent.controller.get_device_info()
    actual_w = actual_info.get("width", 0)
    actual_h = actual_info.get("height", 0)

    if actual_w and actual_h:
        if actual_w != recording.resolution_w or actual_h != recording.resolution_h:
            return ReplayResult(
                success=False,
                replayed_steps=0,
                error=(
                    f"设备分辨率不匹配: "
                    f"录制 {recording.resolution_w}x{recording.resolution_h}, "
                    f"当前 {actual_w}x{actual_h}"
                ),
            )

    total_steps = len(recording.steps)
    log.info("回放开始", recording=recording.name, total_steps=total_steps)

    step_results: list[ReplayStepResult] = []

    try:
        for i, step in enumerate(recording.steps):
            step_num = step.step_number
            action_desc = str(step.action)

            # 终止动作 - 回放完成
            if step.action.type in (ActionType.DONE, ActionType.FAIL):
                log.info("录制结束", step=step_num, action_type=step.action.type.value)
                break

            # 跳过 WAIT
            if config.skip_wait and step.action.type == ActionType.WAIT:
                log.info("跳过WAIT", step=f"{i + 1}/{total_steps}", action=action_desc)
                step_results.append(
                    ReplayStepResult(step_number=step_num, success=True)
                )
                continue

            # 执行操作
            log.info("执行步骤", step=f"{i + 1}/{total_steps}", action=action_desc)
            try:
                screenshot_path = agent._take_screenshot()
                agent._execute_action(step.action, screenshot_path)
            except Exception as e:
                log.error("步骤执行失败", step=step_num, error=str(e))
                step_results.append(
                    ReplayStepResult(
                        step_number=step_num,
                        success=False,
                        error=str(e),
                    )
                )

                if config.fallback_to_ai:
                    return _ai_fallback(
                        agent,
                        recording,
                        steps_done=len(step_results),
                        failed_at_step=step_num,
                        step_results=step_results,
                        step=step,
                    )
                else:
                    return ReplayResult(
                        success=False,
                        replayed_steps=len(step_results),
                        failed_at_step=step_num,
                        step_results=step_results,
                        error=f"执行步骤 {step_num} 失败: {e}",
                    )

            log.info("步骤执行成功", step=step_num)
            step_results.append(ReplayStepResult(step_number=step_num, success=True))

        # 全部回放成功
        log.info("回放完成", replayed_steps=len(step_results))
        return ReplayResult(
            success=True,
            replayed_steps=len(step_results),
            step_results=step_results,
        )

    except Exception as e:
        log.error("回放异常", error=str(e))
        return ReplayResult(
            success=False,
            replayed_steps=len(step_results),
            step_results=step_results,
            error=str(e),
        )


def _ai_fallback(
    agent: DeviceAgent,
    recording: Recording,
    steps_done: int,
    failed_at_step: int,
    step_results: list[ReplayStepResult],
    step: TaskStep,
) -> ReplayResult:
    """回放失败时切换到AI自主模式"""
    from uiautoagent.ai import check_all_models_available
    from uiautoagent.agent.executor import execute_ai_task

    if not check_all_models_available():
        return ReplayResult(
            success=False,
            replayed_steps=steps_done,
            failed_at_step=failed_at_step,
            step_results=step_results,
            error=f"AI fallback 需要但模型未配置 (步骤 {failed_at_step})",
        )

    action_desc = str(step.action)

    user_context = (
        f"录制回放 '{recording.name}' 在第 {failed_at_step} 步失败: "
        f"'{action_desc}'。"
        f"原始任务: {recording.task}。"
        f"请从当前画面继续完成任务。"
    )

    proposal = TaskProposal(
        original_task=recording.task,
        clarified_task=recording.task,
    )
    agent.proposal = proposal
    agent.task = recording.task

    log.info(
        "回放失败，切换到AI模式",
        failed_at_step=failed_at_step,
        action=action_desc,
    )

    result = execute_ai_task(agent, proposal, user_context=user_context)

    ai_success = result.success if result else False
    return ReplayResult(
        success=ai_success,
        replayed_steps=steps_done,
        failed_at_step=failed_at_step,
        step_results=step_results,
        ai_fallback_triggered=True,
        final_result=result.result if result else None,
        error=(
            None if ai_success else f"回放 & AI fallback 均失败 (步骤 {failed_at_step})"
        ),
    )
