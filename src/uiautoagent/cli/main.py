"""设备Agent - AI自主执行手机任务（命令行入口）"""

from __future__ import annotations

import argparse

from uiautoagent.agent import Action, ActionType, AgentConfig, DeviceAgent
from uiautoagent.agent.plan import TapParams, WaitParams, InputParams
from uiautoagent.agent.executor import run_ai_task
from uiautoagent.ai import check_all_models_available
from uiautoagent.controller import AndroidController, IOSController


def demo_manual_control(platform: str = "android", serial: str | None = None):
    """演示手动控制Agent执行任务（适用于已知步骤的任务）"""
    print("=" * 50)
    print("📱 设备Agent - 手动控制模式")
    print("=" * 50)

    # 检查设备
    if platform == "ios":
        if serial:
            controller = IOSController(udid=serial)
        else:
            devices = IOSController.list_devices()
            if not devices:
                print("❌ 未检测到iOS设备")
                return
            controller = IOSController(udid=devices[0])
    else:
        devices = AndroidController.list_devices()
        if not devices:
            print("❌ 未检测到Android设备，请确保ADB已连接")
            return
        serial = serial or devices[0]
        controller = AndroidController(serial)

    print("✅ 检测到设备")

    # 创建Agent
    agent = DeviceAgent(
        controller,
        config=AgentConfig(
            max_steps=20,
            save_screenshots=True,
        ),
    )

    info = controller.get_device_info()
    print(f"📋 设备信息: {info['model']} ({info['width']}x{info['height']})\n")

    # 示例：打开应用并执行操作（手动步骤）
    steps = [
        Action(
            type=ActionType.TAP,
            thought="打开应用",
            params=TapParams(target="微信图标"),
        ),
        Action(
            type=ActionType.WAIT,
            thought="等待应用启动",
            params=WaitParams(wait_ms=2000),
        ),
        Action(
            type=ActionType.TAP,
            thought="点击搜索框",
            params=TapParams(target="搜索框"),
        ),
        Action(
            type=ActionType.INPUT,
            thought="输入搜索关键词",
            params=InputParams(text="test"),
        ),
        Action(
            type=ActionType.DONE,
            thought="任务完成",
        ),
    ]

    # 执行步骤
    for action in steps:
        agent.step(action)

    # 保存历史
    agent.save_history()
    agent.print_summary()


def demo_ai_assisted_task(
    task: str = "修改昵称为kitty",
    platform: str = "android",
    serial: str | None = None,
    max_steps: int = 30,
    context: str | None = None,
):
    """
    演示AI辅助任务执行 - AI自主决策并完成任务

    Args:
        task: 要执行的任务描述
        platform: 设备平台
        serial: 设备序列号/UDID
        max_steps: 最大执行步数
        context: 用户提供的任务上下文
    """
    run_ai_task(
        task, serial=serial, max_steps=max_steps, platform=platform, context=context
    )


def demo_find_and_click(
    target: str = "返回按钮", platform: str = "android", serial: str | None = None
):
    """演示简单的查找并点击"""
    print("=" * 50)
    print("📱 设备Agent - 查找并点击")
    print("=" * 50)

    if platform == "ios":
        if serial:
            controller = IOSController(udid=serial)
        else:
            devices = IOSController.list_devices()
            if not devices:
                print("❌ 未检测到iOS设备")
                return
            controller = IOSController(udid=devices[0])
    else:
        devices = AndroidController.list_devices()
        if not devices:
            print("❌ 未检测到Android设备")
            return
        controller = AndroidController(devices[0])
    agent = DeviceAgent(controller)

    # 查找并点击元素
    agent.step(
        Action(
            type=ActionType.TAP,
            thought=f"查找并点击{target}",
            params=TapParams(target=target),
        )
    )

    agent.save_history()


def main():
    """Main entry point for the uiautoagent CLI."""
    parser = argparse.ArgumentParser(
        description="设备Agent - AI自主执行手机任务",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=["manual", "ai", "find"],
        default="find",
        help="运行模式",
    )
    parser.add_argument(
        "-t",
        "--task",
        default="修改昵称为kitty",
        help="要执行的任务描述（ai/find模式使用）",
    )
    parser.add_argument(
        "-s",
        "--serial",
        default=None,
        help="指定设备序列号/UDID（默认使用第一个可用设备）",
    )
    parser.add_argument(
        "-p",
        "--platform",
        choices=["android", "ios"],
        default="android",
        help="设备平台",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=30,
        help="最大执行步数",
    )
    parser.add_argument(
        "--context-file",
        "-cf",
        default=None,
        help="任务上下文文件路径，提供任务相关的背景信息以提高执行成功率",
    )
    parser.add_argument(
        "--context",
        "-c",
        default=None,
        help="直接传入任务上下文文本",
    )
    args = parser.parse_args()

    if not check_all_models_available():
        return

    # 读取任务上下文
    context = None
    if args.context:
        context = args.context.strip()
    elif args.context_file:
        from pathlib import Path

        kpath = Path(args.context_file)
        if not kpath.exists():
            print(f"❌ 任务上下文文件不存在: {kpath}")
            return
        context = kpath.read_text(encoding="utf-8").strip()
        if not context:
            print("⚠️  任务上下文文件为空，已忽略")
            context = None

    if args.mode == "manual":
        demo_manual_control(platform=args.platform, serial=args.serial)
    elif args.mode == "ai":
        demo_ai_assisted_task(
            args.task,
            platform=args.platform,
            serial=args.serial,
            max_steps=args.max_steps,
            context=context,
        )
    else:
        demo_find_and_click(
            target=args.task, platform=args.platform, serial=args.serial
        )


if __name__ == "__main__":
    main()
