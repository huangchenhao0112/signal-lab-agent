import argparse
import json
from typing import Any

from agent.langgraph_agent import GraphRunResult
from agent.langgraph_agent import SignalLabGraphAgent
from agent.langchain_runtime import (
    create_real_hybrid_search_fn,
)
from agent.tool_factory import (
    create_signal_lab_tool_registry,
)


TEST_CASES = [
    {
        "question": (
            "多窗口FFT在NCSD-Net中有什么作用？"
        ),
        "expected_route": "rag",
        "expected_tool": (
            "search_research_docs"
        ),
    },
    {
        "question": (
            "查询NCSD-Net在Overall分组下的"
            "hit_rate精确数值。"
        ),
        "expected_route": "tool",
        "expected_tool": (
            "get_model_metrics"
        ),
    },
    {
        "question": (
            "比较NCSD-Net和TFA-Net在Overall"
            "分组下的hit_rate。"
        ),
        "expected_route": "tool",
        "expected_tool": (
            "compare_models"
        ),
    },
    {
        "question": (
            "查询FROZEN_SUPPORT_V1实验协议"
            "的配置和可比较范围。"
        ),
        "expected_route": "tool",
        "expected_tool": (
            "get_experiment_config"
        ),
    },
    {
        "question": (
            "生成NCSD-Net和TFA-Net在Overall"
            "分组下hit_rate的对比图。"
        ),
        "expected_route": "tool",
        "expected_tool": (
            "generate_metric_plot"
        ),
    },
]


def create_graph_agent(
) -> SignalLabGraphAgent:
    """创建使用真实数据的LangGraph Agent。"""

    search_fn = (
        create_real_hybrid_search_fn()
    )

    registry = (
        create_signal_lab_tool_registry(
            search_fn=search_fn,
        )
    )

    return SignalLabGraphAgent(
        registry=registry,
    )


def pretty_json(
    value: Any,
) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        default=str,
    )


def print_result(
    result: GraphRunResult,
) -> None:
    print()
    print("=" * 80)
    print("路由结果")
    print("=" * 80)
    print(f"route：{result.route}")
    print(
        f"selected_tool："
        f"{result.selected_tool}"
    )

    print()
    print("=" * 80)
    print("节点运行轨迹")
    print("=" * 80)

    for index, step in enumerate(
        result.trace,
        start=1,
    ):
        print()
        print(
            f"步骤{index}："
            f"{step.get('node')}"
        )
        print(
            pretty_json(step)
        )

    print()
    print("=" * 80)
    print("最终回答")
    print("=" * 80)
    print(result.answer)


def run_interactive() -> None:
    print("正在初始化LangGraph Agent……")

    agent = create_graph_agent()

    print("初始化完成。")
    print(
        "输入问题开始运行；"
        "输入exit退出。"
    )

    while True:
        print()

        question = input(
            "请输入问题："
        ).strip()

        if question.lower() in {
            "exit",
            "quit",
            "退出",
        }:
            break

        if not question:
            print("问题不能为空")
            continue

        try:
            result = agent.invoke(
                question=question,
                session_id="day23-demo",
            )

            print_result(result)

        except Exception as error:
            print(
                f"{type(error).__name__}: "
                f"{error}"
            )


def run_checks() -> bool:
    print("正在初始化LangGraph Agent……")

    agent = create_graph_agent()

    passed_count = 0

    for index, case in enumerate(
        TEST_CASES,
        start=1,
    ):
        print()
        print("#" * 80)
        print(
            f"测试{index}："
            f"{case['question']}"
        )
        print("#" * 80)

        try:
            result = agent.invoke(
                case["question"]
            )

            print_result(result)

            route_ok = (
                result.route
                == case["expected_route"]
            )

            tool_ok = (
                result.selected_tool
                == case["expected_tool"]
            )

            passed = (
                route_ok
                and tool_ok
            )

            print()
            print(
                f"预期route："
                f"{case['expected_route']}"
            )
            print(
                f"实际route："
                f"{result.route}"
            )
            print(
                f"预期tool："
                f"{case['expected_tool']}"
            )
            print(
                f"实际tool："
                f"{result.selected_tool}"
            )
            print(
                "结果："
                + (
                    "通过"
                    if passed
                    else "未通过"
                )
            )

            if passed:
                passed_count += 1

        except Exception as error:
            print(
                f"{type(error).__name__}: "
                f"{error}"
            )

    print()
    print("=" * 80)
    print("测试汇总")
    print("=" * 80)
    print(
        f"通过："
        f"{passed_count}/{len(TEST_CASES)}"
    )

    return (
        passed_count
        == len(TEST_CASES)
    )


def show_graph() -> None:
    agent = create_graph_agent()

    print(
        agent.draw_mermaid()
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--check-all",
        action="store_true",
        help="运行5类路由和工具测试",
    )

    parser.add_argument(
        "--show-graph",
        action="store_true",
        help="输出Mermaid工作流代码",
    )

    parser.add_argument(
        "--session-id",
        type=str,
        default="day24-persistent",
        help="LangGraph会话ID，相同ID共享历史记录",
    )
    
    args = parser.parse_args()

    if args.show_graph:
        show_graph()
        return

    if args.check_all:
        passed = run_checks()

        if not passed:
            raise SystemExit(1)

        return

    run_interactive()


if __name__ == "__main__":
    main()