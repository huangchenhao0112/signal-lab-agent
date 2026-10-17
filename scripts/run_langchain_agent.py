#提供单问题运行和五工具批量测试入口
import argparse
import json
from typing import Any

from agent.langchain_agent import (
    AgentRunResult,
)
from agent.langchain_runtime import (
    create_real_langchain_agent,
)


TEST_CASES = [
    {
        "id": "tool_01",
        "question": (
            "查询NCSD-Net在Overall分组下的"
            "hit_rate精确数值。"
        ),
        "expected_tool": (
            "get_model_metrics"
        ),
    },
    {
        "id": "tool_02",
        "question": (
            "比较NCSD-Net和TFA-Net在Overall"
            "分组下的hit_rate，并说明谁更高。"
        ),
        "expected_tool": (
            "compare_models"
        ),
    },
    {
        "id": "tool_03",
        "question": (
            "多窗口FFT在NCSD-Net中有什么作用？"
            "请从项目知识库检索后回答。"
        ),
        "expected_tool": (
            "search_research_docs"
        ),
    },
    {
        "id": "tool_04",
        "question": (
            "查询实验协议FROZEN_SUPPORT_V1"
            "的配置和可比较范围。"
        ),
        "expected_tool": (
            "get_experiment_config"
        ),
    },
    {
        "id": "tool_05",
        "question": (
            "生成NCSD-Net和TFA-Net在Overall"
            "分组下hit_rate的对比图。"
        ),
        "expected_tool": (
            "generate_metric_plot"
        ),
    },
]


def _format_json(
    value: Any,
) -> str:
    """格式化字典或JSON字符串。"""

    if isinstance(value, str):
        try:
            parsed = json.loads(value)

            return json.dumps(
                parsed,
                ensure_ascii=False,
                indent=2,
            )

        except json.JSONDecodeError:
            return value

    return json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        default=str,
    )


def print_agent_result(
    question: str,
    result: AgentRunResult,
    expected_tool: str | None = None,
) -> bool:
    """打印完整轨迹，并返回工具选择是否正确。"""

    print()
    print("=" * 80)
    print("用户问题")
    print("=" * 80)
    print(question)

    print()
    print("=" * 80)
    print("工具调用轨迹")
    print("=" * 80)

    called_tools: list[str] = []

    if not result.trace:
        print("本次回答没有调用工具")

    for index, step in enumerate(
        result.trace,
        start=1,
    ):
        step_type = step.get("type")
        tool_name = step.get("tool")

        print()
        print(
            f"步骤{index}：{step_type}"
        )

        if tool_name:
            print(
                f"工具名称：{tool_name}"
            )

        if step_type == "tool_call":
            if tool_name:
                called_tools.append(
                    str(tool_name)
                )

            print("调用参数：")

            print(
                _format_json(
                    step.get(
                        "arguments",
                        {},
                    )
                )
            )

        elif step_type == "tool_result":
            print("工具返回：")

            print(
                _format_json(
                    step.get(
                        "content",
                        "",
                    )
                )
            )

    print()
    print("=" * 80)
    print("最终回答")
    print("=" * 80)
    print(result.answer)

    if expected_tool is None:
        return True

    passed = (
        expected_tool
        in called_tools
    )

    print()
    print("=" * 80)
    print("工具选择检查")
    print("=" * 80)
    print(
        f"预期工具：{expected_tool}"
    )
    print(
        f"实际工具：{called_tools}"
    )
    print(
        "结果："
        + (
            "通过"
            if passed
            else "未通过"
        )
    )

    return passed


def run_interactive() -> None:
    """交互运行，可以连续输入多个问题。"""

    print("正在初始化真实SignalLab Agent……")

    agent = (
        create_real_langchain_agent()
    )

    print("初始化完成。")
    print(
        "输入问题后按回车；"
        "输入exit或quit结束。"
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
            print("运行结束")
            break

        if not question:
            print("问题不能为空")
            continue

        try:
            result = agent.invoke(
                question
            )

            print_agent_result(
                question=question,
                result=result,
            )

        except Exception as error:
            print()
            print("Agent运行失败：")
            print(
                f"{type(error).__name__}: "
                f"{error}"
            )


def run_all_tool_checks() -> bool:
    """运行5类真实工具选择测试。"""

    print("正在初始化真实SignalLab Agent……")

    agent = (
        create_real_langchain_agent()
    )

    print("初始化完成。")
    print(
        "开始执行5类真实工具测试。"
    )

    passed_count = 0

    for case in TEST_CASES:
        print()
        print("#" * 80)
        print(
            f"测试编号：{case['id']}"
        )
        print("#" * 80)

        try:
            result = agent.invoke(
                case["question"]
            )

            passed = print_agent_result(
                question=case["question"],
                result=result,
                expected_tool=(
                    case["expected_tool"]
                ),
            )

            if passed:
                passed_count += 1

        except Exception as error:
            print()
            print("本条测试执行失败：")
            print(
                f"{type(error).__name__}: "
                f"{error}"
            )

    print()
    print("=" * 80)
    print("五工具测试汇总")
    print("=" * 80)
    print(
        f"通过数量："
        f"{passed_count}/{len(TEST_CASES)}"
    )

    all_passed = (
        passed_count
        == len(TEST_CASES)
    )

    print(
        "最终结果："
        + (
            "全部通过"
            if all_passed
            else "存在未通过项目"
        )
    )

    return all_passed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "运行SignalLab LangChain Agent"
        )
    )

    parser.add_argument(
        "--check-all",
        action="store_true",
        help=(
            "自动运行五种真实工具选择测试"
        ),
    )

    args = parser.parse_args()

    if args.check_all:
        all_passed = (
            run_all_tool_checks()
        )

        if not all_passed:
            raise SystemExit(1)

        return

    run_interactive()


if __name__ == "__main__":
    main()