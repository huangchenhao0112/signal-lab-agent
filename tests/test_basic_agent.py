from dataclasses import dataclass

from agent.basic_agent import BasicAgent


@dataclass
class FakeLLMResult:
    """
    模拟项目中的LLMResult。
    """

    ok: bool
    content: str | None = None
    error_type: str | None = None
    error_message: str | None = None


class FakeLLMClient:
    """
    模拟LLMClient。

    每调用一次chat，就按顺序返回一个预设结果。
    """

    def __init__(
        self,
        responses: list[FakeLLMResult],
    ) -> None:
        self.responses = responses
        self.call_count = 0
        self.received_messages: list[
            list[dict[str, str]]
        ] = []

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.2,
    ) -> FakeLLMResult:
        self.received_messages.append(
            messages
        )

        response = self.responses[
            self.call_count
        ]

        self.call_count += 1

        return response


def test_agent_calls_tool_then_finishes():
    """
    测试完整流程：

    第一次LLM选择工具
    -> 工具返回Observation
    -> 第二次LLM生成最终答案
    """

    first_response = FakeLLMResult(
        ok=True,
        content="""
        {
          "thought": "需要查询相关资料",
          "action": "tool",
          "tool_name": "lookup_demo",
          "tool_input": {
            "topic": "多窗口FFT"
          }
        }
        """,
    )

    second_response = FakeLLMResult(
        ok=True,
        content="""
        {
          "thought": "已经获得相关资料",
          "action": "finish",
          "answer": "多窗口FFT可以提供不同分辨率的频谱参考。"
        }
        """,
    )

    fake_llm_client = FakeLLMClient(
        responses=[
            first_response,
            second_response,
        ]
    )

    def lookup_demo(
        topic: str,
    ) -> dict:
        """
        根据主题查询测试资料。
        """

        return {
            "topic": topic,
            "content": (
                "多窗口FFT融合多个窗口的"
                "频谱参考信息。"
            ),
        }

    agent = BasicAgent(
        llm_client=fake_llm_client,
        tools={
            "lookup_demo": lookup_demo,
        },
        max_steps=5,
    )

    result = agent.run(
        "多窗口FFT有什么作用？"
    )

    assert result.ok is True

    assert result.answer == (
        "多窗口FFT可以提供不同分辨率"
        "的频谱参考。"
    )

    assert len(result.steps) == 2

    assert (
        result.steps[0].action
        == "tool"
    )

    assert (
        result.steps[0].tool_name
        == "lookup_demo"
    )

    assert (
        "多窗口FFT融合多个窗口"
        in result.steps[0].observation
    )

    assert (
        result.steps[1].action
        == "finish"
    )

    assert fake_llm_client.call_count == 2

    second_messages = (
        fake_llm_client.received_messages[1]
    )

    assert (
        "多窗口FFT融合多个窗口"
        in second_messages[1]["content"]
    )


def test_agent_stops_at_max_steps():
    """
    测试模型一直调用工具时，
    Agent能否在最大步数后停止。
    """

    repeated_response = FakeLLMResult(
        ok=True,
        content="""
        {
          "thought": "继续查询",
          "action": "tool",
          "tool_name": "lookup_demo",
          "tool_input": {
            "topic": "FFT"
          }
        }
        """,
    )

    fake_llm_client = FakeLLMClient(
        responses=[
            repeated_response,
            repeated_response,
        ]
    )

    def lookup_demo(
        topic: str,
    ) -> str:
        """
        返回测试查询结果。
        """

        return f"查询结果：{topic}"

    agent = BasicAgent(
        llm_client=fake_llm_client,
        tools={
            "lookup_demo": lookup_demo,
        },
        max_steps=2,
    )

    result = agent.run(
        "请查询FFT资料"
    )

    assert result.ok is False

    assert (
        result.error_type
        == "max_steps_exceeded"
    )

    assert len(result.steps) == 2

    assert fake_llm_client.call_count == 2


def test_agent_handles_llm_error():
    """
    测试LLM调用失败时，
    Agent能否停止并返回错误。
    """

    fake_llm_client = FakeLLMClient(
        responses=[
            FakeLLMResult(
                ok=False,
                error_type="timeout",
                error_message="大模型请求超时",
            )
        ]
    )

    agent = BasicAgent(
        llm_client=fake_llm_client,
        tools={},
        max_steps=5,
    )

    result = agent.run(
        "多窗口FFT有什么作用？"
    )

    assert result.ok is False

    assert result.error_type == "timeout"

    assert (
        result.error_message
        == "大模型请求超时"
    )

    assert result.steps == []