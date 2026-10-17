from agent.langchain_agent import (
    create_langchain_tools,
)
from agent.tool_registry import ToolRegistry


def test_create_five_langchain_tools():
    registry = ToolRegistry()

    tools = create_langchain_tools(
        registry
    )

    tool_names = {
        tool.name
        for tool in tools
    }

    assert len(tools) == 5

    assert tool_names == {
        "get_model_metrics",
        "compare_models",
        "search_research_docs",
        "get_experiment_config",
        "generate_metric_plot",
    }


def test_tool_has_description_and_schema():
    registry = ToolRegistry()

    tools = create_langchain_tools(
        registry
    )

    for tool in tools:
        assert tool.name
        assert tool.description
        assert tool.args_schema is not None