from __future__ import annotations

import pytest

from tools.document_tools import (
    SearchResearchDocsTool,
)


def test_search_tool_empty_question():
    tool = SearchResearchDocsTool(
        search_fn=lambda **kwargs: []
    )

    with pytest.raises(
        ValueError,
        match="question不能为空",
    ):
        tool(
            question="",
            top_k=2,
        )


def test_search_tool_invalid_top_k():
    tool = SearchResearchDocsTool(
        search_fn=lambda **kwargs: []
    )

    with pytest.raises(
        ValueError,
        match="top_k",
    ):
        tool(
            question="FFT",
            top_k=20,
        )


def test_search_backend_failure():
    def broken_search(
        question: str,
        top_k: int,
    ):
        raise RuntimeError(
            "vector store unavailable"
        )

    tool = SearchResearchDocsTool(
        search_fn=broken_search
    )

    with pytest.raises(
        RuntimeError,
        match="vector store unavailable",
    ):
        tool(
            question="FFT",
            top_k=3,
        )