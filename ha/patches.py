from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

from langchain_core.messages import (AIMessage, BaseMessage, ChatMessage,
                                     FunctionMessage, HumanMessage,
                                     SystemMessage, ToolMessage)
from langchain_core.output_parsers.openai_tools import (make_invalid_tool_call,
                                                        parse_tool_call)

def patched_convert_dict_to_message(_dict: Mapping[str, Any]) -> BaseMessage:
    """Convert a dictionary to a LangChain message.

    Args:
        _dict: The dictionary.

    Returns:
        The LangChain message.
    """
    role = _dict.get("role")
    name = _dict.get("name")
    id_ = _dict.get("id")
    if role == "user":
        return HumanMessage(content=_dict.get("content", ""), id=id_, name=name)
    if role == "assistant":
        # Fix for azure
        # Also OpenAI returns None for tool invocations
        content = _dict.get("content", "") or ""
        additional_kwargs: dict = {}
        if function_call := _dict.get("function_call"):
            additional_kwargs["function_call"] = dict(function_call)
        tool_calls = []
        invalid_tool_calls = []
        if raw_tool_calls := _dict.get("tool_calls"):
            for raw_tool_call in raw_tool_calls:
                try:
                    tool_calls.append(parse_tool_call(raw_tool_call, return_id=True))
                except Exception as e:
                    invalid_tool_calls.append(
                        make_invalid_tool_call(raw_tool_call, str(e))
                    )
        if audio := _dict.get("audio"):
            additional_kwargs["audio"] = audio

        # fix reasoning_content miss
        if "reasoning_content" in _dict:
            reasoning_content = _dict.get("reasoning_content", "") or ""
            return AIMessage(
                content=content,
                reasoning_content=reasoning_content,
                additional_kwargs=additional_kwargs,
                name=name,
                id=id_,
                tool_calls=tool_calls,
                invalid_tool_calls=invalid_tool_calls,
            )

        if "reasoning" in _dict:
            reasoning = _dict.get("reasoning", "") or ""
            return AIMessage(
                content=content,
                reasoning=reasoning,
                additional_kwargs=additional_kwargs,
                name=name,
                id=id_,
                tool_calls=tool_calls,
                invalid_tool_calls=invalid_tool_calls,
            )

        return AIMessage(
            content=content,
            additional_kwargs=additional_kwargs,
            name=name,
            id=id_,
            tool_calls=tool_calls,
            invalid_tool_calls=invalid_tool_calls,
        )
    if role in ("system", "developer"):
        additional_kwargs = {"__openai_role__": role} if role == "developer" else {}
        return SystemMessage(
            content=_dict.get("content", ""),
            name=name,
            id=id_,
            additional_kwargs=additional_kwargs,
        )
    if role == "function":
        return FunctionMessage(
            content=_dict.get("content", ""), name=cast(str, _dict.get("name")), id=id_
        )
    if role == "tool":
        additional_kwargs = {}
        if "name" in _dict:
            additional_kwargs["name"] = _dict["name"]
        return ToolMessage(
            content=_dict.get("content", ""),
            tool_call_id=cast(str, _dict.get("tool_call_id")),
            additional_kwargs=additional_kwargs,
            name=name,
            id=id_,
        )
    return ChatMessage(content=_dict.get("content", ""), role=role, id=id_)  # type: ignore[arg-type]


#import langchain_openai.chat_models.base as lc_base
#lc_base._convert_dict_to_message = patched_convert_dict_to_message

