"""Append-only chat transcripts and tool-call parsing.

The transcript is kept as token ids, never re-rendered from messages, so turn k's prompt is an
exact prefix of turn k+1's. That keeps token positions valid for teacher-forced replay (the
CAPE design rule "append-only contexts"). Chat-template pieces come from the model's own
template: the initial prompt is rendered once, and the per-turn tool-response wrapper is
derived by rendering a probe conversation and cutting at sentinel strings.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

_A, _R = "@@ASSISTANT_SENTINEL@@", "@@RESULT_SENTINEL@@"


@dataclass
class Wrapper:
    """Text that surrounds a tool result between two assistant generations."""
    end_of_turn: str     # what closes an assistant turn (e.g. "<|im_end|>")
    before_result: str   # from end of assistant turn to the start of the tool result
    after_result: str    # from end of tool result through the next generation prompt
    think_open: str      # the generation prompt's opening think tag, if any
    think_close: str


def tool_wrapper(tokenizer, tools) -> Wrapper:
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
            {"role": "assistant", "content": _A}, {"role": "tool", "content": _R}]
    text = tokenizer.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=True)
    tail = text.split(_A, 1)[1]
    before, after = tail.split(_R, 1)
    eot = "<|im_end|>" if "<|im_end|>" in before else (tokenizer.eos_token or "")
    think_open = "<think>\n" if after.endswith("<think>\n") else ""
    return Wrapper(end_of_turn=eot, before_result=before, after_result=after,
                   think_open=think_open, think_close="</think>")


def initial_prompt(tokenizer, system: str, user: str, tools) -> str:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return tokenizer.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=True)


def openai_tools(schemas):
    return [{"type": "function", "function": s} for s in schemas]


# ---------------------------------------------------------------- parsing
_XML_CALL = re.compile(r"<tool_call>\s*<function=([\w\-\.]+)>(.*?)</function>\s*(?:</tool_call>|$)", re.DOTALL)
_XML_PARAM = re.compile(r"<parameter=([\w\-\.]+)>\n?(.*?)\n?</parameter>", re.DOTALL)
_JSON_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*(?:</tool_call>|$)", re.DOTALL)


@dataclass
class Parsed:
    reasoning: str
    content: str        # visible text before the tool call (or the final answer)
    tool: str | None
    args: dict | None
    parse_error: str | None


def split_reasoning(gen_text: str, wrapper: Wrapper) -> tuple[str, str]:
    """The generation prompt already opened <think>; everything before </think> is reasoning."""
    if wrapper.think_close in gen_text:
        r, rest = gen_text.split(wrapper.think_close, 1)
        return r.replace("<think>", "").strip(), rest
    if wrapper.think_open:
        return gen_text.strip(), ""          # never closed its reasoning
    return "", gen_text


def parse_generation(gen_text: str, wrapper: Wrapper) -> Parsed:
    reasoning, rest = split_reasoning(gen_text, wrapper)
    rest = rest.replace(wrapper.end_of_turn, "")
    if "<tool_call>" not in rest:
        return Parsed(reasoning, rest.strip(), None, None, None)
    content = rest.split("<tool_call>", 1)[0].strip()
    m = _XML_CALL.search(rest)
    if m:
        args = {k: v for k, v in _XML_PARAM.findall(m.group(2))}
        for k, v in list(args.items()):
            if v.strip()[:1] in "[{":
                try:
                    args[k] = json.loads(v)
                except json.JSONDecodeError:
                    pass
        return Parsed(reasoning, content, m.group(1), args, None)
    m = _JSON_CALL.search(rest)
    if m:
        try:
            obj = json.loads(m.group(1))
            args = obj.get("arguments", obj.get("parameters", {}))
            if isinstance(args, str):
                args = json.loads(args)
            return Parsed(reasoning, content, obj.get("name"), args, None)
        except (json.JSONDecodeError, AttributeError) as e:
            return Parsed(reasoning, content, None, None, f"bad json tool call: {e}")
    return Parsed(reasoning, content, None, None, "unparseable <tool_call> block")
