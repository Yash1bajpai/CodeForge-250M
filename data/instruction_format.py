"""Normalize public instruction records without executing their code or tools."""
import ast
import json
import re

EOS = "<|endoftext|>"
CONTROL = (EOS, "<|tool_call|>", "<|tool_result|>", "<|fim_prefix|>",
           "<|fim_middle|>", "<|fim_suffix|>", "<|pad|>", "<|unk|>",
           "<|thinking|>", "<|json_start|>", "<|json_end|>")


def _value(text):
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        try:
            value = ast.literal_eval(text)
        except (ValueError, SyntaxError, TypeError):
            raise ValueError("Malformed structured tool data") from None
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        raise ValueError("Tool data must be JSON serializable") from None
    return value


def _object(text):
    value = _value(text)
    if not isinstance(value, dict):
        raise ValueError("Tool data must be an object")
    return value


def _call(text):
    value = _object(text.strip())
    name, arguments = value.get("name"), value.get("arguments")
    if isinstance(arguments, str):
        arguments = _object(arguments)
    if not isinstance(name, str) or not name.strip() or not isinstance(arguments, dict):
        raise ValueError("Tool call requires a name and argument object")
    return "<|tool_call|>" + json.dumps(
        {"name": name, "arguments": arguments}, ensure_ascii=False, sort_keys=True)


def _content(role, text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Empty conversation turn")
    if any(token in text for token in CONTROL):
        raise ValueError("Untrusted control token in conversation")
    if re.search(r"(?m)^\[(?:system|user|assistant|tool)\]$", text):
        raise ValueError("Ambiguous role header in conversation")
    if role == "tool":
        body = text.strip()
        if body.startswith("<tool_response>") and body.endswith("</tool_response>"):
            body = body[len("<tool_response>"):-len("</tool_response>")].strip()
        return "<|tool_result|>" + json.dumps(_value(body), ensure_ascii=False, sort_keys=True)
    if role != "assistant":
        return text
    if "<functioncall>" in text:
        prefix, body = text.split("<functioncall>", 1)
        body = body.strip().removesuffix("</functioncall>").rstrip()
        return prefix + _call(body)
    if "<tool_call>" in text:
        parts, offset = [], 0
        for match in re.finditer(r"<tool_call>(.*?)</tool_call>", text, flags=re.S):
            parts.extend((text[offset:match.start()], _call(match.group(1))))
            offset = match.end()
        parts.append(text[offset:])
        normalized = "".join(parts)
        if "<tool_call>" in normalized or "</tool_call>" in normalized:
            raise ValueError("Unclosed tool call")
        return normalized
    return text


def normalize_instruction(row, format_name):
    """Return normalized text and half-open assistant character spans.

    Input/user/tool tokens are context only. Assistant EOS is supervised so
    the model learns to stop after a reply or yield after a tool call.
    """
    if format_name == "glaive":
        chat, system = row.get("chat"), row.get("system", "")
        if not isinstance(chat, str) or not isinstance(system, str):
            raise ValueError("Invalid Glaive record")
        matches = list(re.finditer(
            r"(?:^|\s+|(?<=<\|endoftext\|>))(USER|ASSISTANT|FUNCTION RESPONSE):\s*", chat))
        if not matches or chat[:matches[0].start()].strip():
            raise ValueError("Malformed Glaive turn delimiters")
        turns = [("system", system.removeprefix("SYSTEM:").lstrip())] if system.strip() else []
        roles = {"USER": "user", "ASSISTANT": "assistant", "FUNCTION RESPONSE": "tool"}
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(chat)
            body = chat[match.end():end].strip()
            if match.group(1) == "ASSISTANT":
                body = body.removesuffix(EOS).rstrip()
            turns.append((roles[match.group(1)], body))
    elif format_name in ("opencoder", "hermes"):
        messages = row.get("messages") or row.get("conversations")
        if messages is None and format_name == "opencoder":
            prompt, answer = row.get("instruction"), row.get("output")
            extra = row.get("input", "")
            if not isinstance(prompt, str) or not isinstance(answer, str) or not isinstance(extra, str):
                raise ValueError("Invalid instruction/answer record")
            turns = [("user", prompt + ("\n" + extra if extra else "")), ("assistant", answer)]
        else:
            if not isinstance(messages, list):
                raise ValueError("Missing message list")
            mapping = {"human": "user", "gpt": "assistant", "function": "tool"}
            turns = []
            for message in messages:
                if not isinstance(message, dict):
                    raise ValueError("Invalid message")
                role = message.get("role", message.get("from"))
                turns.append((mapping.get(role, role), message.get("content", message.get("value"))))
    else:
        raise ValueError("Unsupported instruction format")
    pieces, spans, position, previous, pending = [], [], 0, None, 0
    for role, raw in turns:
        if role not in ("system", "user", "assistant", "tool"):
            raise ValueError("Unsupported role")
        if ((role == "system" and previous is not None)
                or (role == "user" and previous not in (None, "system", "assistant"))
                or (role == "assistant" and (previous not in ("user", "tool")
                                             or (previous == "tool" and pending > 0)))
                or (role == "tool" and (previous not in ("assistant", "tool") or pending <= 0))):
            raise ValueError("Invalid conversation order")
        if role == "user" and pending:
            raise ValueError("Unanswered tool call")
        body = _content(role, raw)
        if role == "assistant":
            pending = body.count("<|tool_call|>")
        elif role == "tool":
            pending -= 1
        prefix = f"[{role}]\n"
        part = prefix + body + EOS + "\n"
        if role == "assistant":
            spans.append([position + len(prefix), position + len(prefix) + len(body) + len(EOS)])
        pieces.append(part)
        position += len(part)
        previous = role
    if not spans or previous != "assistant":
        raise ValueError("No completed assistant turn")
    return {"text": "".join(pieces), "assistant_spans": spans}
