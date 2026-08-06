"""Shared LLM call utilities — single place for MultiModalConversation logic."""

from http import HTTPStatus
import dashscope


def llm_chat(model: str, system: str, user: str,
             temperature: float = 0.1, max_tokens: int = 4096,
             label: str = 'LLM') -> tuple[str, dict]:
    """Call dashscope MultiModalConversation, returns (content, usage)."""
    messages = [
        {'role': 'system', 'content': [{'text': system}]},
        {'role': 'user', 'content': [{'text': user}]},
    ]
    try:
        resp = dashscope.MultiModalConversation.call(
            model=model, messages=messages,
            temperature=temperature, max_tokens=max_tokens,
        )
        if resp.status_code != HTTPStatus.OK:
            print(f"    [WARN] {label} status={resp.status_code} code={resp.code} msg={resp.message}")
            return '', {'input': 0, 'output': 0}
        content = resp.output.choices[0].message.content[0]['text']
        usage = {'input': resp.usage.input_tokens, 'output': resp.usage.output_tokens}
        return content, usage
    except Exception as e:
        print(f"    [ERROR] {label}: {e}")
        return '', {'input': 0, 'output': 0}
