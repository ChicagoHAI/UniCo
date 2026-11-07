"""Shared prompt utilities for data preprocessing"""

from typing import List


def format_prompt_chat_template(tokenizer, inst: str) -> str:
    """Apply the tokenizer's chat template with a single user message.

    Raises ValueError if the tokenizer does not have a chat_template.
    """
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(
            f"Tokenizer '{tokenizer.name_or_path}' does not have a chat_template. "
            "A chat_template is required for prompt formatting."
        )

    messages = [
        {"role": "user", "content": inst},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    return prompt


def find_all_boxed(text: str) -> List[str]:
    """Return inner contents of every \\boxed{...} occurrence using brace depth tracking."""
    results: List[str] = []
    search_start = 0
    needle = "\\boxed{"

    while True:
        start = text.find(needle, search_start)
        if start == -1:
            break
        idx = start + len(needle)
        depth = 1

        while idx < len(text) and depth > 0:
            char = text[idx]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            idx += 1

        if depth == 0:
            inner = text[start + len(needle) : idx - 1].strip()
            results.append(inner)
            search_start = idx
        else:
            # Unbalanced braces; stop scanning to avoid infinite loop.
            break

    return results
