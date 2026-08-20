"""A character-level stand-in for a Qwen-style chat tokenizer.

Character-level keeps the token boundaries exact, so tests can assert on decoded
text without downloading a real model. The template mimics Qwen3.5's ChatML shape,
including the `enable_thinking` switch and an opt-in *reasoning-stripping* mode so
the deck §16 failure can be reproduced deterministically.
"""
from __future__ import annotations

import re

IM_START, IM_END = "<|im_start|>", "<|im_end|>"


class FakeTokenizer:
    def __init__(self, strip_thinking: bool = False, prefix_unstable: bool = False):
        self.strip_thinking = strip_thinking
        self.prefix_unstable = prefix_unstable
        self.eos_token = IM_END

    # --- template ---------------------------------------------------------
    def _render(self, messages, add_generation_prompt=False, enable_thinking=None):
        parts = []
        for n, m in enumerate(messages):
            content = m["content"]
            if self.strip_thinking or enable_thinking is False:
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.S)
            if self.prefix_unstable and m["role"] == "user" and n == 0 and len(messages) > 1:
                # pathological: rewrites turn 0 once the conversation grows
                content = content.upper()
            parts.append(f"{IM_START}{m['role']}\n{content}{IM_END}\n")
        if add_generation_prompt:
            parts.append(f"{IM_START}assistant\n")
        return "".join(parts)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False,
                            enable_thinking=None, **kw):
        text = self._render(messages, add_generation_prompt, enable_thinking)
        return self.encode(text, add_special_tokens=False) if tokenize else text

    # --- codec ------------------------------------------------------------
    def encode(self, text: str, add_special_tokens: bool = True):
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens: bool = False):
        text = "".join(chr(i) for i in ids)
        if skip_special_tokens:
            text = text.replace(IM_START, "").replace(IM_END, "")
        return text
