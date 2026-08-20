"""A stand-in for a Qwen3.5-style chat tokenizer.

Deliberately reproduces the two behaviours that broke the first implementation of
`labkit.data.build_example` on the real model:

1. **The think scaffold.** `add_generation_prompt=True` ends the render with
   `<think>\\n`, while the full render continues `\\n</think>\\n\\n` — even when the
   answer contains no reasoning at all.
2. **Token merging across that boundary.** Consecutive newlines merge into ONE token.
   So the prefix's trailing `\\n` and the full render's `\\n\\n` are different tokens,
   and the two token lists are *not* prefix-related even though the strings are.

Any masking implementation that diffs token lists fails here — which is the point.
Offsets are provided because the real implementation maps characters to tokens.
"""
from __future__ import annotations

import re

IM_START, IM_END = "<|im_start|>", "<|im_end|>"
SPECIALS = (IM_START, IM_END, "<think>", "</think>")


class FakeTokenizer:
    def __init__(self, strip_thinking: bool = False, prefix_unstable: bool = False,
                 think_scaffold: bool = True, fast: bool = True):
        self.strip_thinking = strip_thinking
        self.prefix_unstable = prefix_unstable
        self.think_scaffold = think_scaffold
        self.fast = fast
        self.eos_token = IM_END
        self.pad_token = None

    # --- template ---------------------------------------------------------
    def _render(self, messages, add_generation_prompt=False, enable_thinking=None):
        parts = []
        for n, m in enumerate(messages):
            content = m["content"]
            if self.strip_thinking or enable_thinking is False:
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.S)
            if self.prefix_unstable and m["role"] == "user" and n == 0 and len(messages) > 1:
                content = content.upper()      # pathological: rewrites turn 0
            if m["role"] == "assistant" and self.think_scaffold:
                # Qwen3.5 NORMALIZES the reasoning block: it always renders
                # `<think>\n{body}\n</think>\n\n{answer}`, emitting an empty block even
                # when the answer carries no reasoning. Model that faithfully, or the
                # fake disagrees with the real template about where the prefix ends.
                mt = re.search(r"<think>(.*?)</think>", content, flags=re.S)
                body = mt.group(1).strip() if mt else ""
                answer = re.sub(r"<think>.*?</think>", "", content, flags=re.S).lstrip()
                content = f"<think>\n{body}\n</think>\n\n{answer}"
            parts.append(f"{IM_START}{m['role']}\n{content}{IM_END}\n")
        if add_generation_prompt:
            tail = f"{IM_START}assistant\n"
            if self.think_scaffold:
                tail += "<think>\n"           # <- the boundary that breaks token diffing
            parts.append(tail)
        return "".join(parts)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False,
                            enable_thinking=None, **kw):
        text = self._render(messages, add_generation_prompt, enable_thinking)
        return self._tokenize(text)[0] if tokenize else text

    # --- codec ------------------------------------------------------------
    def _tokenize(self, text: str):
        """Char-level, except specials are one token and runs of \\n merge into one."""
        ids, offsets, i = [], [], 0
        while i < len(text):
            for sp in SPECIALS:
                if text.startswith(sp, i):
                    ids.append(-(SPECIALS.index(sp) + 1))
                    offsets.append((i, i + len(sp)))
                    i += len(sp)
                    break
            else:
                if text[i] == "\n":
                    j = i
                    while j < len(text) and text[j] == "\n":
                        j += 1
                    ids.append(1000 + (j - i))     # \n vs \n\n -> DIFFERENT tokens
                    offsets.append((i, j))
                    i = j
                else:
                    ids.append(ord(text[i]))
                    offsets.append((i, i + 1))
                    i += 1
        return ids, offsets

    def __call__(self, text, add_special_tokens=True, return_offsets_mapping=False, **kw):
        ids, offsets = self._tokenize(text)
        out = {"input_ids": ids}
        if return_offsets_mapping:
            if not self.fast:
                return {"input_ids": ids}          # simulate a slow tokenizer
            out["offset_mapping"] = offsets
        return out

    def encode(self, text: str, add_special_tokens: bool = True):
        return self._tokenize(text)[0]

    def decode(self, ids, skip_special_tokens: bool = False):
        chunks = []
        for t in ids:
            if t < 0:
                chunks.append("" if skip_special_tokens else SPECIALS[-t - 1])
            elif t >= 1000:
                chunks.append("\n" * (t - 1000))
            else:
                chunks.append(chr(t))
        return "".join(chunks)
