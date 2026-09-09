"""Token-counting abstraction used by the deterministic chunker.

The chunking algorithm needs to slice text into fixed-size, overlapping
windows measured in *tokens* rather than characters or words, because
downstream embedding/LLM calls are priced and limited in tokens.

Two implementations are provided behind a common ``Tokenizer`` protocol:

* ``Tiktoken Tokenizer`` - wraps OpenAI's ``tiktoken`` library using the
  ``cl100k_base`` encoding (the encoding used by the embedding/chat models
  this project targets). This gives token counts that match what the real
  OpenAI API will bill/limit on.
* ``WhitespaceTokenizer`` - a dependency-free, network-free fallback that
  approximates token counts by splitting on whitespace. It is used
  automatically when ``tiktoken`` cannot load its encoding file (for
  example, in a sandboxed/offline environment with no egress to
  ``openaipublic.blob.core.windows.net``), and always in tests, so the
  chunking test suite never depends on network access.

The fallback is explicit and observable: ``get_default_tokenizer()``
returns a tokenizer whose ``.name`` is either ``"tiktoken:cl100k_base"`` or
``"whitespace-fallback"``. Callers (and the ingestion CLI) should surface
this name rather than silently assuming exact GPT token counts - counts
produced by the fallback are an approximation, never a substitute for the
real tokenizer's counts.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Protocol, Sequence


class Tokenizer(Protocol):
    """Anything that can deterministically split text into token units
    and reassemble a slice of those units back into text."""

    name: str

    def encode(self, text: str) -> Sequence[object]:
        ...

    def decode(self, tokens: Sequence[object]) -> str:
        ...

    def count(self, text: str) -> int:
        ...


class TiktokenTokenizer:
    """Real GPT-style tokenizer backed by ``tiktoken``."""

    name = "tiktoken:cl100k_base"

    def __init__(self) -> None:
        import tiktoken

        # This is the line that actually touches the network the first
        # time it runs in a given environment (to fetch the encoding's
        # merge-rank file). If that fails, the caller falls back to the
        # whitespace tokenizer instead of pretending this succeeded.
        self._encoding = tiktoken.get_encoding("cl100k_base")

    def encode(self, text: str) -> list[int]:
        return self._encoding.encode(text)

    def decode(self, tokens: Sequence[int]) -> str:
        return self._encoding.decode(list(tokens))

    def count(self, text: str) -> int:
        return len(self.encode(text))


_WHITESPACE_TOKEN_PATTERN = re.compile(r"\S+")


class WhitespaceTokenizer:
    """Deterministic, dependency-free fallback tokenizer.

    Tokens are whitespace-delimited "words" (including attached
    punctuation). This is a coarse approximation of true subword
    tokenization - typically within the same order of magnitude as GPT
    token counts for English clinical prose - but it must never be
    reported to a caller as an exact tiktoken count.
    """

    name = "whitespace-fallback"

    def encode(self, text: str) -> list[str]:
        return _WHITESPACE_TOKEN_PATTERN.findall(text)

    def decode(self, tokens: Sequence[str]) -> str:
        return " ".join(tokens)

    def count(self, text: str) -> int:
        return len(self.encode(text))


@lru_cache(maxsize=1)
def get_default_tokenizer() -> Tokenizer:
    """Return the best available tokenizer.

    Prefers the real ``tiktoken`` encoding; falls back to the whitespace
    tokenizer (logging nothing sensitive - this never touches document
    content) if ``tiktoken`` cannot be constructed, e.g. because the
    encoding file cannot be downloaded in this environment.
    """

    try:
        return TiktokenTokenizer()
    except Exception:
        return WhitespaceTokenizer()
