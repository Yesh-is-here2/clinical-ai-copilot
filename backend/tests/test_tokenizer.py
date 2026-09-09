from app.retrieval.tokenizer import (
    WhitespaceTokenizer,
    get_default_tokenizer,
)


def test_whitespace_tokenizer_round_trips_word_count() -> None:
    tokenizer = WhitespaceTokenizer()
    text = "Hypertension is defined as sustained elevated blood pressure."

    tokens = tokenizer.encode(text)

    assert len(tokens) == len(text.split())
    assert tokenizer.count(text) == len(tokens)


def test_whitespace_tokenizer_decode_reconstructs_content() -> None:
    tokenizer = WhitespaceTokenizer()
    text = "monitor blood pressure regularly"

    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)

    assert decoded == text


def test_whitespace_tokenizer_is_deterministic() -> None:
    tokenizer = WhitespaceTokenizer()
    text = "the quick brown fox jumps over the lazy dog"

    assert tokenizer.encode(text) == tokenizer.encode(text)


def test_default_tokenizer_reports_an_explicit_name() -> None:
    tokenizer = get_default_tokenizer()

    # Whichever implementation wins (tiktoken vs. the offline fallback),
    # callers must be able to tell which one they got - never silently
    # get a fallback that claims to be the real tokenizer.
    assert tokenizer.name in {
        "tiktoken:cl100k_base",
        "whitespace-fallback",
    }
