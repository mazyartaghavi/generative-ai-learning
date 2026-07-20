from __future__ import annotations

import httpx


OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"


TEXT_SAMPLES = [
    (
        "SHORT ENGLISH",
        "AI helps.",
    ),
    (
        "LONGER ENGLISH",
        (
            "Artificial intelligence helps software systems "
            "recognize patterns and make predictions."
        ),
    ),
    (
        "UNCOMMON WORD",
        (
            "The vehicle behaved uncharacteristically "
            "during inspection."
        ),
    ),
    (
        "PUNCTUATION",
        "API, API? API!",
    ),
    (
        "PERSIAN",
        "هوش مصنوعی به تحلیل داده‌ها کمک می‌کند.",
    ),
]


def get_model_token_count(text: str) -> int:
    """Ask Ollama how many input tokens the model processed."""

    request_body = {
        "model": MODEL_NAME,
        "prompt": text,
        "raw": True,
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": 1,
        },
    }

    try:
        response = httpx.post(
            OLLAMA_GENERATE_URL,
            json=request_body,
            timeout=120.0,
        )
        response.raise_for_status()
    except httpx.ConnectError as error:
        raise RuntimeError(
            "Could not connect to Ollama. "
            "Confirm that the Ollama application is running."
        ) from error
    except httpx.TimeoutException as error:
        raise RuntimeError(
            "Ollama did not respond before the timeout."
        ) from error
    except httpx.HTTPStatusError as error:
        raise RuntimeError(
            "Ollama returned HTTP status "
            f"{error.response.status_code}."
        ) from error

    response_body = response.json()
    prompt_token_count = response_body.get(
        "prompt_eval_count"
    )

    if not isinstance(prompt_token_count, int):
        raise RuntimeError(
            "Ollama did not return a valid "
            "prompt token count."
        )

    return prompt_token_count


def count_whitespace_words(text: str) -> int:
    """Count pieces separated by ordinary whitespace."""

    return len(text.split())


def print_sample_report(
    label: str,
    text: str,
) -> None:
    """Print character, word, and model-token measurements."""

    character_count = len(text)
    whitespace_word_count = count_whitespace_words(text)
    model_token_count = get_model_token_count(text)

    print(label)
    print("-" * len(label))
    print("Text:", text)
    print("Character count:", character_count)
    print(
        "Whitespace-separated word count:",
        whitespace_word_count,
    )
    print("Model-reported token count:", model_token_count)
    print()


def main() -> None:
    """Compare words, characters, and model token counts."""

    print("TOKENIZATION COMPARISON")
    print("=======================")
    print()

    for label, text in TEXT_SAMPLES:
        print_sample_report(
            label=label,
            text=text,
        )

    print("IMPORTANT")
    print("---------")
    print(
        "A token is not necessarily one word or "
        "one character."
    )
    print(
        "Token counts depend on the tokenizer used "
        "by the selected model."
    )


if __name__ == "__main__":
    main()