from __future__ import annotations

import httpx


OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3.2:3b"


def ask_ollama(prompt: str) -> str:
    """Send a prompt to Ollama and return the model's answer."""

    request_body = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a careful Generative AI tutor. "
                    "Explain technical terms using simple language."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "stream": False,
    }

    try:
        response = httpx.post(
            OLLAMA_CHAT_URL,
            json=request_body,
            timeout=120.0,
        )
        response.raise_for_status()
    except httpx.ConnectError as error:
        raise RuntimeError(
            "Could not connect to Ollama. Confirm that Ollama is running."
        ) from error
    except httpx.TimeoutException as error:
        raise RuntimeError(
            "Ollama did not respond before the 120-second timeout."
        ) from error
    except httpx.HTTPStatusError as error:
        raise RuntimeError(
            f"Ollama returned an HTTP error: {error.response.status_code}"
        ) from error

    response_body = response.json()

    message = response_body.get("message")
    if not isinstance(message, dict):
        raise RuntimeError("Ollama returned an unexpected message format.")

    content = message.get("content")
    if not isinstance(content, str):
        raise RuntimeError("Ollama did not return a text answer.")

    return content.strip()


def main() -> None:
    question = "What is an API? Explain it using a restaurant analogy."

    print("Question:")
    print(question)
    print()

    answer = ask_ollama(question)

    print("Llama's answer:")
    print(answer)


if __name__ == "__main__":
    main()