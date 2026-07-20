from __future__ import annotations

from lesson_02_ollama_api import ask_ollama


BASIC_PROMPT = (
    "Explain what an API is using a restaurant analogy."
)

IMPROVED_PROMPT = """
Explain what an API is using a restaurant analogy.

Use this exact mapping:
- customer = client application
- waiter = API
- menu = API documentation
- order = request
- kitchen = backend server
- prepared dish = response

Requirements:
1. Use exactly six bullet points.
2. Explain one mapping in each bullet point.
3. Do not say that the restaurant itself is the API.
4. Finish with one concise technical definition of an API.
5. Keep the complete answer under 180 words.
""".strip()


def main() -> None:
    print("BASIC PROMPT")
    print("------------")
    print(BASIC_PROMPT)
    print()

    basic_answer = ask_ollama(BASIC_PROMPT)

    print("BASIC ANSWER")
    print("------------")
    print(basic_answer)
    print()
    print()

    print("IMPROVED PROMPT")
    print("---------------")
    print(IMPROVED_PROMPT)
    print()

    improved_answer = ask_ollama(IMPROVED_PROMPT)

    print("IMPROVED ANSWER")
    print("---------------")
    print(improved_answer)
    print()
    print()

    print("MANUAL EVALUATION CHECKLIST")
    print("---------------------------")
    print("1. Does the answer contain exactly six bullet points?")
    print("2. Is the waiter correctly identified as the API?")
    print("3. Is the kitchen correctly identified as the backend server?")
    print("4. Does the answer avoid identifying the restaurant as the API?")
    print("5. Does it end with a concise technical definition?")
    print("6. Is the complete answer under 180 words?")


if __name__ == "__main__":
    main()