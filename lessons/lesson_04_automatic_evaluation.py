from __future__ import annotations

from lesson_02_ollama_api import ask_ollama


PROMPT = """
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
4. Finish with one concise technical definition beginning with "API:".
5. Keep the complete answer under 180 words.
""".strip()


def count_words(text: str) -> int:
    """Count the words in a piece of text."""

    words = text.split()
    return len(words)


def count_bullet_points(text: str) -> int:
    """Count lines that begin with an asterisk or hyphen."""

    lines = text.splitlines()
    bullet_count = 0

    for line in lines:
        cleaned_line = line.strip()

        if cleaned_line.startswith("* ") or cleaned_line.startswith("- "):
            bullet_count = bullet_count + 1

    return bullet_count


def ends_with_api_definition(text: str) -> bool:
    """Check whether the final non-empty line starts with 'API:'."""

    lines = text.strip().splitlines()
    final_line = lines[-1].strip()

    return final_line.lower().startswith("api:")


def evaluate_answer(answer: str) -> dict[str, bool]:
    """Evaluate measurable requirements of the generated answer."""

    lowercase_answer = answer.lower()
    word_count = count_words(answer)
    bullet_count = count_bullet_points(answer)

    checks = {
        "Exactly six bullet points": bullet_count == 6,
        "Fewer than 180 words": word_count < 180,
        "Contains waiter and API": (
            "waiter" in lowercase_answer
            and "api" in lowercase_answer
        ),
        "Contains kitchen and backend server": (
            "kitchen" in lowercase_answer
            and "backend server" in lowercase_answer
        ),
        "Contains prepared dish and response": (
            "prepared dish" in lowercase_answer
            and "response" in lowercase_answer
        ),
        "Ends with an API definition": ends_with_api_definition(answer),
        "Does not say 'restaurant = API'": (
            "restaurant = api" not in lowercase_answer
        ),
        "Does not say 'restaurant is the API'": (
            "restaurant is the api" not in lowercase_answer
        ),
    }

    return checks


def main() -> None:
    print("Sending the prompt to Llama...")
    print()

    answer = ask_ollama(PROMPT)

    print("GENERATED ANSWER")
    print("----------------")
    print(answer)
    print()

    word_count = count_words(answer)
    bullet_count = count_bullet_points(answer)
    checks = evaluate_answer(answer)

    print("AUTOMATIC MEASUREMENTS")
    print("----------------------")
    print("Word count:", word_count)
    print("Bullet-point count:", bullet_count)
    print()

    print("AUTOMATIC EVALUATION")
    print("--------------------")

    for requirement, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"{status}: {requirement}")


if __name__ == "__main__":
    main()