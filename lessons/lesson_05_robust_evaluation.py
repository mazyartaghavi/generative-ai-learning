from __future__ import annotations

from lesson_02_ollama_api import ask_ollama


PROMPT = """
Explain what an API is using a restaurant analogy.

Use these six exact mappings:
- customer = client application
- waiter = API
- menu = API documentation
- order = request
- kitchen = backend server
- prepared dish = response

Requirements:
1. Produce exactly six bullet points.
2. Start every bullet point with a hyphen and a space: "- ".
3. Explain exactly one mapping in each bullet point.
4. Do not combine multiple mappings in one bullet point.
5. After the bullets, write one concise definition beginning with "API:".
6. Keep the complete answer under 180 words.
7. Do not identify the restaurant itself as the API.
""".strip()


BULLET_PREFIXES = ("- ", "* ", "• ")

EXPECTED_MAPPINGS = [
    ("customer", "client application"),
    ("waiter", "api"),
    ("menu", "api documentation"),
    ("order", "request"),
    ("kitchen", "backend server"),
    ("prepared dish", "response"),
]


def count_words(text: str) -> int:
    """Count whitespace-separated words in text."""

    return len(text.split())


def extract_bullet_points(text: str) -> list[str]:
    """Extract bullet-point text using several common bullet symbols."""

    bullet_points: list[str] = []

    for line in text.splitlines():
        cleaned_line = line.strip()

        for prefix in BULLET_PREFIXES:
            if cleaned_line.startswith(prefix):
                bullet_text = cleaned_line.removeprefix(prefix).strip()
                bullet_points.append(bullet_text)
                break

    return bullet_points


def contains_mapping(
    bullet: str,
    first_term: str,
    second_term: str,
) -> bool:
    """Check whether both mapping terms occur in one bullet point."""

    lowercase_bullet = bullet.lower()

    return (
        first_term.lower() in lowercase_bullet
        and second_term.lower() in lowercase_bullet
    )


def count_expected_mappings(bullet: str) -> int:
    """Count how many expected mappings occur in one bullet point."""

    mapping_count = 0

    for first_term, second_term in EXPECTED_MAPPINGS:
        if contains_mapping(bullet, first_term, second_term):
            mapping_count = mapping_count + 1

    return mapping_count


def all_mappings_are_present(bullet_points: list[str]) -> bool:
    """Check that every expected mapping appears in at least one bullet."""

    for first_term, second_term in EXPECTED_MAPPINGS:
        mapping_found = False

        for bullet in bullet_points:
            if contains_mapping(bullet, first_term, second_term):
                mapping_found = True
                break

        if not mapping_found:
            return False

    return True


def one_mapping_per_bullet(bullet_points: list[str]) -> bool:
    """Check that every bullet contains exactly one expected mapping."""

    if len(bullet_points) != 6:
        return False

    for bullet in bullet_points:
        if count_expected_mappings(bullet) != 1:
            return False

    return True


def ends_with_api_definition(text: str) -> bool:
    """Check whether the final non-empty line begins with 'API:'."""

    lines = text.strip().splitlines()

    if not lines:
        return False

    final_line = lines[-1].strip()
    return final_line.lower().startswith("api:")


def evaluate_answer(answer: str) -> dict[str, bool]:
    """Evaluate the measurable answer requirements."""

    lowercase_answer = answer.lower()
    bullet_points = extract_bullet_points(answer)

    return {
        "Exactly six bullet points": len(bullet_points) == 6,
        "Fewer than 180 words": count_words(answer) < 180,
        "All six mappings are present": all_mappings_are_present(
            bullet_points
        ),
        "Exactly one mapping appears in each bullet": (
            one_mapping_per_bullet(bullet_points)
        ),
        "Ends with an API definition": ends_with_api_definition(answer),
        "Does not say 'restaurant = API'": (
            "restaurant = api" not in lowercase_answer
        ),
        "Does not say 'restaurant is the API'": (
            "restaurant is the api" not in lowercase_answer
        ),
    }


def print_mapping_report(bullet_points: list[str]) -> None:
    """Print whether every expected mapping was detected."""

    print("MAPPING REPORT")
    print("--------------")

    for first_term, second_term in EXPECTED_MAPPINGS:
        matching_bullet_found = False

        for bullet in bullet_points:
            if contains_mapping(bullet, first_term, second_term):
                matching_bullet_found = True
                break

        status = "PASS" if matching_bullet_found else "FAIL"

        print(
            f"{status}: {first_term} = {second_term}"
        )


def main() -> None:
    print("Sending the improved prompt to Llama...")
    print()

    answer = ask_ollama(PROMPT)
    bullet_points = extract_bullet_points(answer)
    checks = evaluate_answer(answer)

    print("GENERATED ANSWER")
    print("----------------")
    print(answer)
    print()

    print("EXTRACTED BULLET POINTS")
    print("-----------------------")

    for number, bullet in enumerate(bullet_points, start=1):
        mapping_count = count_expected_mappings(bullet)

        print(
            f"{number}. {bullet}"
        )
        print(
            f"   Expected mappings detected: {mapping_count}"
        )

    print()
    print("AUTOMATIC MEASUREMENTS")
    print("----------------------")
    print("Word count:", count_words(answer))
    print("Bullet-point count:", len(bullet_points))
    print()

    print_mapping_report(bullet_points)
    print()

    print("FINAL EVALUATION")
    print("----------------")

    for requirement, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"{status}: {requirement}")


if __name__ == "__main__":
    main()