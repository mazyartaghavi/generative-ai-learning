from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class ApiMapping(BaseModel):
    """Represent one restaurant-to-software mapping."""

    model_config = ConfigDict(extra="forbid")

    restaurant_term: str = Field(min_length=1)
    software_term: str = Field(min_length=1)
    explanation: str = Field(min_length=10)


VALID_DATA: dict[str, object] = {
    "restaurant_term": "waiter",
    "software_term": "API",
    "explanation": (
        "The waiter carries requests from the customer to the kitchen."
    ),
}

INVALID_DATA: dict[str, object] = {
    "restaurant_term": "waiter",
    "software_term": 123,
    "explanation": "",
    "unexpected_field": True,
}


def validate_and_print(
    title: str,
    input_data: dict[str, object],
) -> None:
    """Validate input data and print either the result or its errors."""

    print(title)
    print("-" * len(title))

    try:
        mapping = ApiMapping.model_validate(input_data)
    except ValidationError as error:
        print("FAIL: The input data is invalid.")
        print()

        for issue in error.errors():
            location = " -> ".join(
                str(part) for part in issue["loc"]
            )
            message = issue["msg"]
            error_type = issue["type"]

            print(f"Field: {location}")
            print(f"Problem: {message}")
            print(f"Error type: {error_type}")
            print()
    else:
        print("PASS: The input data is valid.")
        print()

        print("PYDANTIC OBJECT")
        print(mapping)
        print()

        print("PYTHON DICTIONARY")
        print(mapping.model_dump())
        print()

        print("JSON")
        print(mapping.model_dump_json(indent=2))
        print()


def main() -> None:
    validate_and_print(
        title="VALID EXAMPLE",
        input_data=VALID_DATA,
    )

    print()

    validate_and_print(
        title="INVALID EXAMPLE",
        input_data=INVALID_DATA,
    )


if __name__ == "__main__":
    main()