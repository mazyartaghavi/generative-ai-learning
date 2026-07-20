from __future__ import annotations

from lessons.lesson_28_guarded_rag import (
    GuardedRAGResult,
)
from lessons.lesson_42_application_settings import (
    get_settings,
)
from lessons.lesson_47_application_runtime import (
    ApplicationRuntime,
)


SUPPORTED_QUESTION = (
    "What should a driver do when an engine warning "
    "light appears?"
)

UNSUPPORTED_QUESTION = (
    "What is the fleet insurance policy number?"
)


def print_answer(
    *,
    label: str,
    result: GuardedRAGResult,
) -> None:
    """Print one guarded-RAG result without hiding metadata."""

    print(label)
    print("-" * len(label))
    print(
        "Question:",
        result.question,
    )
    print(
        "Decision:",
        result.decision,
    )
    print(
        "Strategy:",
        result.answer_strategy,
    )
    print(
        "Top similarity:",
        f"{result.top_similarity:.4f}",
    )
    print(
        "Generation called:",
        result.generation_called,
    )
    print(
        "Generation input tokens:",
        result.generation_input_tokens,
    )
    print(
        "Generation output tokens:",
        result.generation_output_tokens,
    )
    print(
        "Answer:"
    )
    print(
        result.answer
    )
    print()


def main() -> None:
    """Demonstrate one shared Ollama client across RAG."""

    settings = get_settings()

    print("SHARED OLLAMA–RAG INTEGRATION")
    print("=============================")
    print()
    print(
        "Creating the configured application runtime..."
    )
    print()

    with ApplicationRuntime.from_settings(
        settings
    ) as runtime:
        report = runtime.validate()

        adapter = runtime.rag_adapter

        if adapter is None:
            raise RuntimeError(
                "The application runtime did not configure "
                "the shared Ollama RAG adapter."
            )

        if not runtime.rag_service.model_providers_configured:
            raise RuntimeError(
                "The guarded-RAG service did not activate "
                "the shared model providers."
            )

        print("RUNTIME VALIDATION")
        print("------------------")
        print(
            "Collection:",
            report.collection_name,
        )
        print(
            "Stored points:",
            report.stored_point_count,
        )
        print(
            "Generation model:",
            settings.generation_model,
        )
        print(
            "Embedding model:",
            settings.embedding_model,
        )
        print(
            "Shared providers configured:",
            (
                runtime.rag_service
                .model_providers_configured
            ),
        )
        print()

        print("CALL COUNTS BEFORE QUESTIONS")
        print("----------------------------")
        print(
            "Query embeddings:",
            adapter.query_embedding_calls,
        )
        print(
            "Generations:",
            adapter.generation_calls,
        )
        print()

        supported_result = (
            runtime.rag_service.answer(
                SUPPORTED_QUESTION
            )
        )

        print_answer(
            label="SUPPORTED QUESTION",
            result=supported_result,
        )

        print("CALL COUNTS AFTER SUPPORTED QUESTION")
        print("------------------------------------")
        print(
            "Query embeddings:",
            adapter.query_embedding_calls,
        )
        print(
            "Generations:",
            adapter.generation_calls,
        )
        print()

        generation_calls_before_abstention = (
            adapter.generation_calls
        )

        unsupported_result = (
            runtime.rag_service.answer(
                UNSUPPORTED_QUESTION
            )
        )

        print_answer(
            label="UNSUPPORTED QUESTION",
            result=unsupported_result,
        )

        print("CALL COUNTS AFTER UNSUPPORTED QUESTION")
        print("--------------------------------------")
        print(
            "Query embeddings:",
            adapter.query_embedding_calls,
        )
        print(
            "Generations:",
            adapter.generation_calls,
        )
        print()

        if (
            adapter.generation_calls
            != generation_calls_before_abstention
        ):
            raise RuntimeError(
                "The retrieval-abstention path called the "
                "generation model unexpectedly."
            )

        print("LIFECYCLE INSIDE CONTEXT")
        print("------------------------")
        print(
            "Runtime closed:",
            runtime.is_closed,
        )
        print(
            "RAG service closed:",
            runtime.rag_service.is_closed,
        )
        print(
            "Ollama client closed:",
            runtime.ollama_client.is_closed,
        )
        print()

    print("LIFECYCLE AFTER CONTEXT")
    print("-----------------------")
    print(
        "Runtime closed:",
        runtime.is_closed,
    )
    print()

    print("STATUS")
    print("------")
    print(
        "One ConfiguredOllamaClient generated query "
        "embeddings and grounded answers."
    )
    print(
        "Low-confidence retrieval still abstained before "
        "calling the generation model."
    )


if __name__ == "__main__":
    main()
