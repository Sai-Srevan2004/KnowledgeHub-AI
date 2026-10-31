from knowledgehub_ai.retrieval.query_rewriter import rewrite_query


def main():
    questions = [
        "How should employees report a lost or stolen company laptop?",
        "Where may employees obtain software for company devices?",
        "What details and documents must an employee include when submitting an expense claim?",
    ]

    for question in questions:
        rewritten = rewrite_query(question)

        print("\n" + "=" * 80)
        print(f"ORIGINAL:  {question}")
        print(f"REWRITTEN: {rewritten}")


if __name__ == "__main__":
    main()