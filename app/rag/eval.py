from app.rag.vector_retriever import retrieve

test_cases = [
    {
        "query": "asyncio.gather 有什么作用？",
        "expected_source": "python_knowledge.txt"
    },
    {
        "query": "RAG 是什么？",
        "expected_source": "agent_knowledge.txt"
    },
    {
        "query": "Tool Calling 是什么？",
        "expected_source": "agent_knowledge.txt"
    }
]

def evaluate() -> None:
    hit = 0
    for case in test_cases:
        results = retrieve(
            case["query"],
            top_k=1,
            min_score=0.0
        )
        if not results:
            print("MISS:",case["query"])
            continue
        actual_source = (
            results[0].chunk.metadata["source"]
        )
        success = (
            actual_source == case["expected_source"]
        )
        if success:
            hit += 1
        print(case["query"],"→",actual_source,"√" if success else "×")
    accuracy = hit / len(test_cases)
    print(f"Top-1 Accuracy:{accuracy:.2%}")

if __name__ == "__main__":
    evaluate()