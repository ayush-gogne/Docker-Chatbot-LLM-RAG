import json
import os
import random
import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None


def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


set_seed(42)

from rag_system import (
    load_documents,
    make_chunks,
    create_bm25,
    create_embeddings,
    bm25_search,
    dense_search,
    generate_answer
)

with open(
    "data/evaluation.json",
    encoding="utf-8"
) as f:
    questions = json.load(f)

documents = load_documents()
chunks = make_chunks(
    documents
)
print(
    "Chunks number:",
    len(chunks)
)

# Creating the BM25 search system
print("BM25 system")
bm25 = create_bm25(
    chunks
)

# Creating the document embeddings for dense search
embeddings = create_embeddings(
    chunks
)

def run_test(method):

    print("\n======================")
    print(method)
    print("======================")
    correct = 0
    for q in questions:
        question = q["question"]
        print("\nQUESTION:")
        print(question)
        # Finding the best chunk using the selected BM25 method
        if method == "bm25":
            chunk = bm25_search(
                bm25,
                chunks,
                question
            )
        else:
            chunk = dense_search(
                chunks,
                embeddings,
                question
            )

        # Asking Llama to answer using the retrieved chunk
        answer = generate_answer(
            question,
            chunk
        )
        print("\n The answer llama gave:")
        print(answer)
        print("\n What was actually expected:")
        print(
            q["answer"]
        )

        # Checking if the answer is correct
        result = input(
            "\nCorrect? type y or n and press enter"
        )

        if result.lower() == "y":
            correct += 1
    # using the accuracy metric to evaluate
    accuracy = (
        correct / len(questions)
    ) * 100
    print("\nThe resultt:")
    print(
        method,
        "accuracy:",
        accuracy,
        "%"
    )
# Run evaluation for BM25
run_test("bm25")
# Run evaluation for Dense Retrieval
run_test("dense")
