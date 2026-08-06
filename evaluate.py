#importing the neccassary files
#testing how well the system works
#Uses evaluation.json which contains the questions and expected answers

#firt thing I have to do is import the neccassary json library
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

#importing functions from rag system file
from rag_system import (
    load_documents,
    make_chunks,
    create_bm25,
    create_embeddings,
    bm25_search,
    dense_search,
    generate_answer
)

# Load the questions we will use to test the system
with open(
    "data/evaluation.json",
    encoding="utf-8"
) as f:
    questions = json.load(f)
# Load documents and prepare the RAG system
documents = load_documents()
# Split documents into smaller chunks
chunks = make_chunks(
    documents
)
print(
    "Chunks number:",
    len(chunks)
)
# Now we are going to make the BM25 search system
print("BM25 system")
bm25 = create_bm25(
    chunks
)
# Create document embeddings for dense search
embeddings = create_embeddings(
    chunks
)

def run_test(method):

    print("\n======================")
    #This is just for printing the retrieval method name
    print(method)
    print("======================")
    # counter for counting the number of correct answers
    correct = 0
    #testing each question in evaluation.json
    for q in questions:
        question = q["question"]
        print("\nQUESTION:")
        print(question)
        # Finding the best chunk using the selected method
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

        #Asking Llama to answer using the retrieved chunk
        answer = generate_answer(
            question,
            chunk
        )
        print("\nthe answer llama gave:")
        print(answer)
        print("\n what was actually expected:")
        print(
            q["answer"]
        )

        #Manually checking if the answer is correct
        result = input(
            "\nCorrect? type y or n and press enter"
        )

        if result.lower() == "y":
            correct += 1
    # using the a metric to evaluate. we chose accuracy
    accuracy = (
        correct / len(questions)
    ) * 100
    print("\nthe resultt:")
    print(
        method,
        "accuracy:",
        accuracy,
        "%"
    )
#Run evaluation for BM25
run_test("bm25")
# Run evaluation for Dense Retrieval
run_test("dense")