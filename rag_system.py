import os
import random
import numpy as np
import ollama
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

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

def loadDocs():
    docs = []
    for file in os.listdir("data/documents"):
    
        with open(
            "data/documents/" + file,
            encoding="utf-8"
        ) as f:
            docs.append(
                f.read()
            )
    return docs

def splitChunks(docs):
    chunks = []

    for docNum, doc in enumerate(docs):

        words = doc.split()

        for i in range(0, len(words), 250):
            chunkText = " ".join(
                words[i:i+250]
            )

            chunks.append({
                "chunkId": len(chunks),
                "docId": docNum,
                "source": "Document " + str(docNum),
                "text": chunkText
            })
    return chunks

def makeBM25(chunks):
    allWords = []
    for chunk in chunks:
        allWords.append(
            chunk["text"].split()
        )
    return BM25Okapi(allWords)

def searchBM25(model, chunks, question):
    
    scores = model.get_scores(
        question.split()
    )
    best = scores.argmax()
    return chunks[best]

encoder = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

def makeEmbeddings(chunks):
    texts = []
    
    for chunk in chunks:
        texts.append(
            chunk["text"]
        )
    
    vectors = encoder.encode(
        texts
    )
    return vectors

def searchDense(chunks, vectors, question):
    
    questionVector = encoder.encode(
        [question]
    )
    
    scores = cosine_similarity(
        questionVector,
        vectors
    )[0]

    best = scores.argmax()
    return chunks[best]

def getAnswer(question, chunk):
    
    prompt = f"""
Answer only using the context.
If the answer is not available,
say "I don't know".
the Context:
{chunk["text"]}
the Question:
{question}
"""
    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    answer = response["message"]["content"]

    return answer

# Loading and preparing the RAG system
docs = loadDocs()
# Splitting into chunks
chunks = splitChunks(docs)
# Creating the BM25 search
bm25 = makeBM25(chunks)
# Create the vectors for dense search
vectors = makeEmbeddings(chunks)
def askQuestion(question, type="dense"):
    if type == "bm25":
        chunk = searchBM25(
            bm25,
            chunks,
            question
        )
    else:
        chunk = searchDense(
            chunks,
            vectors,
            question
        )
    answer = getAnswer(
        question,
        chunk
    )
    return answer
    
# This part starts our RAG system as a chatbot
if __name__ == "__main__":
    
    # Keep asking questions until the user wants to stop
    while True:
        question = input(
            "\nQuestion: "
        )
        # if the user types exit then stop the program
        if question == "exit":
            break
        # Send the question to the RAG system and print the generated answer
        print(
            askQuestion(question)
        )
